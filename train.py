# train.py 
import os
import time
import math
import random
# 添加pickle导入，用于保存验证集划分
import pickle
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR
from torch.amp import autocast, GradScaler

# 添加sklearn导入，用于验证集划分
from sklearn.model_selection import train_test_split

# === 你项目里的数据与模型 ===
from datasets.dataset import MultiModalDataset, BalancedBatchSampler, compatible_collate_fn
from models.model import MultiModalReIDModel  
from configs.config import TrainingConfig


# ------------------------------
# 实用工具
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ViT + 数据增广场景下一般不强制 deterministic，benchmark=True 更快
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )


def move_batch_to_device(batch, device):
    """将批次数据移动到指定设备（仅移动 Tensor）"""
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            out[k] = move_batch_to_device(v, device)
        return out
    elif isinstance(batch, list):
        return [move_batch_to_device(x, device) for x in batch]
    elif isinstance(batch, tuple):
        return tuple(move_batch_to_device(x, device) for x in batch)
    elif torch.is_tensor(batch):
        return batch.to(device)
    else:
        return batch


# ------------------------------
# 评估指标（修正版）
# ------------------------------
def compute_map(query_features, gallery_features, query_labels, gallery_labels, k=100):
    """mAP@k（稳定写法）"""
    query_features = F.normalize(query_features, p=2, dim=1)
    gallery_features = F.normalize(gallery_features, p=2, dim=1)
    sim = torch.mm(query_features, gallery_features.t())  # (Q, G)

    aps = []
    for i in range(sim.size(0)):
        scores = sim[i]
        qy = query_labels[i]
        _, idx = torch.sort(scores, descending=True)
        ranked = gallery_labels[idx[:k]]  # (k,)

        matches = (ranked == qy).float()
        if matches.sum() > 0:
            cum_matches = torch.cumsum(matches, dim=0)
            ranks = torch.arange(1, matches.numel() + 1, device=matches.device, dtype=matches.dtype)
            precision = cum_matches / ranks
            ap = precision[matches.bool()].mean().item()
            aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def compute_cmc(query_features, gallery_features, query_labels, gallery_labels, k=10):
    """CMC@k（稳定写法）"""
    query_features = F.normalize(query_features, p=2, dim=1)
    gallery_features = F.normalize(gallery_features, p=2, dim=1)
    sim = torch.mm(query_features, gallery_features.t())
    correct = 0
    for i in range(sim.size(0)):
        _, idx = torch.sort(sim[i], descending=True)
        topk_labels = gallery_labels[idx[:k]]
        correct += (topk_labels == query_labels[i]).any().item()
    return correct / sim.size(0) if sim.size(0) > 0 else 0.0


# ------------------------------
# 阶段性训练策略：
# Stage 1: 80%训练 + 20%验证，用于模型选择和超参数调优
# Stage 2: 100%训练数据，训练最终提交模型
# ------------------------------

def split_train_val_by_identity(dataset, val_ratio=0.2, seed=42):
    """
    按身份划分训练集和验证集，确保同一身份不会同时出现在训练集和验证集中
    """
    # 获取所有身份ID
    all_person_ids = sorted(list(set([item['person_id'] for item in dataset.data_list])))
    
    # 按身份划分
    train_ids, val_ids = train_test_split(
        all_person_ids, 
        test_size=val_ratio, 
        random_state=seed,
        shuffle=True
    )
    
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)
    
    # 根据身份划分样本
    train_indices = []
    val_indices = []
    
    for idx, item in enumerate(dataset.data_list):
        person_id = item['person_id']
        if person_id in train_ids_set:
            train_indices.append(idx)
        elif person_id in val_ids_set:
            val_indices.append(idx)
    
    logging.info(f"训练身份数: {len(train_ids)}, 验证身份数: {len(val_ids)}")
    logging.info(f"训练样本数: {len(train_indices)}, 验证样本数: {len(val_indices)}")
    
    return train_indices, val_indices, train_ids, val_ids


# ------------------------------
# 赛制对齐：验证集构建（RGB=gallery，其它模态=queries）
# 假设：dataset[i] 返回：
#   {
#     'person_id': Tensor[1] or int,
#     'images': {'vis': Tensor[C,H,W], 'nir': ..., 'sk': ..., 'cp': ...}  # 仅存在的模态会出现
#     'text_description': str 或 list[str]
#   }
# ------------------------------
MODALITIES = ['vis', 'nir', 'sk', 'cp', 'text']


def _peek_modalities_of_index(dataset, idx) -> Dict[str, bool]:
    """
    只访问一次 __getitem__ 来判断该身份有哪些模态可用
    返回：{'vis': True/False, 'nir': ..., 'sk': ..., 'cp': ..., 'text': True/False}
    """
    rec = dataset[idx]
    has = {m: False for m in MODALITIES}
    # 图像模态
    imgs = rec.get('images', {})
    for m in ['vis', 'nir', 'sk', 'cp']:
        if isinstance(imgs, dict) and m in imgs and torch.is_tensor(imgs[m]):
            has[m] = True
    # 文本
    td = rec.get('text_description', "")
    if isinstance(td, list):
        has['text'] = len(td) > 0 and isinstance(td[0], str) and len(td[0]) > 0
    elif isinstance(td, str):
        has['text'] = len(td) > 0
    else:
        has['text'] = False
    return has


def build_val_presence_table(dataset, val_indices):
    presence = {}
    for idx in val_indices:
        entry = dataset.data_list[idx]
        pid_str = entry['person_id_str']
        has = {m: False for m in ['vis','nir','sk','cp','text']}

        # 图像模态：看缓存里有没有文件路径
        cache = dataset.image_cache.get(pid_str, {})
        for m in ['vis','nir','sk','cp']:
            has[m] = len(cache.get(m, [])) > 0

        # 文本：看 data_list 里的合并描述是否非空
        td = entry.get('text_description', '')
        has['text'] = isinstance(td, str) and len(td) > 0

        presence[idx] = has
    return presence



class GalleryOnlyVIS(Dataset):
    """画廊只保留可见光（vis）的 wrapper"""
    def __init__(self, base_dataset: Dataset, indices: List[int], presence: Dict[int, Dict[str, bool]]):
        self.base = base_dataset
        # 只保留有 vis 的身份
        self.indices = [i for i in indices if presence.get(i, {}).get('vis', False)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        rec = self.base[base_idx]  # dict
        # 仅保留 vis
        images = {}
        if 'images' in rec and 'vis' in rec['images']:
            images['vis'] = rec['images']['vis']
        pid = rec['person_id']
        if torch.is_tensor(pid):
            pid = int(pid.item())

        # 文本在画廊阶段不需要
        return {
            'person_id': torch.tensor(pid, dtype=torch.long),
            'images': images,                  # 仅 vis
            'text_description': [""],          # 空文本（不会被编码，因为 mask 会禁用）
            'modality_mask': {
                'vis': True, 'nir': False, 'sk': False, 'cp': False, 'text': False
            }
        }


class CombinationQueryDataset(Dataset):
    """
    查询集合：对每个身份 index，选择指定的模态组合（不含 vis），把其它模态过滤掉
    q_items: List[{'idx': int, 'pid': int, 'modalities': List[str]}]
    """
    def __init__(self, base_dataset: Dataset, q_items: List[Dict]):
        self.base = base_dataset
        self.items = q_items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        info = self.items[i]
        base_idx = info['idx']
        mods = info['modalities']
        rec = self.base[base_idx]

        # 过滤 images
        images = {}
        if 'images' in rec and isinstance(rec['images'], dict):
            for m in mods:
                if m in ['nir', 'sk', 'cp'] and m in rec['images'] and torch.is_tensor(rec['images'][m]):
                    images[m] = rec['images'][m]

        # 文本
        text_desc = ""
        if 'text' in mods:
            td = rec.get('text_description', "")
            if isinstance(td, list):
                text_desc = td[0] if len(td) > 0 else ""
            elif isinstance(td, str):
                text_desc = td

        # 关键：label 直接用底层样本的映射后 ID
        pid_t = rec['person_id']
        pid = int(pid_t.item()) if torch.is_tensor(pid_t) else int(pid_t)

        # 完整的modality_mask，确保所有模态都有明确的True/False设置
        modality_mask = {'vis': False, 'nir': False, 'sk': False, 'cp': False, 'text': False}
        for m in mods:
            modality_mask[m] = True
            
        return {
            'person_id': torch.tensor(pid, dtype=torch.long),
            'images': images,
            'text_description': [text_desc],
            'modality_mask': modality_mask
        }


def build_eval_loaders_by_rule(dataset, val_indices, batch_size, num_workers, pin_memory):
    """
    根据赛制构建：
      - gallery_loader：只含 vis 的画廊
      - query_loaders ：{'single':{comb: loader}, 'double':{...}, 'triple':{...}, 'quad':{...}}
    """
    presence = build_val_presence_table(dataset, val_indices)

    # 画廊
    gal_ds = GalleryOnlyVIS(dataset, val_indices, presence)
    gallery_loader = DataLoader(
        gal_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=compatible_collate_fn
    )

    # 查询组合（非 vis）
    non_vis = ['nir', 'sk', 'cp', 'text']

    def _make_items(group: List[str]) -> List[Dict]:
        items = []
        for idx in val_indices:
            has = presence[idx]
            if all(has.get(m, False) for m in group):
                items.append({'idx': idx,  'modalities': group})
        return items

    query_loaders = {'single': {}, 'double': {}, 'triple': {}, 'quad': {}}

    # 单模态
    for m in non_vis:
        items = _make_items([m])
        if items:
            loader = DataLoader(
                CombinationQueryDataset(dataset, items),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=compatible_collate_fn
            )
            query_loaders['single'][m] = loader

    # 双/三/四模态
    import itertools
    for gsize, tag in [(2, 'double'), (3, 'triple'), (4, 'quad')]:
        for comb in itertools.combinations(non_vis, gsize):
            group = list(comb)
            key = '+'.join(group)
            items = _make_items(group)
            if items:
                loader = DataLoader(
                    CombinationQueryDataset(dataset, items),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=compatible_collate_fn
                )
                query_loaders[tag][key] = loader

    return gallery_loader, query_loaders


def validate_competition_style(model, gallery_loader, query_loaders, device, k_map=100, sample_ratio=1.0):
    """
    赛制对齐评测：
      - 用同一 RGB 画廊
      - 对各类（single/double/triple/quad）组合分别算 mAP，然后四类平均
      - 额外汇报合并查询后的 CMC@1/5/10（非赛制指标，仅供参考）
      - sample_ratio: 采样比例，用于快速评估 (0.3表示只用30%数据)
    """
    model.eval()

    # 画廊特征 (支持采样)
    with torch.no_grad():
        gal_feats, gal_labels = [], []
        total_batches = len(gallery_loader)
        sample_batches = max(1, int(total_batches * sample_ratio))
        
        batch_indices = random.sample(range(total_batches), sample_batches) if sample_ratio < 1.0 else range(total_batches)
        
        for i, batch in enumerate(tqdm(gallery_loader, desc=f'提取画廊特征(采样{sample_ratio:.1f})')):
            if sample_ratio < 1.0 and i not in batch_indices:
                continue
            batch = move_batch_to_device(batch, device)
            with autocast(device_type='cuda', enabled=device.type == 'cuda'):
                feats = model(batch, return_features=True)
            labels = batch['person_id']
            gal_feats.append(feats.cpu()); gal_labels.append(labels.cpu())
        gal_feats = torch.cat(gal_feats, dim=0)
        gal_labels = torch.cat(gal_labels, dim=0)

    detail = {'single': {}, 'double': {}, 'triple': {}, 'quad': {}}
    buckets = {'single': [], 'double': [], 'triple': [], 'quad': []}
    all_q_feats, all_q_labels = [], []

    with torch.no_grad():
        for tag, group in query_loaders.items():
            for key, qloader in group.items():
                qf, ql = [], []
                total_batches = len(qloader)
                sample_batches = max(1, int(total_batches * sample_ratio))
                batch_indices = random.sample(range(total_batches), sample_batches) if sample_ratio < 1.0 else range(total_batches)
                
                for i, batch in enumerate(tqdm(qloader, desc=f'提取查询特征[{tag}:{key}](采样{sample_ratio:.1f})')):
                    if sample_ratio < 1.0 and i not in batch_indices:
                        continue
                    batch = move_batch_to_device(batch, device)
                    with autocast(device_type='cuda', enabled=device.type == 'cuda'):
                        feats = model(batch, return_features=True)
                    labels = batch['person_id']
                    qf.append(feats.cpu()); ql.append(labels.cpu())
                if not qf:
                    continue
                qf = torch.cat(qf, dim=0); ql = torch.cat(ql, dim=0)
                km = min(k_map, gal_feats.size(0))
                m = compute_map(qf, gal_feats, ql, gal_labels, k=km)

                detail[tag][key] = float(m)
                buckets[tag].append(m)
                all_q_feats.append(qf); all_q_labels.append(ql)

    def _avg(x): return float(np.mean(x)) if x else 0.0

    map_single = _avg(buckets['single'])
    map_double = _avg(buckets['double'])
    map_triple = _avg(buckets['triple'])
    map_quad   = _avg(buckets['quad'])
    map_avg4   = float(np.mean([map_single, map_double, map_triple, map_quad]))

    if all_q_feats:
        all_q_feats = torch.cat(all_q_feats, dim=0)
        all_q_labels = torch.cat(all_q_labels, dim=0)
        cmc1 = compute_cmc(all_q_feats, gal_feats, all_q_labels, gal_labels, k=1)
        cmc5 = compute_cmc(all_q_feats, gal_feats, all_q_labels, gal_labels, k=5)
        cmc10 = compute_cmc(all_q_feats, gal_feats, all_q_labels, gal_labels, k=10)
    else:
        cmc1 = cmc5 = cmc10 = 0.0

    return {
        'map_single': map_single,
        'map_double': map_double,
        'map_triple': map_triple,
        'map_quad': map_quad,
        'map_avg4': map_avg4,
        'detail': detail,
        'cmc1': cmc1, 'cmc5': cmc5, 'cmc10': cmc10
    }


# ------------------------------
# 训练一个 epoch（使用模型自带损失）
# ------------------------------
def train_epoch(model, dataloader, optimizer, device, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    ce_loss_sum = 0.0
    contrastive_loss_sum = 0.0
    correct = 0
    total = 0
    
    # 新增：特征范数统计
    feature_norms = []
    
    use_amp = scaler is not None

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device)
        labels = batch['person_id']

        optimizer.zero_grad()
        
        # 混合精度前向传播
        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model(batch)
            loss_dict = model.compute_loss(outputs, labels)
            loss = loss_dict['total_loss']
        
        ce_loss = loss_dict.get('ce_loss', torch.tensor(0.0, device=device))
        cont_loss = loss_dict.get('contrastive_loss', torch.tensor(0.0, device=device))

        # 混合精度反向传播
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        ce_loss_sum += ce_loss.item()
        contrastive_loss_sum += cont_loss.item()

        # 分类准确率（仅作训练监控）
        if isinstance(outputs, dict) and 'logits' in outputs:
            _, predicted = outputs['logits'].max(1)
        else:
            _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 特征质量评估：同类样本的特征相似度（更贴近检索任务）
        if isinstance(outputs, dict) and 'reid_features' in outputs and batch_idx % 10 == 0:
            reid_features = outputs['reid_features']  # 已归一化的特征
            batch_size = reid_features.size(0)
            
            # 计算批内相似度矩阵
            similarity_matrix = torch.mm(reid_features, reid_features.t())
            
            # 计算同类样本的平均相似度
            same_id_similarities = []
            for i in range(batch_size):
                for j in range(i+1, batch_size):
                    if labels[i] == labels[j]:
                        same_id_similarities.append(similarity_matrix[i, j].item())
            
            if same_id_similarities:
                avg_same_id_sim = np.mean(same_id_similarities)
                feature_norms.append(avg_same_id_sim)  # 复用feature_norms列表存储
        
        # 新增：特征范数监控
        if isinstance(outputs, dict) and 'reid_features_raw' in outputs:
            norms = torch.norm(outputs['reid_features_raw'].detach(), p=2, dim=1)
            feature_norms.extend(norms.cpu().numpy())
        elif isinstance(outputs, dict) and 'features' in outputs:
            norms = torch.norm(outputs['features'].detach(), p=2, dim=1)
            feature_norms.extend(norms.cpu().numpy())


        # 计算平均特征范数
        avg_norm = np.mean(feature_norms) if feature_norms else 0.0

        pbar.set_postfix({
            'Loss': f'{loss.item():.3f}',
            'CE': f'{float(ce_loss.item()):.3f}',
            'Cont': f'{float(cont_loss.item()):.3f}',
            'FeatNorm': f'{avg_norm:.3f}'
        })

    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = 100. * correct / max(1, total)
    avg_feat_norm = np.mean(feature_norms) if feature_norms else 0.0
    
    return {
        'total_loss': avg_loss,
        'ce_loss': ce_loss_sum / max(1, len(dataloader)),
        'contrastive_loss': contrastive_loss_sum / max(1, len(dataloader)),
        'accuracy': accuracy,
        'feature_norm': avg_feat_norm
    }


# ------------------------------
# 训练主流程
# ------------------------------
def train_multimodal_reid(stage: str = "stage1"):
    """
    阶段性训练函数
    stage1: 用80%训练+20%验证进行模型选择和超参数调优
    stage2: 用100%训练数据训练最终模型
    """
    # 配置与设备
    config = TrainingConfig()
    set_seed(getattr(config, "seed", 42))

    device_str = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    setup_logging(getattr(config, "log_dir", "./logs"))

    # 加载完整训练数据集
    logging.info("加载完整训练数据集...")
    full_dataset = MultiModalDataset(config, split='train')
    
    # 输出数据集统计
    all_person_ids = sorted(list(set([item['person_id'] for item in full_dataset.data_list])))
    logging.info(f"完整数据集: {len(all_person_ids)} 个身份, {len(full_dataset.data_list)} 个样本")
    
    # 根据训练阶段决定数据划分
    if stage == "stage1":
        logging.info("=== Stage 1: 模型选择和超参数调优阶段 ===")
        
        # 检查是否存在已保存的划分
        split_file = os.path.join(config.log_dir, "train_val_split.pkl")
        if os.path.exists(split_file):
            logging.info("加载已保存的训练/验证集划分...")
            with open(split_file, 'rb') as f:
                split_data = pickle.load(f)
                train_indices = split_data['train_indices']
                val_indices = split_data['val_indices']
                train_ids = split_data['train_ids']
                val_ids = split_data['val_ids']
        else:
            logging.info("创建新的训练/验证集划分...")
            train_indices, val_indices, train_ids, val_ids = split_train_val_by_identity(
                full_dataset, val_ratio=config.val_ratio, seed=config.seed
            )
            
            # 保存划分结果
            split_data = {
                'train_indices': train_indices,
                'val_indices': val_indices, 
                'train_ids': train_ids,
                'val_ids': val_ids
            }
            with open(split_file, 'wb') as f:
                pickle.dump(split_data, f)
            logging.info(f"训练/验证集划分已保存至: {split_file}")
        
        # 创建训练和验证子集
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        # 更新类别数为训练集身份数
        config.num_classes = len(train_ids)
        
        logging.info(f"Stage 1 - 训练身份: {len(train_ids)}, 验证身份: {len(val_ids)}")
        logging.info(f"Stage 1 - 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
        
    elif stage == "stage2":
        logging.info("=== Stage 2: 最终模型训练阶段 ===")
        
        # 使用全部训练数据
        train_dataset = full_dataset
        val_dataset = None
        
        # 更新类别数为全部身份数
        config.num_classes = len(all_person_ids)
        
        logging.info(f"Stage 2 - 使用全部训练数据: {len(all_person_ids)} 个身份, {len(train_dataset)} 个样本")
        
    else:
        raise ValueError(f"未知的训练阶段: {stage}")

    logging.info(f"模型类别数: {config.num_classes}")



    # 训练 DataLoader（Balanced PK 采样）
    num_instances = 4
    effective_batch_size = max(getattr(config, "batch_size", 32), 16)
    effective_batch_size = max(num_instances, (effective_batch_size // num_instances) * num_instances)

    train_sampler = BalancedBatchSampler(train_dataset, effective_batch_size, num_instances=num_instances)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=getattr(config, "num_workers", 4),
        pin_memory=getattr(config, "pin_memory", True),
        collate_fn=compatible_collate_fn
    )

    # 验证集DataLoader创建（仅Stage 1）
    if stage == "stage1" and val_dataset is not None:
        logging.info("构建验证集评估数据加载器...")
        gallery_loader, query_loaders = build_eval_loaders_by_rule(
            full_dataset,  # 使用原始数据集获取模态信息
            val_indices,   # 验证集索引
            batch_size=getattr(config, "inference_batch_size", 32),
            num_workers=getattr(config, "num_workers", 4),
            pin_memory=getattr(config, "pin_memory", True)
        )
        logging.info("验证集数据加载器构建完成")
    else:
        gallery_loader, query_loaders = None, None
        if stage == "stage1":
            logging.info("警告: Stage 1 但无验证集")
        else:
            logging.info("Stage 2: 跳过验证集构建")


    # 模型
    model = MultiModalReIDModel(config).to(device)

    # 优化器
    optimizer = AdamW(model.parameters(),
                      lr=getattr(config, "learning_rate", 3e-4),
                      weight_decay=getattr(config, "weight_decay", 1e-4))
    
    # 混合精度训练
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        logging.info("启用混合精度训练 (AMP)")
    else:
        logging.info("使用全精度训练")

    # 学习率调度器（含warmup）
    warmup_epochs = getattr(config, "warmup_epochs", 5)
    total_epochs = getattr(config, "num_epochs", 100)
    scheduler_type = getattr(config, "scheduler", "cosine")
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    elif scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=[40, 70, 90], gamma=0.1)
    else:
        scheduler = None
    
    # 记录初始学习率
    initial_lr = optimizer.param_groups[0]['lr']
    logging.info(f"初始学习率: {initial_lr}, Warmup轮数: {warmup_epochs}")

    # 训练循环
    best_loss = float('inf')
    train_history = []
    num_epochs = getattr(config, "num_epochs", 100)
    eval_freq = getattr(config, "eval_freq", 1)
    save_dir = getattr(config, "save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    # 早停参数 - 根据训练阶段调整策略
    patience = getattr(config, "patience", 15)  # 15轮不改善就停止
    patience_counter = 0
    best_score = float('-inf')  # 最佳分数（Stage 1: 验证mAP, Stage 2: 组合分数）
    best_val_map = 0.0  # Stage 1 专用：最佳验证mAP
    
    if stage == "stage1":
        # Stage 1: 主要基于验证集mAP进行早停
        logging.info("Stage 1 早停策略: 基于验证集mAP@100")
        eval_freq = getattr(config, "eval_freq", 10)  # 每10轮评估一次
    else:
        # Stage 2: 基于训练指标组合进行早停（无验证集）
        logging.info("Stage 2 早停策略: 基于训练指标组合")
        eval_freq = float('inf')  # Stage 2不进行验证集评估
        
        # 早停策略权重：平衡训练效果和过拟合风险
        loss_weight = 0.5      # 训练损失权重
        acc_weight = 0.2       # 分类准确率权重
        contrastive_weight = 0.2  # 对比损失权重
        stability_weight = 0.1    # 训练稳定性权重

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Warmup学习率调整
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr * warmup_factor

        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, scaler)

        # 记录训练历史
        epoch_data = {'epoch': epoch, **train_metrics}
        
        # 验证集评估（仅Stage 1）
        if stage == "stage1" and gallery_loader is not None and epoch % eval_freq == 0:
            logging.info(f"Epoch {epoch}: 开始验证集评估...")
            
            val_results = validate_competition_style(
                model, gallery_loader, query_loaders, device, 
                k_map=100, sample_ratio=0.5  # 使用50%采样加速验证
            )
            
            val_map = val_results['map_avg4']
            epoch_data.update({
                'val_map_avg4': val_map,
                'val_map_single': val_results['map_single'],
                'val_map_double': val_results['map_double'], 
                'val_map_triple': val_results['map_triple'],
                'val_map_quad': val_results['map_quad'],
                'val_cmc1': val_results['cmc1']
            })
            
            logging.info(f"验证结果 - mAP@100: {val_map:.4f}, CMC@1: {val_results['cmc1']:.4f}")
            
            # Stage 1早停：基于验证mAP
            if val_map > best_val_map:
                best_val_map = val_map
                best_score = val_map
                patience_counter = 0
                
                # 保存最佳模型
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_map, config,
                    os.path.join(save_dir, 'best_model_stage1.pth')
                )
                logging.info(f"新的最佳验证mAP: {val_map:.4f}，已保存Stage1最佳模型")
            else:
                patience_counter += 1
                
        elif stage == "stage2":
            # Stage 2早停：基于训练指标组合
            current_loss = train_metrics['total_loss']
            current_acc = train_metrics['accuracy']
            current_contrastive = train_metrics['contrastive_loss']
            
            # 计算训练稳定性
            recent_losses = [train_history[i]['total_loss'] for i in range(max(0, len(train_history)-4), len(train_history))]
            stability_score = 1.0 / (1.0 + np.var(recent_losses)) if len(recent_losses) > 1 else 1.0
            
            # 组合分数
            combined_score = (
                loss_weight * (1.0 / (1.0 + current_loss)) +
                acc_weight * (current_acc / 100.0) +
                contrastive_weight * (1.0 / (1.0 + current_contrastive)) +
                stability_weight * stability_score
            )
            
            epoch_data['combined_score'] = combined_score
            
            if combined_score > best_score:
                best_score = combined_score
                patience_counter = 0
                
                # 保存最佳模型
                save_checkpoint(
                    model, optimizer, scheduler, epoch, combined_score, config,
                    os.path.join(save_dir, 'best_model_stage2.pth')
                )
                logging.info(f"新的最佳组合分数: {combined_score:.4f}，已保存Stage2最佳模型")
            else:
                patience_counter += 1
        
        train_history.append(epoch_data)
        
        # 定期保存检查点
        save_freq = getattr(config, "save_freq", 20)
        if epoch % save_freq == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_metrics['total_loss'], config,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch}_{stage}.pth')
            )
            logging.info(f"已保存第{epoch}轮检查点 ({stage})")

        # 过拟合检测（基于训练准确率）
        overfitting_warning = ""
        if epoch > 10:  # 前10轮跳过检测
            recent_acc = [train_history[i]['accuracy'] for i in range(max(0, len(train_history)-5), len(train_history))]
            if len(recent_acc) >= 5 and train_metrics['accuracy'] > 95.0 and np.std(recent_acc) < 0.1:
                overfitting_warning = " [可能过拟合]"
        
        # 根据训练阶段调整日志输出
        if stage == "stage1":
            val_info = ""
            if 'val_map_avg4' in epoch_data:
                val_info = f" - Val mAP: {epoch_data['val_map_avg4']:.4f}"
            
            logging.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f} - "
                f"Train ClsAcc: {train_metrics['accuracy']:.2f}% - "
                f"ConLoss: {train_metrics['contrastive_loss']:.4f}"
                f"{val_info} - "
                f"用时: {time.time() - start_time:.2f}s{overfitting_warning}"
            )
        else:  # stage2
            logging.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f} - "
                f"Train ClsAcc: {train_metrics['accuracy']:.2f}% - "
                f"ConLoss: {train_metrics['contrastive_loss']:.4f} - "
                f"CombScore: {epoch_data.get('combined_score', 0):.4f} - "
                f"用时: {time.time() - start_time:.2f}s{overfitting_warning}"
            )

        # 只在warmup结束后使用学习率调度器
        if scheduler and epoch > warmup_epochs:
            scheduler.step()
            
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        if epoch <= 5 or epoch % 10 == 0:  # 前5轮和每10轮记录学习率
            logging.info(f"Epoch {epoch} - 当前学习率: {current_lr:.2e}")
        
        # 早停检查
        if patience_counter >= patience:
            if stage == "stage1":
                logging.info(f"验证mAP在 {patience} 轮内没有改善，触发早停")
                logging.info(f"最佳验证mAP: {best_val_map:.4f}")
            else:
                logging.info(f"组合评估分数在 {patience} 轮内没有改善，触发早停")
                logging.info(f"最佳组合分数: {best_score:.4f}")
            break

    # 训练完成，保存最终模型
    logging.info(f"{stage.upper()} 训练完成，保存最终模型...")
    save_checkpoint(
        model, optimizer, scheduler, epoch, best_score, config,
        os.path.join(save_dir, f'final_model_{stage}.pth')
    )

    # 保存训练历史
    log_dir = getattr(config, "log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    pd.DataFrame(train_history).to_csv(os.path.join(log_dir, f'train_history_{stage}.csv'), index=False)

    if stage == "stage1":
        logging.info(f"Stage 1 完成. 最佳验证mAP: {best_val_map:.4f}")
        logging.info(f"最佳模型已保存到: {os.path.join(save_dir, 'best_model_stage1.pth')}")
        return best_val_map  # 返回最佳验证性能
    else:
        logging.info(f"Stage 2 完成. 最佳组合分数: {best_score:.4f}")
        logging.info(f"最终模型已保存到: {os.path.join(save_dir, 'best_model_stage2.pth')}")
        return best_score


def save_checkpoint(model, optimizer, scheduler, epoch, best_map, config, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_map': best_map,
        'config': config.__dict__ if hasattr(config, '__dict__') else str(config)
    }
    torch.save(checkpoint, filename)


def main():
    """主函数：执行阶段性训练"""
    import argparse
    
    parser = argparse.ArgumentParser(description='阶段性多模态ReID训练')
    parser.add_argument('--stage', type=str, choices=['stage1', 'stage2', 'both'], 
                       default='both', help='训练阶段选择')
    parser.add_argument('--stage1_model', type=str, default=None,
                       help='Stage2使用的Stage1最佳模型路径')
    
    args = parser.parse_args()
    
    if args.stage == 'stage1':
        logging.info("=== 仅执行 Stage 1 训练 ===")
        best_val_map = train_multimodal_reid('stage1')
        logging.info(f"Stage 1 最佳验证mAP: {best_val_map:.4f}")
        
    elif args.stage == 'stage2':
        logging.info("=== 仅执行 Stage 2 训练 ===")
        if args.stage1_model:
            logging.info(f"将从Stage1模型继续训练: {args.stage1_model}")
        best_score = train_multimodal_reid('stage2')
        logging.info(f"Stage 2 最佳组合分数: {best_score:.4f}")
        
    else:  # both
        logging.info("=== 执行完整的阶段性训练 ===")
        
        # Stage 1: 模型选择和超参数调优
        logging.info("开始 Stage 1...")
        best_val_map = train_multimodal_reid('stage1')
        logging.info(f"Stage 1 完成，最佳验证mAP: {best_val_map:.4f}")
        
        # Stage 2: 最终模型训练
        logging.info("开始 Stage 2...")
        best_score = train_multimodal_reid('stage2')
        logging.info(f"Stage 2 完成，最佳组合分数: {best_score:.4f}")
        
        logging.info("=== 阶段性训练全部完成 ===")
        logging.info("最终提交模型: ./checkpoints/best_model_stage2.pth")


if __name__ == "__main__":
    main()
