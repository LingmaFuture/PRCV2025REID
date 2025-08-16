# train.py 
import os
import time
import math
import random
# 删除pickle导入，不再需要保存数据集划分
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

# 删除sklearn导入，不再需要训练集划分

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
# 注意：已删除训练集划分逻辑，使用全部训练数据
# ------------------------------


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
def train_multimodal_reid():
    # 配置与设备
    config = TrainingConfig()
    set_seed(getattr(config, "seed", 42))

    device_str = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    setup_logging(getattr(config, "log_dir", "./logs"))

    # 加载完整训练数据集，不进行划分
    logging.info("加载完整训练数据集...")
    train_dataset = MultiModalDataset(config, split='train')
    
    # 输出数据集统计
    all_person_ids = sorted(list(set([item['person_id'] for item in train_dataset.data_list])))
    logging.info(f"训练集: {len(all_person_ids)} 个身份, {len(train_dataset.data_list)} 个样本")
    
    # 不使用本地验证集，设为None
    local_val_dataset = None
    logging.info("已删除训练集划分逻辑，使用全部训练数据进行训练")

    # 更新类别数为全部身份数（确保标签映射正确）
    config.num_classes = len(all_person_ids)



    # 训练 DataLoader（Balanced PK 采样，直接用 train_dataset）
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

    # 不使用本地验证集，设置为None
    gallery_loader, query_loaders = None, None
    logging.info("跳过本地验证数据加载器构建")


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
    
    # 早停参数 - 基于多指标组合判断
    patience = getattr(config, "patience", 15)  # 15轮不改善就停止
    patience_counter = 0
    best_combined_score = float('-inf')  # 组合分数，越高越好
    
    # 早停策略：平衡训练效果和过拟合风险
    loss_weight = 0.6  # 训练损失权重
    acc_weight = 0.3   # 分类准确率权重  
    stability_weight = 0.1  # 训练稳定性权重

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Warmup学习率调整
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr * warmup_factor

        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, scaler)

        # 记录训练历史
        train_history.append({'epoch': epoch, **train_metrics})
        
        # 计算组合评估分数（防止过拟合）
        current_loss = train_metrics['total_loss']
        current_acc = train_metrics['accuracy']
        
        # 计算训练稳定性（基于最近几轮的损失方差）
        recent_losses = [train_history[i]['total_loss'] for i in range(max(0, len(train_history)-4), len(train_history))]
        stability_score = 1.0 / (1.0 + np.var(recent_losses)) if len(recent_losses) > 1 else 1.0
        
        # 组合分数：平衡训练效果和稳定性
        combined_score = (
            loss_weight * (1.0 / (1.0 + current_loss)) +  # 损失越低越好，转换为正向分数
            acc_weight * (current_acc / 100.0) +           # 准确率越高越好
            stability_weight * stability_score              # 稳定性越好越好
        )
        
        # 检查是否是最佳模型
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_loss = current_loss  # 保持兼容性
            patience_counter = 0
            # 保存最佳模型
            save_checkpoint(
                model, optimizer, scheduler, epoch, current_loss, config,
                os.path.join(save_dir, 'best_model.pth')
            )
            logging.info(f"新的最佳组合分数: {combined_score:.4f} (Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%)，已保存最佳模型")
        else:
            patience_counter += 1
        
        # 定期保存检查点
        save_freq = getattr(config, "save_freq", 20)
        if epoch % save_freq == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, current_loss, config,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            )
            logging.info(f"已保存第{epoch}轮检查点")

        # 过拟合检测
        overfitting_warning = ""
        if epoch > 10:  # 前10轮跳过检测
            recent_acc = [train_history[i]['accuracy'] for i in range(max(0, len(train_history)-5), len(train_history))]
            if len(recent_acc) >= 5 and current_acc > 95.0 and np.std(recent_acc) < 0.1:
                overfitting_warning = " [可能过拟合]"
        
        logging.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_metrics['total_loss']:.4f} - "
            f"Train ClsAcc: {train_metrics['accuracy']:.2f}% - "
            f"ConLoss: {train_metrics['contrastive_loss']:.4f} - "
            f"CombScore: {combined_score:.4f} - "
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
            logging.info(f"组合评估分数在 {patience} 轮内没有改善，触发早停")
            logging.info(f"最佳组合分数: {best_combined_score:.4f}，对应训练损失: {best_loss:.4f}")
            break

    # 训练完成，保存最终模型
    logging.info("训练完成，保存最终模型...")
    save_checkpoint(
        model, optimizer, scheduler, num_epochs, 0.0, config,
        os.path.join(save_dir, 'final_model.pth')
    )

    # 保存训练历史
    log_dir = getattr(config, "log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    pd.DataFrame(train_history).to_csv(os.path.join(log_dir, 'train_history.csv'), index=False)

    logging.info(f"训练完成. 最终模型已保存到: {os.path.join(save_dir, 'final_model.pth')}")


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


if __name__ == "__main__":
    train_multimodal_reid()
