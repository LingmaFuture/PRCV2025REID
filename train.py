# train.py
import os
import time
import math
import random
import pickle
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, LambdaLR, ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

# === 你项目里的数据与模型 ===
from datasets.dataset import MultiModalDataset, BalancedBatchSampler, compatible_collate_fn
from models.model import CLIPBasedMultiModalReIDModel  
from configs.config import TrainingConfig

# ------------------------------
# 实用工具
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # 加速优化：启用TF32和优化CUDA操作
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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

def _sanitize_grads(model):
    """将非有限梯度置零，避免 GradNorm=nan/inf 传播"""
    any_bad = False
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        bad = ~torch.isfinite(g)
        if bad.any():
            g[bad] = 0.0
            any_bad = True
    return any_bad

# ------------------------------
# 评估指标
# ------------------------------
def compute_map(query_features, gallery_features, query_labels, gallery_labels, k=100):
    """mAP@k"""
    query_features = F.normalize(query_features, p=2, dim=1)
    gallery_features = F.normalize(gallery_features, p=2, dim=1)
    sim = torch.mm(query_features, gallery_features.t())  # (Q, G)

    aps = []
    for i in range(sim.size(0)):
        scores = sim[i]
        qy = query_labels[i]
        _, idx = torch.sort(scores, descending=True)
        ranked = gallery_labels[idx[:k]]

        matches = (ranked == qy).float()
        if matches.sum() > 0:
            cum_matches = torch.cumsum(matches, dim=0)
            ranks = torch.arange(1, matches.numel() + 1, device=matches.device, dtype=matches.dtype)
            precision = cum_matches / ranks
            ap = precision[matches.bool()].mean().item()
            aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0

def compute_cmc(query_features, gallery_features, query_labels, gallery_labels, k=10):
    """CMC@k"""
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
# 划分
# ------------------------------
def split_train_dataset(dataset, val_ratio=0.2, seed=42):
    person_ids = [dataset.data_list[i]['person_id'] for i in range(len(dataset))]
    person_ids = sorted(list(set(person_ids)))
    train_ids, val_ids = train_test_split(person_ids, test_size=val_ratio, random_state=seed)

    train_indices, val_indices = [], []
    for i, sample in enumerate(dataset.data_list):
        if sample['person_id'] in train_ids:
            train_indices.append(i)
        else:
            val_indices.append(i)


    return train_indices, val_indices, train_ids, val_ids

# ------------------------------
# 赛制对齐数据构建
# ------------------------------
MODALITIES = ['vis', 'nir', 'sk', 'cp', 'text']

# 模态名称映射：兼容新CLIP+MER架构
MODALITY_MAPPING = {
    'vis': 'rgb',      # 可见光 -> RGB
    'nir': 'ir',       # 近红外 -> IR
    'sk': 'sketch',    # 素描 -> sketch
    'cp': 'cpencil',   # 彩铅 -> cpencil
    'text': 'text'     # 文本保持不变
}

def map_modality_name(old_name: str) -> str:
    """将数据集的模态名称映射到新架构的模态名称"""
    return MODALITY_MAPPING.get(old_name, old_name)


def move_batch_to_device(batch, device):
    """将batch数据移动到指定设备"""
    def _move_to_device(obj):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: _move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_move_to_device(item) for item in obj]
        else:
            return obj
    
    return _move_to_device(batch)


def convert_batch_for_clip_model(batch):
    """
    将数据集的batch格式转换为CLIP+MER模型的输入格式
    Args:
        batch: 数据集返回的batch，格式为 {'images': {...}, 'text_description': [...], ...}
    Returns:
        images: {modality: tensor} 图像字典，模态名称已映射
        texts: List[str] 文本列表
    """
    images = {}
    texts = None
    
    # 处理图像数据
    if 'images' in batch:
        for old_modality, image_tensor in batch['images'].items():
            if torch.is_tensor(image_tensor) and image_tensor.numel() > 0:
                new_modality = map_modality_name(old_modality)
                images[new_modality] = image_tensor
    
    # 处理文本数据
    if 'text_description' in batch:
        text_descriptions = batch['text_description']
        if isinstance(text_descriptions, list) and len(text_descriptions) > 0:
            # 过滤空文本
            texts = [text for text in text_descriptions if text and text.strip()]
            if not texts:  # 如果所有文本都是空的
                texts = None
    
    return images, texts


def call_model_with_batch(model, batch, return_features=False):
    """
    使用batch数据调用CLIP+MER模型
    Args:
        model: CLIP+MER模型
        batch: 数据集batch
        return_features: 是否返回特征
    Returns:
        模型输出
    """
    images, texts = convert_batch_for_clip_model(batch)
    
    # 确保至少有一种模态的输入
    if not images and not texts:
        raise ValueError("Batch中没有有效的图像或文本数据")
    
    # 调用模型
    return model(images=images if images else None, 
                texts=texts if texts else None, 
                return_features=return_features)

def build_val_presence_table(dataset, val_indices):
    presence = {}
    for idx in val_indices:
        entry = dataset.data_list[idx]
        pid_str = entry['person_id_str']
        has = {m: False for m in ['vis','nir','sk','cp','text']}
        cache = dataset.image_cache.get(pid_str, {})
        for m in ['vis','nir','sk','cp']:
            has[m] = len(cache.get(m, [])) > 0
        td = entry.get('text_description', '')
        has['text'] = isinstance(td, str) and len(td) > 0
        presence[idx] = has
    return presence

class GalleryOnlyVIS(Dataset):
    """画廊只保留可见光（vis）的 wrapper"""
    def __init__(self, base_dataset: Dataset, indices: List[int], presence: Dict[int, Dict[str, bool]]):
        self.base = base_dataset
        self.indices = [i for i in indices if presence.get(i, {}).get('vis', False)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        rec = self.base[base_idx]
        images = {}
        if 'images' in rec and 'vis' in rec['images']:
            images['vis'] = rec['images']['vis']
        pid = rec['person_id']
        if torch.is_tensor(pid):
            pid = int(pid.item())
        return {
            'person_id': torch.tensor(pid, dtype=torch.long),
            'images': images,
            'text_description': [""],
            'modality_mask': {'vis': True, 'nir': False, 'sk': False, 'cp': False, 'text': False}
        }

class CombinationQueryDataset(Dataset):
    """
    查询集合：对每个身份 index，选择指定的模态组合（不含 vis）
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

        images = {}
        if 'images' in rec and isinstance(rec['images'], dict):
            for m in mods:
                if m in ['nir', 'sk', 'cp'] and m in rec['images'] and torch.is_tensor(rec['images'][m]):
                    images[m] = rec['images'][m]

        text_desc = ""
        if 'text' in mods:
            td = rec.get('text_description', "")
            if isinstance(td, list):
                text_desc = td[0] if len(td) > 0 else ""
            elif isinstance(td, str):
                text_desc = td

        pid_t = rec['person_id']
        pid = int(pid_t.item()) if torch.is_tensor(pid_t) else int(pid_t)

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
    presence = build_val_presence_table(dataset, val_indices)
    gal_ds = GalleryOnlyVIS(dataset, val_indices, presence)
    gallery_loader = DataLoader(
        gal_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=compatible_collate_fn
    )
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
                persistent_workers=True,
                prefetch_factor=2,
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
                    persistent_workers=True,
                    prefetch_factor=2,
                    collate_fn=compatible_collate_fn
                )
                query_loaders[tag][key] = loader

    return gallery_loader, query_loaders


def _subsample_features(feats: torch.Tensor, labels: torch.Tensor, ratio: float, rng: Optional[torch.Generator]=None):
    """在拼接后按样本维度进行子采样，避免按 batch 采样的偏置"""
    if ratio >= 0.999 or feats.size(0) <= 1:
        return feats, labels
    n = max(1, int(math.ceil(feats.size(0) * ratio)))
    perm = torch.randperm(feats.size(0), generator=rng, device=feats.device)[:n]
    return feats[perm], labels[perm]

def validate_competition_style(model, gallery_loader, query_loaders, device, k_map=100, sample_ratio=1.0):
    """赛制对齐评测：mAP/CMC；按样本子集采样避免偏置"""
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(1234)

    with torch.no_grad():
        gal_feats, gal_labels = [], []
        for batch in tqdm(gallery_loader, desc=f'提取画廊特征(全量)', 
                          leave=False, ncols=120, mininterval=1.0):
            batch = move_batch_to_device(batch, device)
            with autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                outputs = call_model_with_batch(model, batch, return_features=True)
                feats = outputs['features']  # 提取融合后的特征用于ReID
            labels = batch['person_id']
            gal_feats.append(feats.cpu()); gal_labels.append(labels.cpu())
        gal_feats = torch.cat(gal_feats, dim=0)
        gal_labels = torch.cat(gal_labels, dim=0)

    # 对画廊进行样本级采样（仅用于相似度计算阶段）
    if sample_ratio < 1.0 and gal_feats.size(0) > 1:
        idx = torch.randperm(gal_feats.size(0))[:max(1, int(gal_feats.size(0)*sample_ratio))]
        gal_feats = gal_feats[idx]; gal_labels = gal_labels[idx]

    # 快速验证：只测single和quad模态，跳过double和triple
    detail = {'single': {}, 'quad': {}}
    buckets = {'single': [], 'quad': []}
    all_q_feats, all_q_labels = [], []

    with torch.no_grad():
        for tag, group in query_loaders.items():
            # 只处理单模态和四模态查询，跳过双模态和三模态
            if tag not in ['single', 'quad']:
                continue
                
            for key, qloader in group.items():
                qf, ql = [], []
                for batch in tqdm(qloader, desc=f'提取查询特征[{tag}:{key}]', 
                                  leave=False, ncols=120, mininterval=1.0):
                    batch = move_batch_to_device(batch, device)
                    with autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                        outputs = call_model_with_batch(model, batch, return_features=True)
                        feats = outputs['features']  # 提取融合后的特征用于ReID
                    labels = batch['person_id']
                    qf.append(feats.cpu()); ql.append(labels.cpu())
                if not qf:
                    continue
                qf = torch.cat(qf, dim=0); ql = torch.cat(ql, dim=0)

                # 对查询也做样本级采样
                qf, ql = _subsample_features(qf, ql, sample_ratio)

                km = min(k_map, gal_feats.size(0))
                m = compute_map(qf, gal_feats, ql, gal_labels, k=km)

                detail[tag][key] = float(m)
                buckets[tag].append(m)
                all_q_feats.append(qf); all_q_labels.append(ql)

    def _avg(x): return float(np.mean(x)) if x else 0.0
    map_single = _avg(buckets['single'])
    map_quad   = _avg(buckets['quad'])
    # 快速评估：只用single和quad的平均
    map_avg2   = float(np.mean([map_single, map_quad])) if (map_single > 0 or map_quad > 0) else 0.0

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
        'map_quad': map_quad,
        'map_avg2': map_avg2,  # single和quad的平均
        'detail': detail,
        'cmc1': cmc1, 'cmc5': cmc5, 'cmc10': cmc10
    }

# ------------------------------
# 训练一个 epoch
# ------------------------------
def train_epoch(model, dataloader, optimizer, device, epoch, scaler=None, adaptive_clip=True):
    model.train()
    total_loss = 0.0
    ce_loss_sum = 0.0
    contrastive_loss_sum = 0.0
    correct = 0
    total = 0
    
    feature_norms = []
    loss_spikes = 0
    grad_norms = []
    
    use_amp = (scaler is not None and getattr(scaler, "is_enabled", lambda: True)())

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', 
                leave=False, ncols=150, mininterval=1.0, maxinterval=3.0,
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    # 添加批次构成监控（前3个batch）
    if epoch <= 3:
        pbar.write(f"=== Epoch {epoch} 批次构成监控 ===")
    
    for batch_idx, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device)
        labels = batch['person_id']
        
        # 批次构成监控（前3个epoch的前3个batch）
        if epoch <= 3 and batch_idx < 3:
            unique_ids, counts = torch.unique(labels, return_counts=True)
            num_ids_per_batch = len(unique_ids)
            avg_instances_per_id = float(counts.float().mean().item())  # 转换为浮点数再计算均值
            pbar.write(f"Batch {batch_idx}: {num_ids_per_batch} IDs, 平均每ID {avg_instances_per_id:.1f} 样本 (K-1正样本数≈{avg_instances_per_id-1:.1f})")

        optimizer.zero_grad(set_to_none=True)
        # === 对比损失提前升温 + 加长升温期 ===
        cont_max = getattr(model.config, "contrastive_weight", 0.05)
        if epoch <= 5:
            effective_cont_w = 0.0
        elif 6 <= epoch <= 25:
            # 线性升温：6->0.004, 25->cont_max
            t = (epoch - 5) / 20.0
            effective_cont_w = max(0.004, cont_max * t)
        else:
            effective_cont_w = cont_max
            
        # 每个epoch的第一个batch输出有效对比权重
        if batch_idx == 0:
            pbar.write(f"Epoch {epoch}: effective_contrastive_weight = {effective_cont_w:.4f} (max={cont_max:.4f})")

        with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = call_model_with_batch(model, batch, return_features=False)
            loss_dict = model.compute_loss(outputs, labels)
            
            # SDM对比损失监控（仅在实际参与训练时警告）
            sdm_loss = loss_dict.get('sdm_loss', torch.tensor(0.0, device=device))
            cont_loss = loss_dict.get('contrastive_loss', sdm_loss)  # 兼容性：用sdm_loss作为contrastive_loss
            
            # 只有当SDM损失真正参与训练时（权重>0）才显示警告
            if effective_cont_w > 0.0 and sdm_loss.item() > 1.5:  
                original_sdm = sdm_loss.item()
                if batch_idx < 5:  # 只在前几个batch报告，避免日志过多
                    pbar.write(f"Epoch {epoch}, Batch {batch_idx}: SDM对齐损失较高 {original_sdm:.3f}")
            
            # 检查是否有NaN或Inf
            if not torch.isfinite(sdm_loss):
                pbar.write(f"❌ Epoch {epoch}, Batch {batch_idx}: SDM损失出现NaN/Inf, 重置为0")
                loss_dict['sdm_loss'] = torch.tensor(0.0, device=device)
                loss_dict['contrastive_loss'] = torch.tensor(0.0, device=device)
                # 重新计算总损失
                ce_loss = loss_dict.get('ce_loss', torch.tensor(0.0, device=device))
                feat_penalty = loss_dict.get('feat_penalty', torch.tensor(0.0, device=device))
                sdm_weight = getattr(model.config, 'contrastive_weight', 0.1)
                loss_dict['total_loss'] = ce_loss + sdm_weight * loss_dict['sdm_loss'] + feat_penalty
            
            loss = loss_dict['total_loss']
        
        # 用 warmup 后的有效权重重算总损失（包含特征范数正则项）
        ce_loss = loss_dict.get('ce_loss', torch.tensor(0.0, device=device))
        sdm_loss = loss_dict.get('sdm_loss', torch.tensor(0.0, device=device))
        cont_loss = loss_dict.get('contrastive_loss', sdm_loss)  # 兼容性
        feat_penalty = loss_dict.get('feat_penalty', torch.tensor(0.0, device=device))
        loss = ce_loss + effective_cont_w * sdm_loss + feat_penalty  # 使用SDM损失
        loss_dict['total_loss'] = loss

        current_loss = float(loss.item())
        if current_loss > 50.0 or not np.isfinite(current_loss):
            pbar.write(f"⚠️ Epoch {epoch}, Batch {batch_idx}: 异常损失 {current_loss:.3f}, 跳过")
            optimizer.zero_grad(set_to_none=True)
            loss_spikes += 1
            continue
            
        if batch_idx > 0 and current_loss > total_loss / batch_idx * 1.5:
            loss_spikes += 1
        
        # 计算梯度（移回循环内部）
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _sanitize_grads(model)

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    if torch.isfinite(param_norm):
                        total_norm += param_norm.item() ** 2
            total_norm = math.sqrt(total_norm)
            grad_norms.append(float(total_norm))

            # 自适应梯度裁剪（收紧上限提升稳定性）
            if adaptive_clip:
                if len(grad_norms) > 10:
                    recent_norms = grad_norms[-10:]
                    adaptive_max_norm = min(3.0, max(0.5, np.percentile(recent_norms, 70) * 1.15))
                else:
                    adaptive_max_norm = 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(adaptive_max_norm))
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            _sanitize_grads(model)

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    if torch.isfinite(param_norm):
                        total_norm += param_norm.item() ** 2
            total_norm = math.sqrt(total_norm)
            grad_norms.append(float(total_norm))

            if adaptive_clip:
                if len(grad_norms) > 10:
                    recent_norms = grad_norms[-10:]
                    adaptive_max_norm = min(3.0, max(0.5, np.percentile(recent_norms, 70) * 1.15))
                else:
                    adaptive_max_norm = 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(adaptive_max_norm))
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # 统计信息更新（移回循环内部）
        total_loss += current_loss
        ce_loss_sum += float(ce_loss.item())
        contrastive_loss_sum += float(sdm_loss.item())  # 累计SDM损失

        if isinstance(outputs, dict) and 'logits' in outputs:
            _, predicted = outputs['logits'].max(1)
        else:
            _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 建议：同时记录两种特征范数以更好地监控训练
        fused_norms = None
        reid_raw_norms = None
        if isinstance(outputs, dict):
            if 'features' in outputs:
                fused_norms = torch.norm(outputs['features'].detach(), p=2, dim=1)  # 被feat_penalty约束的量
            if 'reid_features_raw' in outputs:
                reid_raw_norms = torch.norm(outputs['reid_features_raw'].detach(), p=2, dim=1)  # 受LayerNorm影响

        # 使用融合特征范数更新累积统计（因为这是正则化目标）
        if fused_norms is not None:
            feature_norms.extend(fused_norms.cpu().numpy())

        # 进度条里分两栏看更直观（显示当前batch的均值）
        avg_fused = float(fused_norms.mean().item()) if fused_norms is not None else 0.0
        avg_reid  = float(reid_raw_norms.mean().item()) if reid_raw_norms is not None else 0.0
        avg_grad_norm = np.mean(grad_norms[-10:]) if grad_norms else 0.0

        # 优化：每3个batch更新一次进度条显示，避免频繁GPU-CPU同步
        if batch_idx % 3 == 0 or batch_idx == len(dataloader) - 1:
            pbar.set_postfix({
                'Loss': f'{current_loss:.3f}',
                'CE': f'{float(ce_loss.item()):.3f}',
                'SDM': f'{float(sdm_loss.item()):.3f}',  # 显示SDM对齐损失
                'Feat(Fused)': f'{avg_fused:.2f}',
                'GradNorm': f'{avg_grad_norm:.2f}',
                'Spikes': loss_spikes
            })

    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = 100. * correct / max(1, total)
    avg_feat_norm = np.mean(feature_norms) if feature_norms else 0.0
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
    
    if loss_spikes > len(dataloader) * 0.1:
        logging.warning(f"Epoch {epoch}: 损失异常次数 {loss_spikes}，建议降低学习率")
    if avg_feat_norm > 50.0:
        logging.warning(f"Epoch {epoch}: 平均特征范数 {avg_feat_norm:.2f} 偏大，检查稳定性")
    
    return {
        'total_loss': avg_loss,
        'ce_loss': ce_loss_sum / max(1, len(dataloader)),
        'contrastive_loss': contrastive_loss_sum / max(1, len(dataloader)),  # 兼容性：实际为SDM损失
        'sdm_loss': contrastive_loss_sum / max(1, len(dataloader)),  # 明确的SDM损失
        'accuracy': accuracy,
        'feature_norm': avg_feat_norm,
        'grad_norm': avg_grad_norm,
        'loss_spikes': loss_spikes,
        'stability_score': max(0.0, 1.0 - loss_spikes / max(1, len(dataloader)))
    }

# ------------------------------
# 训练主流程
# ------------------------------
def _build_lambda_with_warmup_cosine(total_epochs, warmup_epochs, start_factor=0.01, min_factor=0.01):
    """返回一个 epoch->scale 的函数：warmup（线性）后接 cosine 衰减；保持 param group 比例"""
    assert 0.0 < start_factor <= 1.0
    assert 0.0 < min_factor <= 1.0
    def lmbda(epoch: int):
        if epoch < warmup_epochs:
            return start_factor + (1.0 - start_factor) * (epoch + 1) / max(1, warmup_epochs)
        # cosine
        T = max(1, total_epochs - warmup_epochs)
        t = max(0, epoch - warmup_epochs)
        cos = 0.5 * (1.0 + math.cos(math.pi * t / T))
        return min_factor + (1.0 - min_factor) * cos
    return lmbda

def train_multimodal_reid():
    # 配置与设备
    config = TrainingConfig()
    set_seed(getattr(config, "seed", 42))

    device_str = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    setup_logging(getattr(config, "log_dir", "./logs"))

    # 加载完整数据集并进行身份划分
    logging.info("加载数据集并进行身份划分...")
    full_dataset = MultiModalDataset(config, split='train')

    train_indices, val_indices, train_ids, val_ids = split_train_dataset(
        full_dataset,
        val_ratio=getattr(config, "val_ratio", 0.2),
        seed=getattr(config, "seed", 42)
    )

    # 创建完整的 person_id -> label 映射（包含训练集和验证集）
    all_person_ids = sorted(list(set(train_ids + val_ids)))
    pid2label = {pid: i for i, pid in enumerate(all_person_ids)}
    
    logging.info("创建训练和验证数据集...")
    
    # 先保存原始数据列表
    original_data_list = full_dataset.data_list.copy()
    
    # 创建训练数据集
    train_dataset = full_dataset
    train_dataset.person_ids = all_person_ids
    train_dataset.pid2label = pid2label
    train_dataset.data_list = [item for item in original_data_list if item['person_id'] in train_ids]

    # 创建真正的验证集（split='val'，关闭训练模式和模态dropout）
    logging.info("创建验证集数据集（关闭增强和模态dropout）...")
    local_val_dataset = MultiModalDataset(config, split='val', person_ids=all_person_ids)
    local_val_dataset.pid2label = pid2label
    # 只保留验证ID的样本
    local_val_dataset.data_list = [item for item in local_val_dataset.data_list if item['person_id'] in val_ids]

    logging.info(f"数据集划分完成 - 训练集: {len(train_ids)} IDs, {len(train_dataset.data_list)} 样本")
    logging.info(f"数据集划分完成 - 本地验证集: {len(val_ids)} IDs, {len(local_val_dataset.data_list)} 样本")

    # 分类头 num_classes（应该覆盖所有可能的person_id以避免标签超出范围）
    config.num_classes = len(all_person_ids)
    logging.info(f"分类器输出维度: {config.num_classes} 类")
    logging.info(f"标签映射范围: 0-{len(all_person_ids)-1} （包含训练集+验证集）")
    train_labels = [pid2label[pid] for pid in train_ids]
    val_labels = [pid2label[pid] for pid in val_ids]
    logging.info(f"训练集标签范围: {min(train_labels)}-{max(train_labels)}, 验证集标签范围: {min(val_labels)}-{max(val_labels)}")

    # 训练 DataLoader（确保 P×K 采样，K≥4 保证对比学习有效）
    num_instances = 4  # K=4，每个身份4个样本，确保每个锚有K-1=3个正样本
    effective_batch_size = 32  # 固定batch_size=32，确保P=8
    num_pids_per_batch = effective_batch_size // num_instances  # P=8
    logging.info(f"采样策略: P×K = {num_pids_per_batch}×{num_instances} = {effective_batch_size}")
    logging.info(f"每个锚的正样本数: {num_instances-1}")
    
    train_sampler = BalancedBatchSampler(train_dataset, effective_batch_size, num_instances=num_instances)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=getattr(config, "num_workers", 4),
        pin_memory=getattr(config, "pin_memory", True),
        persistent_workers=True,  # 保持工作进程持续运行
        prefetch_factor=2,        # 预取因子提升IO效率
        collate_fn=compatible_collate_fn
    )

    # 本地验证 DataLoader（赛制对齐）
    gallery_loader, query_loaders = build_eval_loaders_by_rule(
        local_val_dataset, list(range(len(local_val_dataset))),
        batch_size=effective_batch_size,
        num_workers=getattr(config, "num_workers", 4),
        pin_memory=getattr(config, "pin_memory", True)
    )

    # 模型：CLIP+MER架构
    model = CLIPBasedMultiModalReIDModel(config).to(device)
    
    # 设置分类器（动态设置ID类别数）
    num_classes = getattr(config, 'num_classes', None)
    if num_classes is not None:
        model.set_num_classes(num_classes)
        logging.info(f"✅ 设置分类器：{num_classes} 个ID类别")
    else:
        logging.warning("⚠️ config中未找到num_classes，请确保在数据加载后设置")

    # 优化器 - 支持CLIP+MER分层学习率
    param_groups = model.get_learnable_params()
    
    # 日志输出各参数组的学习率
    for group in param_groups:
        group_name = group.get('name', 'unknown')
        group_lr = group['lr']
        num_params = len(group['params'])
        logging.info(f"{group_name}: {num_params} 参数, 学习率: {group_lr:.2e}")
    
    optimizer = AdamW(param_groups, weight_decay=getattr(config, "weight_decay", 1e-4))

    # AMP（修复：正确初始化）
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    logging.info("启用混合精度训练 (AMP)" if scaler.is_enabled() else "使用全精度训练")

    # 学习率调度器
    warmup_epochs = getattr(config, "warmup_epochs", 15)
    scheduler_type = getattr(config, "scheduler", "cosine")
    num_epochs = getattr(config, "num_epochs", 100)

    if scheduler_type == 'cosine':
        # 使用 LambdaLR 保持 param group 比例
        start_factor = 0.01
        min_factor = 0.01
        lmbda = _build_lambda_with_warmup_cosine(num_epochs, warmup_epochs, start_factor, min_factor)
        scheduler = LambdaLR(optimizer, lr_lambda=[lmbda] * len(optimizer.param_groups))
        logging.info(f"调度器: Warmup({warmup_epochs}) + Cosine(min_factor={min_factor}) via LambdaLR")
    elif scheduler_type == 'plateau':
        # 从optimizer的第一个参数组获取base_lr
        base_lr = optimizer.param_groups[0]['lr']
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=8, threshold=0.001,
            min_lr=base_lr * 0.001, verbose=True
        )
        logging.info("调度器: ReduceLROnPlateau (基于mAP)")
    elif scheduler_type == 'step':
        step_size = int(50 * getattr(config, "conservative_factor", 0.7))
        step_size = max(step_size, 30)
        gamma = 0.3 + 0.4 * getattr(config, "conservative_factor", 0.7)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logging.info(f"调度器: StepLR(step_size={step_size}, gamma={gamma:.2f})")
    elif scheduler_type == 'multistep':
        cf = getattr(config, "conservative_factor", 0.7)
        milestones = [int(60 * cf), int(80 * cf), int(95 * cf)]
        milestones = [max(m, 30) for m in milestones]
        gamma = 0.2 + 0.5 * cf
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        logging.info(f"调度器: MultiStepLR(milestones={milestones}, gamma={gamma:.2f})")
    else:
        scheduler = None
        logging.info("调度器: None")

    # 训练循环
    best_map = 0.0
    train_history, val_history = [], []
    eval_freq = getattr(config, "eval_freq", 20)
    save_dir = getattr(config, "save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # 每个epoch开始前清除文本缓存（仅在微调文本时清除，冻结时保持缓存）
        if hasattr(model, "text_cache") and not getattr(model, "freeze_text", True):
            model.text_cache.clear()
            if epoch == 1:  # 第一个epoch提醒
                logging.info("文本编码器微调模式：每epoch清除缓存")

        adaptive_clip = getattr(config, "adaptive_gradient_clip", True)
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, scaler, adaptive_clip)
        
        # 监控SDM对齐损失异常并自动调整
        current_sdm_loss = train_metrics.get('sdm_loss', train_metrics['contrastive_loss'])
        if epoch > 5 and len(train_history) >= 2:
            prev_sdm_loss = train_history[-1].get('sdm_loss', train_history[-1]['contrastive_loss'])
            if current_sdm_loss > prev_sdm_loss * 1.8:
                logging.warning(f"Epoch {epoch}: SDM对齐损失异常上升 {prev_sdm_loss:.4f} -> {current_sdm_loss:.4f}")
                # 自动降低SDM损失权重
                current_weight = getattr(config, 'contrastive_weight', 0.1)
                new_weight = max(0.001, current_weight * 0.5)
                config.contrastive_weight = new_weight
                model.config.contrastive_weight = new_weight
                logging.info(f"自动调整SDM损失权重: {current_weight:.4f} -> {new_weight:.4f}")
        
        # 自适应增强强度：5个epoch后如果训练稳定，放宽裁剪强度
        if epoch == 5 and train_metrics['stability_score'] > 0.8:
            logging.info("训练稳定，放宽数据增强强度：裁剪 scale (0.8,1.0) -> (0.6,1.0)")
            # 更新数据增强变换
            new_transform = transforms.Compose([
                transforms.RandomResizedCrop(config.image_size, scale=(0.6, 1.0)),  # 放宽裁剪
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2) if config.color_jitter else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=config.random_erase, scale=(0.02, 0.2)) if config.random_erase > 0 else transforms.Lambda(lambda x: x)
            ])
            # 更新训练数据集的变换
            if hasattr(train_dataset, 'transform'):
                train_dataset.transform = new_transform


        if epoch % eval_freq == 0:
            sample_ratio = getattr(config, "eval_sample_ratio", 0.3)
            comp_metrics = validate_competition_style(model, gallery_loader, query_loaders, device, k_map=100, sample_ratio=sample_ratio)

            train_history.append({'epoch': epoch, **train_metrics})
            val_history.append({'epoch': epoch, **comp_metrics})

            score = comp_metrics['map_avg2']
            if score > best_map:
                best_map = score
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_map, config,
                    os.path.join(save_dir, 'best_model.pth')
                )
                logging.info(f"新的最佳(single+quad平均) mAP: {best_map:.4f}")

            # 输出分模态mAP详情，快速定位拖后腿的模态
            single_detail = comp_metrics.get('detail', {}).get('single', {})
            single_maps = []
            for modality in ['text', 'nir', 'sk', 'cp']:
                if modality in single_detail:
                    single_maps.append(f"{modality}:{single_detail[modality]:.3f}")
                    
            single_detail_str = " | ".join(single_maps) if single_maps else "N/A"
            
            logging.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train ClsAcc: {train_metrics['accuracy']:.2f}% - "
                f"mAP(single/quad/avg2): "
                f"{comp_metrics['map_single']:.4f}/"
                f"{comp_metrics['map_quad']:.4f}/"
                f"{comp_metrics['map_avg2']:.4f} - "
                f"CMC@1/5/10: {comp_metrics['cmc1']:.4f}/"
                f"{comp_metrics['cmc5']:.4f}/"
                f"{comp_metrics['cmc10']:.4f} - "
                f"用时: {time.time() - start_time:.2f}s"
            )
            logging.info(f"分模态mAP详情: {single_detail_str}")

        # 保存 checkpoint
        if epoch % getattr(config, "save_freq", 20) == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_map, config,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            )

        # 调度器步进
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                if epoch % eval_freq == 0:
                    current_map = comp_metrics['map_avg2'] if 'comp_metrics' in locals() else 0.0
                    scheduler.step(current_map)
            else:
                scheduler.step()

            if epoch % 20 == 0:  # 减少查询频率：10->20
                lrs = [pg['lr'] for pg in optimizer.param_groups]
                logging.info(f"Epoch {epoch}: 当前学习率 = {', '.join([f'{lr:.2e}' for lr in lrs])}")

    # 训练完成后全量评估
    logging.info("训练完成，开始本地划分验证集的完整评估...")
    final_metrics = validate_competition_style(model, gallery_loader, query_loaders, device, k_map=100, sample_ratio=1.0)
    # 最终评估的分模态mAP详情
    final_single_detail = final_metrics.get('detail', {}).get('single', {})
    final_single_maps = []
    for modality in ['text', 'nir', 'sk', 'cp']:
        if modality in final_single_detail:
            final_single_maps.append(f"{modality}:{final_single_detail[modality]:.3f}")
    final_single_detail_str = " | ".join(final_single_maps) if final_single_maps else "N/A"
    
    logging.info(f"最终评估 - mAP(single/quad/avg2): "
                f"{final_metrics['map_single']:.4f}/"
                f"{final_metrics['map_quad']:.4f}/"
                f"{final_metrics['map_avg2']:.4f}")
    logging.info(f"最终分模态mAP详情: {final_single_detail_str}")
    
    # 找出拖后腿的模态
    if final_single_detail:
        min_modality = min(final_single_detail.items(), key=lambda x: x[1])
        max_modality = max(final_single_detail.items(), key=lambda x: x[1])
        logging.info(f"性能最差模态: {min_modality[0]}({min_modality[1]:.3f}), 最佳模态: {max_modality[0]}({max_modality[1]:.3f})")

    # 保存历史
    log_dir = getattr(config, "log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)
    pd.DataFrame(train_history).to_csv(os.path.join(log_dir, 'train_history.csv'), index=False)
    pd.DataFrame(val_history).to_csv(os.path.join(log_dir, 'local_val_history.csv'), index=False)
    final_results = {'epoch': 'final', **final_metrics}
    pd.DataFrame([final_results]).to_csv(os.path.join(log_dir, 'local_val_final_evaluation.csv'), index=False)

    # 保存划分
    split_info = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'train_indices': train_indices,
        'val_indices': val_indices
    }
    with open(os.path.join(save_dir, 'dataset_split.pkl'), 'wb') as f:
        pickle.dump(split_info, f)

    logging.info(f"训练完成. 本地划分验证集最佳(四类平均) mAP: {best_map:.4f}")

def save_checkpoint(model, optimizer, scheduler, epoch, best_map, config, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_map': best_map,
        'num_classes': getattr(config, 'num_classes', None),
        'config': config.__dict__ if hasattr(config, '__dict__') else str(config)
    }
    torch.save(checkpoint, filename)

if __name__ == "__main__":
    train_multimodal_reid()
