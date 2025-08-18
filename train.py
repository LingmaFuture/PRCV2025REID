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
                          leave=False, ncols=100, mininterval=0.5):
            batch = move_batch_to_device(batch, device)
            with autocast(device_type='cuda', enabled=device.type == 'cuda'):
                feats = model(batch, return_features=True)
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
                                  leave=False, ncols=100, mininterval=0.5):
                    batch = move_batch_to_device(batch, device)
                    with autocast(device_type='cuda', enabled=device.type == 'cuda'):
                        feats = model(batch, return_features=True)
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
        'map_double': 0.0,  # 未测试，设为0
        'map_triple': 0.0,  # 未测试，设为0
        'map_quad': map_quad,
        'map_avg2': map_avg2,  # 实际是single和quad的平均
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
                leave=True, ncols=120, mininterval=1.0, maxinterval=2.0)
    for batch_idx, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device)
        labels = batch['person_id']

        optimizer.zero_grad(set_to_none=True)
        # === 对比损失"真·关掉"更久 + 平滑拉起 ===
        cont_max = getattr(model.config, "contrastive_weight", 0.02)
        if epoch <= 10:
            effective_cont_w = 0.0
        elif 11 <= epoch <= 20:
            # 线性升温：11->0.004, 20->cont_max
            t = (epoch - 10) / 10.0
            effective_cont_w = max(0.002, cont_max * t)
        else:
            effective_cont_w = cont_max

        with autocast(device_type='cuda', enabled=use_amp):
            outputs = model(batch)
            loss_dict = model.compute_loss(outputs, labels)
            
            # 对比损失监控（仅在实际参与训练时警告）
            cont_loss = loss_dict.get('contrastive_loss', torch.tensor(0.0, device=device))
            # 只有当对比损失真正参与训练时（权重>0）才显示警告
            if effective_cont_w > 0.0 and cont_loss.item() > 1.5:  
                original_cont = cont_loss.item()
                if batch_idx < 5:  # 只在前几个batch报告，避免日志过多
                    logging.info(f"Epoch {epoch}, Batch {batch_idx}: 对比损失较高 {original_cont:.3f}")
            
            # 检查是否有NaN或Inf
            if not torch.isfinite(cont_loss):
                logging.error(f"Epoch {epoch}, Batch {batch_idx}: 对比损失出现NaN/Inf, 重置为0")
                loss_dict['contrastive_loss'] = torch.tensor(0.0, device=device)
                # 重新计算总损失
                ce_loss = loss_dict.get('ce_loss', torch.tensor(0.0, device=device))
                contrastive_weight = getattr(model.config, 'contrastive_weight', 0.02)
                loss_dict['total_loss'] = ce_loss + contrastive_weight * loss_dict['contrastive_loss']
            
            loss = loss_dict['total_loss']
        
        # 用 warmup 后的有效权重重算总损失（包含特征范数正则项）
        ce_loss = loss_dict.get('ce_loss', torch.tensor(0.0, device=device))
        cont_loss = loss_dict.get('contrastive_loss', torch.tensor(0.0, device=device))
        feat_penalty = loss_dict.get('feat_penalty', torch.tensor(0.0, device=device))
        loss = ce_loss + effective_cont_w * cont_loss + feat_penalty
        loss_dict['total_loss'] = loss

        current_loss = float(loss.item())
        if current_loss > 50.0 or not np.isfinite(current_loss):
            logging.warning(f"Epoch {epoch}, Batch {batch_idx}: 异常损失 {current_loss:.3f}, 跳过")
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
        contrastive_loss_sum += float(cont_loss.item())

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

        pbar.set_postfix({
            'Loss': f'{current_loss:.3f}',
            'CE': f'{float(ce_loss.item()):.3f}',
            'Cont': f'{float(cont_loss.item()):.3f}',
            'Feat(Fused)': f'{avg_fused:.2f}',
            #'Feat(ReID)': f'{avg_reid:.2f}',
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
        'contrastive_loss': contrastive_loss_sum / max(1, len(dataloader)),
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

    # 本地验证 DataLoader（赛制对齐）
    gallery_loader, query_loaders = build_eval_loaders_by_rule(
        local_val_dataset, list(range(len(local_val_dataset))),
        batch_size=effective_batch_size,
        num_workers=getattr(config, "num_workers", 4),
        pin_memory=getattr(config, "pin_memory", True)
    )

    # 模型
    model = MultiModalReIDModel(config).to(device)

    # 优化器 - 为文本编码器设置差异化学习率
    base_lr = getattr(config, "learning_rate", 3e-4)
    text_lr = base_lr * 0.1
    param_groups = []
    text_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'text_encoder' in name:
            text_params.append(param)
        else:
            other_params.append(param)
    if text_params:
        param_groups.append({'params': text_params, 'lr': text_lr})
        logging.info(f"文本编码器学习率: {text_lr:.2e}")
    if other_params:
        param_groups.append({'params': other_params, 'lr': base_lr})
        logging.info(f"其他模块学习率: {base_lr:.2e}")
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

        adaptive_clip = getattr(config, "adaptive_gradient_clip", True)
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, scaler, adaptive_clip)
        
        # 监控对比损失异常并自动调整
        current_cont_loss = train_metrics['contrastive_loss']
        if epoch > 5 and len(train_history) >= 2:
            prev_cont_loss = train_history[-1]['contrastive_loss']
            if current_cont_loss > prev_cont_loss * 1.8:
                logging.warning(f"Epoch {epoch}: 对比损失异常上升 {prev_cont_loss:.4f} -> {current_cont_loss:.4f}")
                # 自动降低对比损失权重
                current_weight = getattr(config, 'contrastive_weight', 0.02)
                new_weight = max(0.001, current_weight * 0.5)
                config.contrastive_weight = new_weight
                model.config.contrastive_weight = new_weight
                logging.info(f"自动调整对比损失权重: {current_weight:.4f} -> {new_weight:.4f}")


        if epoch % eval_freq == 0:
            sample_ratio = getattr(config, "eval_sample_ratio", 0.3)
            comp_metrics = validate_competition_style(model, gallery_loader, query_loaders, device, k_map=100, sample_ratio=sample_ratio)

            train_history.append({'epoch': epoch, **train_metrics})
            val_history.append({'epoch': epoch, **comp_metrics})

            score = comp_metrics['map_avg4']
            if score > best_map:
                best_map = score
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_map, config,
                    os.path.join(save_dir, 'best_model.pth')
                )
                logging.info(f"新的最佳(single+quad平均) mAP: {best_map:.4f}")

            logging.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train ClsAcc: {train_metrics['accuracy']:.2f}% - "
                f"mAP(single/quad/avg2): "
                f"{comp_metrics['map_single']:.4f}/"
                f"{comp_metrics['map_quad']:.4f}/"
                f"{comp_metrics['map_avg4']:.4f} - "
                f"CMC@1/5/10: {comp_metrics['cmc1']:.4f}/"
                f"{comp_metrics['cmc5']:.4f}/"
                f"{comp_metrics['cmc10']:.4f} - "
                f"用时: {time.time() - start_time:.2f}s"
            )

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
                    current_map = comp_metrics['map_avg4'] if 'comp_metrics' in locals() else 0.0
                    scheduler.step(current_map)
            else:
                scheduler.step()

            if epoch % 10 == 0:
                lrs = [pg['lr'] for pg in optimizer.param_groups]
                logging.info(f"Epoch {epoch}: 当前学习率 = {', '.join([f'{lr:.2e}' for lr in lrs])}")

    # 训练完成后全量评估
    logging.info("训练完成，开始本地划分验证集的完整评估...")
    final_metrics = validate_competition_style(model, gallery_loader, query_loaders, device, k_map=100, sample_ratio=1.0)
    logging.info(f"最终评估 - mAP(single/double/triple/quad/avg4): "
                f"{final_metrics['map_single']:.4f}/"
                f"{final_metrics['map_double']:.4f}/"
                f"{final_metrics['map_triple']:.4f}/"
                f"{final_metrics['map_quad']:.4f}/"
                f"{final_metrics['map_avg4']:.4f}")

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
