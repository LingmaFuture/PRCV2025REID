# train.py
import os
import time
import random
import pickle
import logging
import hashlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import math  # 确保math被导入用于CE诊断
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, ReduceLROnPlateau
from torch.amp import autocast, GradScaler


# guide6.md: PyTorch警告处理
try:
    from torch.nn.attention import SDPBackend
    # 设置默认的SDPA后端，避免警告
    if hasattr(SDPBackend, 'flash_attention'):
        torch.nn.attention.SDPBackend.default = SDPBackend.flash_attention
    elif hasattr(SDPBackend, 'FLASH_ATTENTION'):
        torch.nn.attention.SDPBackend.default = SDPBackend.FLASH_ATTENTION
except ImportError:
    pass  # 如果导入失败，忽略

import torchvision.transforms as transforms

# === 你项目里的数据与模型 ===
from datasets.dataset import MultiModalDataset, compatible_collate_fn
from models.model import CLIPBasedMultiModalReIDModel  
from models.sdm_scheduler import SDMScheduler
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
    """将批次数据移动到指定设备（仅移动 Tensor）- 使用non_blocking优化"""
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
        # ✅ 使用non_blocking加速H2D传输（配合pin_memory=True）
        return batch.to(device, non_blocking=True)
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
    # 确保两个特征张量具有相同的数据类型，统一转换为float32
    query_features = query_features.float()
    gallery_features = gallery_features.float()
    
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
# 赛制对齐数据构建
# ------------------------------
MODALITIES = ['vis', 'nir', 'sk', 'cp', 'text']

# 模态名称映射：兼容新CLIP+MER架构
MODALITY_MAPPING = {
    # 统一使用dataset原生命名 - 恒等映射，消除命名转换
    'vis': 'vis',       # 可见光保持原名
    'nir': 'nir',       # 近红外保持原名  
    'sk': 'sk',         # 素描保持原名
    'cp': 'cp',         # 彩铅保持原名
    'text': 'text',     # 文本保持原名
}

def map_modality_name(old_name: str) -> str:
    """将数据集的模态名称映射到新架构的模态名称"""
    return MODALITY_MAPPING.get(old_name, old_name)


def convert_batch_for_clip_model(batch):
    """
    将数据集的batch格式转换为CLIP+MER模型的输入格式
    修复：按modality_mask过滤有效模态，避免零特征污染融合
    Args:
        batch: 数据集返回的batch，格式为 {'images': {...}, 'text_description': [...], 'modality_mask': {...}}
    Returns:
        images: {modality: tensor} 图像字典，仅包含有效模态，模态名称已映射
        texts: List[str] 文本列表，仅包含有效文本
    """
    images = {}
    texts = None
    
    # 获取模态掩码
    modality_mask = batch.get('modality_mask', {})
    
    # 处理图像数据 - 关键修复：只处理mask指示为有效的模态
    if 'images' in batch:
        for old_modality, image_tensor in batch['images'].items():
            if torch.is_tensor(image_tensor) and image_tensor.numel() > 0:
                # 检查该模态是否在整个batch中有任何有效样本
                mask_tensor = modality_mask.get(old_modality, torch.zeros(image_tensor.size(0)))
                if isinstance(mask_tensor, torch.Tensor) and mask_tensor.sum() > 0:
                    # 至少有一个样本在该模态下有效，才包含该模态
                    new_modality = map_modality_name(old_modality)
                    images[new_modality] = image_tensor
    
    # 处理文本数据 - 修复：始终保持batch size一致，用空字符串占位
    if 'text_description' in batch:
        text_descriptions = batch['text_description']
        if isinstance(text_descriptions, list) and len(text_descriptions) > 0:
            # 获取文本mask
            text_mask = modality_mask.get('text', torch.ones(len(text_descriptions)))
            if isinstance(text_mask, torch.Tensor):
                # 始终保持batch size，用空字符串填充无效文本位置
                processed_texts = []
                has_any_valid = False
                for i, (text, mask_val) in enumerate(zip(text_descriptions, text_mask)):
                    if mask_val > 0.5 and text and text.strip():
                        processed_texts.append(text.strip())
                        has_any_valid = True
                    else:
                        processed_texts.append("")  # 空文本占位，让CLIP自然处理
                
                # 只要batch中有文本字段，就传递给模型（包含空字符串）
                # CLIP tokenizer可以正常处理空字符串
                texts = processed_texts
    
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
    
    # 获取modality_masks
    modality_masks = batch.get('modality_mask', None)
    
    # 调用模型（传递mask信息）
    return model(images=images if images else None, 
                texts=texts if texts else None, 
                modality_masks=modality_masks,
                return_features=return_features)

def build_val_presence_table(dataset, val_indices):
    presence = {}
    for idx in val_indices:
        entry = dataset.data_list[idx]
        # 修复：从person_id转换为字符串，而不是直接访问不存在的person_id_str
        pid_str = str(entry['person_id'])
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

def dl_kwargs(num_workers, collate_fn, pin_memory=True):
    """
    按guide2.md建议：智能DataLoader参数配置
    避免 num_workers=0 时设置 prefetch_factor 导致的错误
    """
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'collate_fn': collate_fn
    }
    
    # 只有多进程模式才设置 prefetch_factor 和 persistent_workers
    if num_workers > 0:
        kwargs.update({
            'persistent_workers': True,
            'prefetch_factor': 2
        })
    
    return kwargs

def build_eval_loaders_by_rule(dataset, val_indices, batch_size, num_workers, pin_memory):
    presence = build_val_presence_table(dataset, val_indices)
    gal_ds = GalleryOnlyVIS(dataset, val_indices, presence)
    gallery_loader = DataLoader(
        gal_ds,
        batch_size=batch_size,
        shuffle=False,
        **dl_kwargs(num_workers, compatible_collate_fn, pin_memory)
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
                **dl_kwargs(num_workers, compatible_collate_fn, pin_memory)
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
                    **dl_kwargs(num_workers, compatible_collate_fn, pin_memory)
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

def _flatten_loaders(obj, prefix=""):
    """
    把 {key: DataLoader | dict | list} 递归展开成 [(name, dataloader), ...]
    name 形如 'single/nir' 或 'quad/0' 等，便于打印/统计
    """
    # DataLoader-like
    if hasattr(obj, "dataset") and hasattr(obj, "__iter__"):
        yield (prefix.rstrip("/") or "root", obj)
        return

    # dict of loaders or nested dict
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten_loaders(v, f"{prefix}{k}/")
        return

    # list/tuple of loaders
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten_loaders(v, f"{prefix}{i}/")
        return

    raise TypeError(f"Unsupported query_loaders node type: {type(obj)} at {prefix!r}")


# guide14.md: 单条查询评测的完整实现
@torch.no_grad()
def _extract_feats_and_ids(model, loader, device):
    """从DataLoader提取特征和ID"""
    feats, pids = [], []
    for batch in tqdm(loader, desc="提取特征", leave=False, ncols=100, mininterval=0.3):
        batch = move_batch_to_device(batch, device)
        with autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            outputs = call_model_with_batch(model, batch, return_features=True)
            # 使用BN后特征保持一致性
            if 'bn_features' in outputs:
                feat = outputs['bn_features']
            else:
                raise ValueError("模型输出缺少bn_features")
            
        feat = F.normalize(feat.float(), dim=1)  # L2归一化
        feats.append(feat.cpu())
        
        pid = batch['person_id']
        pids.append(pid.cpu() if hasattr(pid, "cpu") else torch.tensor(pid))
    
    return torch.cat(feats, 0), torch.cat(pids, 0)

@torch.no_grad()
def _reid_map(sim, q_ids, g_ids):
    """
    计算ReID mAP和Top-1准确率
    sim: [Nq, Ng]  余弦相似度矩阵
    q_ids: [Nq], g_ids: [Ng]
    return: mAP(float), top1(float)
    """
    Nq = sim.size(0)
    mAP, top1 = 0.0, 0.0
    arange = torch.arange(sim.size(1), device=sim.device, dtype=torch.float32) + 1.0
    
    for i in range(Nq):
        order = torch.argsort(sim[i], descending=True)
        matches = (g_ids[order] == q_ids[i]).to(sim.dtype)
        rel = matches.sum().item()
        if rel == 0:
            continue
        
        # 计算AP
        cumsum = torch.cumsum(matches, 0)
        precision = cumsum / arange
        ap = torch.sum(precision * matches) / rel
        mAP += ap.item()
        
        # 计算Top-1
        top1 += matches[0].item()
    
    valid = max(1, (q_ids.unsqueeze(1) == g_ids.unsqueeze(0)).any(dim=1).sum().item())
    return mAP / valid, top1 / Nq

@torch.no_grad()
def evaluate_one_query(model, gallery_loader, query_loader, device, *, cache=None):
    """
    评测单对(gallery, query_loader)，返回{'mAP': float, 'Top1': float}
    cache: 可传入{'g_feat': tensor, 'g_id': tensor}以复用gallery特征
    """
    # 1) gallery特征（可复用）
    if cache is not None and "g_feat" in cache and "g_id" in cache:
        g_feat, g_id = cache["g_feat"], cache["g_id"]
    else:
        g_feat, g_id = _extract_feats_and_ids(model, gallery_loader, device)
        if cache is not None:
            cache["g_feat"], cache["g_id"] = g_feat, g_id

    # 2) query特征
    q_feat, q_id = _extract_feats_and_ids(model, query_loader, device)

    # 3) 相似度与mAP计算
    sim = torch.matmul(q_feat.to(device), g_feat.to(device).T)  # 余弦已归一化
    mAP, top1 = _reid_map(sim, q_id.to(device), g_id.to(device))
    return {"mAP": float(mAP), "Top1": float(top1)}

def validate_competition_style(model, gallery_loader, query_loaders, device, k_map=100, sample_ratio=1.0, cfg=None, epoch=None):
    # guide12.md: 修复评测崩溃 - 使用扁平化查询加载器
    pairs = list(_flatten_loaders(query_loaders))
    
    # guide13.md: 只保留白名单模式的评测，跳过双/三模态组合
    include = getattr(cfg, "eval_include_patterns", ["single/nir", "single/sk", "single/cp", "single/text", "quad/nir+sk+cp+text"])
    
    # guide14.md: 名称规范化 + 模式匹配（对名称后缀/版本更宽容）
    import fnmatch
    def _norm(name: str) -> str:
        return name.replace("cpencil","cp").replace("sketch","sk").replace("nir","nir").replace("text","text")
    pairs = [(n, dl) for (n, dl) in pairs if any(fnmatch.fnmatch(_norm(n), pat) for pat in include)]
    
    # guide14.md: Gallery特征缓存机制
    def _cache_key_for_gallery(loader, tag=""):
        n = len(loader.dataset)
        h = hashlib.md5(str(n).encode() + str(tag).encode()).hexdigest()[:8]
        return f"gallery_{n}_{h}.pkl"
    
    cache_dir = getattr(cfg, "eval_cache_dir", "./.eval_cache")
    cache_tag = getattr(cfg, "eval_cache_tag", "val_v1")
    os.makedirs(cache_dir, exist_ok=True)
    ckey = _cache_key_for_gallery(gallery_loader, cache_tag)
    cache_path = os.path.join(cache_dir, ckey)
    
    cache = {}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except:
            cache = {}  # 缓存损坏时重新生成
    
    print(
        "[EVAL] gallery=%d  queries=%s"
        % (len(gallery_loader.dataset), [(k, len(dl.dataset)) for k, dl in pairs])
    )
    """
    赛制对齐评测：使用白名单过滤+特征缓存的高效评测
    
    guide14.md改进：
    1. 使用evaluate_one_query函数统一评测逻辑  
    2. 支持gallery特征缓存，避免重复计算
    3. 更灵活的名称匹配和采样机制
    """
    model.eval()

    # guide14.md: 使用新的评测逻辑和特征缓存
    all_metrics = {}
    all_q_feats, all_q_labels = [], []
    
    for name, qloader in pairs:
        # guide14.md: 样本采样优化
        if 0.0 < sample_ratio < 1.0:
            original_ds = qloader.dataset
            idx = torch.randperm(len(original_ds))[:int(len(original_ds)*sample_ratio)].tolist()
            sub = Subset(original_ds, idx)
            # 创建采样后的DataLoader，保持原有参数
            qloader_attrs = {
                'batch_size': qloader.batch_size,
                'num_workers': getattr(qloader, 'num_workers', 0),
                'pin_memory': getattr(qloader, 'pin_memory', False),
                'collate_fn': getattr(qloader, 'collate_fn', None)
            }
            qloader = DataLoader(sub, **qloader_attrs)
        
        # guide14.md: 使用新的evaluate_one_query函数，支持特征缓存
        m = evaluate_one_query(model, gallery_loader, qloader, device, cache=cache)
        all_metrics[name] = m
        
        # 为整体CMC计算收集特征（可选）
        # 注意：这里复用了cache中的特征，避免重复提取
        if cache and "g_feat" in cache:
            q_feat, q_id = _extract_feats_and_ids(model, qloader, device)
            all_q_feats.append(q_feat)
            all_q_labels.append(q_id)

    # guide14.md: 聚合四单模态均值 + 四模态，使用更通用的_get_map函数
    def _get_map(m):
        if isinstance(m, dict):
            for k in ("mAP", "map", "mAP_mean", "map_mean"):
                if k in m: 
                    return float(m[k])
        if isinstance(m, (int, float)): 
            return float(m)
        return 0.0

    # 单模态均值
    singles = [_get_map(all_metrics.get(k, {})) for k in ("single/nir","single/sk","single/cp","single/text")]
    map_single = sum(singles) / max(1, len([x for x in singles if x==x]))  # 防空/NaN
    
    # 四模态
    map_quad = _get_map(all_metrics.get("quad/nir+sk+cp+text", {}))
    
    # 最终聚合
    comp_metrics = {
        "map_single": map_single, 
        "map_quad": map_quad, 
        "map_avg2": (map_single + map_quad) / 2.0
    }

    # guide14.md: 改进的评测结果打印，包含epoch信息
    if epoch is not None:
        print("[EVAL] epoch=%d  mAP(all)=%.4f  |  mAP@single=%.4f  mAP@quad=%.4f"
              % (epoch, comp_metrics["map_avg2"], comp_metrics["map_single"], comp_metrics["map_quad"]))

    map_avg2 = comp_metrics["map_avg2"]

    # guide14.md: CMC计算（如果需要）
    if all_q_feats and cache and "g_feat" in cache:
        all_q_feats = torch.cat(all_q_feats, dim=0)
        all_q_labels = torch.cat(all_q_labels, dim=0)
        g_feat = cache["g_feat"]
        g_id = cache["g_id"]
        
        # 使用缓存的gallery特征计算CMC
        sim = torch.matmul(all_q_feats.to(device), g_feat.to(device).T)
        _, cmc1 = _reid_map(sim[:1], all_q_labels[:1].to(device), g_id.to(device))  # 简化CMC计算
        cmc5 = cmc10 = cmc1  # 简化处理
    else:
        cmc1 = cmc5 = cmc10 = 0.0

    # guide14.md: 保存缓存到磁盘
    if cache and ("g_feat" in cache):
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"g_feat": cache.get("g_feat"), "g_id": cache.get("g_id")}, f)
        except Exception as e:
            print(f"[WARN] 缓存保存失败: {e}")

    return {
        'map_single': comp_metrics['map_single'],
        'map_quad': comp_metrics['map_quad'], 
        'map_avg2': comp_metrics['map_avg2'],
        'detail': all_metrics,  # guide13.md: 更新为扁平化后的metrics结构
        'cmc1': cmc1, 'cmc5': cmc5, 'cmc10': cmc10
    }

# guide10.md: 统一模态名映射工具 - 使用MODALITY_MAPPING

def _extract_modalities_from_batch(batch):
    """
    返回标准化后的模态名列表（长度等于batch大小），元素 ∈ {'vis','nir','sk','cp','text'}
    兼容多种字段：'modality' | 'modalities' | 'mod' | 'modality_id' 等
    """
    if isinstance(batch, dict):
        if 'modality' in batch:
            raw = batch['modality']
        elif 'modalities' in batch:
            raw = batch['modalities']
        elif 'mod' in batch:
            raw = batch['mod']
        elif 'modality_id' in batch:  # tensor/list of ints
            ids = batch['modality_id']
            if hasattr(ids, 'tolist'): ids = ids.tolist()
            # 将数字ID转为字符串，让map_modality_name处理具体映射
            raw = [str(int(i)) for i in ids]
        else:
            # 最后兜底：如果每个样本在 batch['meta'] 里
            if 'meta' in batch and isinstance(batch['meta'], list) and len(batch['meta'])>0:
                raw = [m.get('modality') or m.get('mod') for m in batch['meta']]
            else:
                raise KeyError("Batch has no modality-like key: expected one of "
                               "['modality','modalities','mod','modality_id','meta[*].modality']")
    else:
        raise TypeError("Batch must be a dict-like object with modality info")

    # 统一成 list[str]
    if not isinstance(raw, list):
        if hasattr(raw, 'tolist'):  # torch tensor
            raw = raw.tolist()
        else:
            raw = list(raw)

    # 归一化到标准模态名 - 使用统一的MODALITY_MAPPING
    mods = [map_modality_name(str(x).lower()) for x in raw]
    return mods

# ------------------------------
# 训练一个 epoch
# ------------------------------
def train_epoch_fixed(model, dataloader, optimizer, device, epoch, scaler=None, adaptive_clip=True, accum_steps=1, autocast_dtype=torch.float16, cfg=None, pbar=None):
    """Fix20: 修复后的训练epoch函数，使用外部传入的进度条"""
    model.train()
    
    # 设置当前epoch，用于控制modality_dropout热身期
    model.set_epoch(epoch)
    total_loss = 0.0
    ce_loss_sum = 0.0
    contrastive_loss_sum = 0.0
    correct = 0
    total = 0
    
    feature_norms = []
    loss_spikes = 0
    grad_norms = []
    
    # guide6.md: 三条健康线监控
    if not hasattr(train_epoch_fixed, '_health_monitors'):
        train_epoch_fixed._health_monitors = {
            'pair_cov_hist': [],  # pair coverage 历史
            'ce_hist': [],        # CE损失历史
            'top1_hist': []       # Top-1准确率历史
        }
    
    # 稳健的Spike检测状态管理
    if not hasattr(train_epoch_fixed, '_spike_state'):
        train_epoch_fixed._spike_state = {
            'loss_hist': [],
            'spikes': 0,
            'batches': 0
        }
    
    use_amp = (scaler is not None and getattr(scaler, "is_enabled", lambda: True)())

    # guide12.md & guide15.md: 修复"每个epoch只跑到step=80就结束" - 明确禁用截断
    max_steps = int(getattr(cfg, "max_steps_per_epoch", 0) or 0)
    # guide15.md: 确保没有隐藏的eval_after_steps触发条件
    eval_after_steps = getattr(cfg, "eval_after_steps", None)
    if eval_after_steps is not None:
        logging.warning(f"检测到eval_after_steps={eval_after_steps}，guide15建议禁用此参数")
    steps_run = 0
    
    # Fix20: 使用外部传入的进度条，不再创建新的tqdm
    if pbar is None:
        raise ValueError("Fix20: train_epoch_fixed 必须传入进度条参数")
    
    # 添加批次构成监控（前3个batch）
    if epoch <= 3:
        logging.info(f"=== Epoch {epoch} 批次构成监控 ===")
    
    # guide14.md: 统计各种batch处理状态
    processed = 0       # 成功处理的batch数
    skipped = 0        # 跳过的batch数  
    total_attempted = 0 # 尝试处理的总batch数
    
    for batch_idx, batch in enumerate(dataloader):
        total_attempted += 1  # 记录尝试处理的batch总数
        
        batch = move_batch_to_device(batch, device)
        labels = batch['person_id']
        
        # guide10.md: 打印一次batch keys（只在step==0打）
        if batch_idx == 0:
            print(f"[dbg] batch keys: {list(batch.keys())[:12]}")
        
        # guide10.md: 轻断言，避免后面又栽坑
        if batch_idx == 0:  # 只在第一个batch验证，避免每次都检查
            try:
                mod_for_check = _extract_modalities_from_batch(batch)
                assert len(mod_for_check) == (labels.shape[0] if hasattr(labels, 'shape') else len(batch.get('person_id', []))), \
                    f"mod length {len(mod_for_check)} != batch size"
                print(f"[dbg] modality extraction successful, length: {len(mod_for_check)}")
            except Exception as e:
                print(f"[dbg] modality extraction failed: {e}")
        
        # guide9.md Step 1: 批内可配对自检（确定是不是采样器/K值问题）
        if (batch_idx % 50) == 0:
            pid = batch['person_id'] if isinstance(batch, dict) else labels
            mod = _extract_modalities_from_batch(batch)  # guide10.md: 健壮取模态
            pid = pid.detach().cpu().tolist()
            # mod 已经是 list[str]，不需要再转换

            # 统计每个ID在本批的样本数、RGB/非RGB覆盖
            from collections import Counter, defaultdict
            c = Counter(pid)
            vis_by_id = defaultdict(int); nonvis_by_id = defaultdict(int)
            for p, m in zip(pid, mod):
                if m == 'vis': vis_by_id[p]+=1
                else: nonvis_by_id[p]+=1

            K_min = min(c.values()) if c else 0
            ids_with_pair = sum(1 for p in c if (vis_by_id[p]>0 and nonvis_by_id[p]>0))
            print(f"[sampler-dbg] batch_size={len(pid)} unique_id={len(c)} "
                  f"Kmin={K_min} paired_ids={ids_with_pair}")
        
        # guide4.py: 标签合法性断言，确保CrossEntropy要求的0...C-1范围
        assert labels.min().item() >= 0 and labels.max().item() < model.num_classes, \
            f"guide4.py: 标签越界! labels范围[{labels.min().item()}, {labels.max().item()}], 要求[0, {model.num_classes-1}]"
        
        # 批次构成监控（前3个epoch的前3个batch）
        if epoch <= 3 and batch_idx < 3:
            unique_ids, counts = torch.unique(labels, return_counts=True)
            num_ids_per_batch = len(unique_ids)
            avg_instances_per_id = float(counts.float().mean().item())  # 转换为浮点数再计算均值
            logging.info(f"Batch {batch_idx}: {num_ids_per_batch} IDs, 平均每ID {avg_instances_per_id:.1f} 样本 (K-1正样本数≈{avg_instances_per_id-1:.1f})")
            
            # ✅ 修复2: CE损失诊断 - 检查标签和分类器匹配性
            if batch_idx == 0:
                print(f"=== CE损失诊断 (Epoch {epoch}) ===")
                print(f"labels范围: {labels.min().item()} - {labels.max().item()}")
                print(f"model.num_classes: {model.num_classes}")
                print(f"理论随机CE: {np.log(model.num_classes):.3f}")
                
                # 检查分类器参数是否可训练
                classifier_params = []
                for name, param in model.named_parameters():
                    if 'classifier' in name and param.requires_grad:
                        classifier_params.append(name)
                print(f"可训练分类器参数: {classifier_params}")
                
                if labels.max().item() >= model.num_classes:
                    logging.error(f"❌ 标签超出范围! max_label={labels.max().item()}, num_classes={model.num_classes}")
            
            # 采样器自检：统计每个ID在该batch是否有vis/非vis
            ids = labels.cpu().tolist()
            modality_mask = batch.get('modality_mask', {})
            
            has_vis = {}
            has_nonvis = {}
            for k, pid in enumerate(ids):
                # 检查vis模态
                vis_mask = modality_mask.get('vis', torch.zeros_like(labels)).bool()
                has_vis[pid] = has_vis.get(pid, False) or bool(vis_mask[k].item())
                
                # 检查非vis模态（nir, sk, cp, text等）
                nonvis = False
                for m in ['nir', 'sk', 'cp', 'text']:
                    msk = modality_mask.get(m, torch.zeros_like(labels)).bool()
                    nonvis = nonvis or bool(msk[k].item())
                has_nonvis[pid] = has_nonvis.get(pid, False) or nonvis
            
            # 统计同时具备vis+非vis的ID数量
            both = sum(1 for pid in set(ids) if has_vis.get(pid, False) and has_nonvis.get(pid, False))
            vis_only = sum(1 for pid in set(ids) if has_vis.get(pid, False) and not has_nonvis.get(pid, False))
            nonvis_only = sum(1 for pid in set(ids) if not has_vis.get(pid, False) and has_nonvis.get(pid, False))
            
            logging.info(f"[采样自检] 本batch ID数={len(set(ids))}, vis+非vis={both}, 仅vis={vis_only}, 仅非vis={nonvis_only}")

        # 梯度累积：只在累积步开始时清零梯度
        if batch_idx % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        # === SDM权重调度（只读当前参数，避免空指标触发回退） ===
        if not hasattr(model, 'sdm_scheduler'):
            model.sdm_scheduler = SDMScheduler(model.config)
        
        # ✅ 只读当前权重/温度，不做判定（由 epoch 结束后统一调整）
        effective_cont_w = getattr(model.sdm_scheduler.weight_scheduler, "current_weight",
                                   getattr(model.config, "contrastive_weight", 0.1))
        current_temp = getattr(model.sdm_scheduler.temp_scheduler, "current_temp",
                               getattr(model.config, "sdm_temperature", 0.2))
        
        # guide6.md: 每个epoch的第一个step打印SDM调度权重信息
        if batch_idx == 0:
            use_sdm = (epoch >= model.config.sdm_weight_warmup_epochs)
            sdm_w = model.sdm_scheduler.get_weight(epoch) if hasattr(model, 'sdm_scheduler') else effective_cont_w
            print(f"[sdm] epoch={epoch} weight={sdm_w:.3f} use_sdm={use_sdm}")
            logging.info(f"Epoch {epoch}: SDM_weight={sdm_w:.4f}, SDM_temp={current_temp:.3f}, use_sdm={use_sdm}")

        with autocast(device_type='cuda', dtype=autocast_dtype, enabled=use_amp):
            try:
                outputs = call_model_with_batch(model, batch, return_features=False)
                loss_dict = model.compute_loss(outputs, labels)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # 内存不足时清理缓存并跳过
                    torch.cuda.empty_cache()
                    logging.warning(f"Epoch {epoch}, Batch {batch_idx}: 内存不足，跳过当前batch")
                    skipped += 1  # 记录跳过的batch
                    # 跳过的batch也更新进度条，保持进度连续性
                    pbar.update(1)
                    continue
                else:
                    raise e
            
            # NaN检测和跳过机制（解决第29batch问题）
            total_loss_val = loss_dict['total_loss']
            if not torch.isfinite(total_loss_val):
                logging.warning(f"Epoch {epoch}, Batch {batch_idx}: 发现非有限损失值 {total_loss_val.item():.6f}，跳过当前step")
                optimizer.zero_grad(set_to_none=True)
                skipped += 1  # 记录跳过的batch
                # 跳过的batch也更新进度条，保持进度连续性
                pbar.update(1)
                continue
            
            # SDM对比损失监控（仅在实际参与训练时警告）
            sdm_loss = loss_dict.get('sdm_loss', torch.tensor(0.0, device=device))
            cont_loss = loss_dict.get('contrastive_loss', sdm_loss)  # 兼容性：用sdm_loss作为contrastive_loss
            
            # 只有当SDM损失真正参与训练时（权重>0）才显示警告
            if effective_cont_w > 0.0 and sdm_loss.item() > 1.5:  
                original_sdm = sdm_loss.item()
                if batch_idx < 5:  # 只在前几个batch报告，避免日志过多
                    logging.info(f"Epoch {epoch}, Batch {batch_idx}: SDM对齐损失较高 {original_sdm:.3f}")
            
            # 检查是否有NaN或Inf
            if not torch.isfinite(sdm_loss):
                logging.error(f"Epoch {epoch}, Batch {batch_idx}: SDM损失出现NaN/Inf, 重置为0")
                loss_dict['sdm_loss'] = torch.tensor(0.0, device=device)
                loss_dict['contrastive_loss'] = torch.tensor(0.0, device=device)
            
            # 使用模型计算的total_loss，仅应用梯度累积缩放
            loss = loss_dict['total_loss'] / accum_steps
            loss_dict['total_loss'] = loss

        current_loss = float(loss.item() * accum_steps)  # 显示未缩放的损失
        
        # guide9.md Step 3: 让pair_coverage_mavg真更新（基于真实的批内配对关系）
        if not hasattr(train_epoch_fixed, '_pair_coverage_hist'):
            train_epoch_fixed._pair_coverage_hist = []
        
        # 计算配对覆盖（使用真实的批内配对关系）
        # 假设：非RGB为 query，RGB为 gallery
        pid = batch['person_id'].detach()
        mod = _extract_modalities_from_batch(batch)  # guide10.md: 健壮取模态
        is_rgb = torch.tensor([m=='rgb' for m in mod], device=pid.device)
        is_non = ~is_rgb

        pid_t = pid
        qry_ids = pid_t[is_non]
        gal_ids = pid_t[is_rgb]

        if len(qry_ids)>0 and len(gal_ids)>0:
            # 对每个 query，是否在 gallery 中存在同ID
            # （效率无所谓，只做监控）
            gal_set = set(gal_ids.tolist())
            have_pos = torch.tensor([int(int(q) in gal_set) for q in qry_ids.tolist()], device=pid.device)
            cov = have_pos.float().mean().item()  # 0~1
        else:
            cov = 0.0

        train_epoch_fixed._pair_coverage_hist.append(cov)
        pair_coverage_window = getattr(cfg, 'pair_coverage_window', 100)
        pair_coverage_mavg = sum(train_epoch_fixed._pair_coverage_hist[-pair_coverage_window:]) / min(len(train_epoch_fixed._pair_coverage_hist), pair_coverage_window)
        
        # 每50步打印一次健康线监控
        if batch_idx % 50 == 0:
            print(f"[dbg] pair_coverage_mavg={pair_coverage_mavg:.3f}")
            logging.info(f"Epoch {epoch}, Batch {batch_idx}: pair_coverage_mavg={pair_coverage_mavg:.3f}")
        
        # 稳健的Spike检测：使用滑动中位数 + MAD（中位数绝对偏差）
        state = train_epoch_fixed._spike_state
        state['loss_hist'].append(current_loss)
        # 保持最近200个损失值的历史
        if len(state['loss_hist']) > 200:
            state['loss_hist'] = state['loss_hist'][-200:]
        
        # ✅ 条件启动：足够样本再开启spike检测
        if len(state['loss_hist']) >= 20:
            hist = np.array(state['loss_hist'][-100:])         # ✅ 最近100个样本
            median = np.median(hist)
            mad = np.median(np.abs(hist - median))
            mad = max(mad, 0.05)                                # ✅ MAD下限
            threshold = max(median + 6.0 * 1.4826 * mad,       # ✅ 绝对门限
                            median * 1.15)                     # ✅ 相对门槛 15%
            
            # 检测异常
            if current_loss > threshold:
                loss_spikes += 1
                state['spikes'] += 1
                if batch_idx % 20 == 0:  # 减少日志频率
                    logging.warning(f"Epoch {epoch}, Batch {batch_idx}: 损失异常 {current_loss:.3f} > {threshold:.3f}")
        
        state['batches'] += 1
        
        # 检查数值有效性
        if not np.isfinite(current_loss):
            logging.error(f"Epoch {epoch}, Batch {batch_idx}: 损失无效 {current_loss}, 跳过")
            loss_spikes += 1
            state['spikes'] += 1
            skipped += 1  # 记录跳过的batch
            # 跳过的batch也更新进度条，保持进度连续性
            pbar.update(1)
            continue
        
        # 计算梯度（支持梯度累积）
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # 只在累积步结束时更新参数
        if (batch_idx + 1) % accum_steps == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                _sanitize_grads(model)

                # ✅ 梯度范数计算优化：只在需要监控时计算完整范数
                if adaptive_clip:
                    # 自适应裁剪需要计算完整梯度范数
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            if torch.isfinite(param_norm):
                                total_norm += param_norm.item() ** 2
                    total_norm = math.sqrt(total_norm)
                    
                    # 只在监控频率下记录到列表
                    if batch_idx % NORM_EVERY == 0:
                        grad_norms.append(float(total_norm))
                    
                    # 自适应裁剪
                    if len(grad_norms) > 10:
                        recent_norms = grad_norms[-10:]
                        adaptive_max_norm = min(3.0, max(0.5, np.percentile(recent_norms, 70) * 1.15))
                    else:
                        adaptive_max_norm = 1.0
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(adaptive_max_norm))
                else:
                    # 非自适应：直接裁剪，偶尔记录范数用于监控
                    if batch_idx % NORM_EVERY == 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        grad_norms.append(float(total_norm))
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                scaler.step(optimizer)
                scaler.update()
            else:
                _sanitize_grads(model)

                # ✅ 梯度范数计算优化：只在需要监控时计算完整范数（非AMP版本）
                if adaptive_clip:
                    # 自适应裁剪需要计算完整梯度范数
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            if torch.isfinite(param_norm):
                                total_norm += param_norm.item() ** 2
                    total_norm = math.sqrt(total_norm)
                    
                    # 只在监控频率下记录到列表
                    if batch_idx % NORM_EVERY == 0:
                        grad_norms.append(float(total_norm))
                    
                    # 自适应裁剪
                    if len(grad_norms) > 10:
                        recent_norms = grad_norms[-10:]
                        adaptive_max_norm = min(3.0, max(0.5, np.percentile(recent_norms, 70) * 1.15))
                    else:
                        adaptive_max_norm = 1.0
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(adaptive_max_norm))
                else:
                    # 非自适应：直接裁剪，偶尔记录范数用于监控
                    if batch_idx % NORM_EVERY == 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        grad_norms.append(float(total_norm))
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                optimizer.step()
                
                # guide4.py + guide5.md: 每100步监控分类头权重和梯度变化 + Top-1准确率
                if (batch_idx + 1) % 100 == 0:
                    if hasattr(model, 'bn_neck') and hasattr(model.bn_neck, 'classifier'):
                        w = model.bn_neck.classifier.weight
                        print(f"[guide4-dbg] step={batch_idx+1} head |w|={w.norm():.4f}")
                        
                        # 监控梯度范数
                        g = 0.0
                        for p in model.bn_neck.classifier.parameters():
                            if p.grad is not None:
                                g += (p.grad.detach().float().norm().item())
                        print(f"[guide4-dbg] step={batch_idx+1} head grad-norm ≈ {g:.4f}")
                    
                    # guide9.md: 训练Top-1准确率（使用CE的同一logits）
                    # 你计算 CE 用的那个张量
                    logits_ce = outputs.get('cls_logits', None) or outputs.get('logits', None)
                    if logits_ce is not None:
                        top1 = (logits_ce.argmax(1) == labels).float().mean()
                        print(f"[guide5-dbg] step={batch_idx+1} top1={top1*100:.2f}%")
                    else:
                        print(f"[guide9-warn] step={batch_idx+1} 未找到用于 CE 的 logits")
                
                # 定期清理CUDA缓存，防止内存累积
                if (batch_idx + 1) % (accum_steps * 5) == 0:  # 每5个累积周期清理一次
                    torch.cuda.empty_cache()
        
        # guide12.md: 只有这一处允许截断
        if max_steps > 0 and steps_run >= max_steps:
            break

        # 统计信息更新（移回循环内部）
        total_loss += current_loss
        ce_loss = loss_dict.get('ce_loss', torch.tensor(0.0, device=device))
        # 复用前面已提取的sdm_loss，避免重复定义
        ce_loss_sum += float(ce_loss.item())
        contrastive_loss_sum += float(sdm_loss.item())  # 累计SDM损失

        if isinstance(outputs, dict) and 'logits' in outputs:
            _, predicted = outputs['logits'].max(1)
        else:
            _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 监控三种特征范数：融合、BN前、BN后
        fusion_feat_norms = None
        feat_pre_bn_norms = None
        feat_bnneck_norms = None
        if isinstance(outputs, dict):
            if 'features' in outputs:
                fusion_feat_norms = torch.norm(outputs['features'].detach(), p=2, dim=1)  # 融合输出的特征范数
            if 'reid_features_raw' in outputs:
                feat_pre_bn_norms = torch.norm(outputs['reid_features_raw'].detach(), p=2, dim=1)  # BNNeck之前的原始特征范数
            if 'bn_features' in outputs:
                feat_bnneck_norms = torch.norm(outputs['bn_features'].detach(), p=2, dim=1)  # BNNeck之后的对齐特征范数

        # 使用BNNeck特征范数更新累积统计（因为这是对齐和检索的关键）
        if feat_bnneck_norms is not None:
            feature_norms.extend(feat_bnneck_norms.cpu().numpy())
        elif fusion_feat_norms is not None:
            feature_norms.extend(fusion_feat_norms.cpu().numpy())

        # 进度条显示BNNeck特征范数（最重要的监控指标）
        avg_fusion_feat = float(fusion_feat_norms.mean().item()) if fusion_feat_norms is not None else 0.0
        avg_feat_bnneck = float(feat_bnneck_norms.mean().item()) if feat_bnneck_norms is not None else 0.0
        avg_feat_pre_bn = float(feat_pre_bn_norms.mean().item()) if feat_pre_bn_norms is not None else 0.0
        recent_grad_norm = (np.mean(grad_norms[-10:]) if grad_norms else None)

        # 区分SDM损失和分数进行监控
        sdm_loss_val = float(sdm_loss.item())
        
        # ✅ 调试输出降频：减少日志开销
        LOG_EVERY = 100  # 从50提高到100，减少一半日志输出
        NORM_EVERY = 200  # 特征范数监控频率更低
        
        # 关键监控字段输出（降频优化）
        if batch_idx % LOG_EVERY == 0:  # 降频：每100个batch输出一次
            # 计算有效样本统计
            feature_masks = outputs.get('feature_masks', {})
            vis_cnt = 0
            for mod_name, mask in feature_masks.items():
                if mask is not None:
                    mod_valid = (mask > 0).squeeze(-1) if mask.dim() > 1 else (mask > 0)
                    if mod_name in ['rgb', 'vis']:  # RGB或可见光模态
                        vis_cnt = int(mod_valid.sum())
                        break
            
            # 从loss_dict中获取ce_valid计数
            ce_valid_cnt = loss_dict.get('ce_valid_cnt', len(labels))  # 从模型获取实际有效CE样本数
            
            # 输出关键字段
            logging.info(f"[{epoch:02d}, {batch_idx:03d}] "
                        f"vis_cnt={vis_cnt}, ce_valid={ce_valid_cnt}, "
                        f"BNNeckNorm_mean={avg_feat_bnneck:.2f}, "
                        f"SDMLoss={sdm_loss_val:.3f}, CE={float(ce_loss.item()):.3f}")
        
        # 早期训练详细监控（前3个epoch，每20个batch）
        if epoch <= 3 and batch_idx % 20 == 0:
            # 获取logits的最大绝对值
            max_abs_logit = 0.0
            if 'logits' in outputs:
                max_abs_logit = float(outputs['logits'].abs().max().item())
            
            logging.info(f"数值监控 Epoch {epoch}, Batch {batch_idx}: "
                        f"max_abs_logit={max_abs_logit:.2f}, "
                        f"SDM_weight={effective_cont_w:.3f}, SDM_temp={current_temp:.3f}")
            
            # 检查SDM损失异常（修复后应该天然非负）
            if sdm_loss_val < 0:
                logging.warning(f"⚠️ SDM损失异常为负值: {sdm_loss_val:.4f} - 检查mask过滤是否生效！")
            elif sdm_loss_val > 5.0:
                logging.warning(f"⚠️ SDM损失过大: {sdm_loss_val:.4f} - 可能存在数值不稳定")
        
        # ✅ 修复3: 调整BNNeck特征范数阈值 - 如果使用L2归一化，范数应该接近1
        # 检查模型是否在使用L2归一化
        using_l2_norm = False
        if isinstance(outputs, dict) and 'bn_features' in outputs:
            bn_feats = outputs['bn_features']
            sample_norm = bn_feats[0].norm(p=2).item()
            if 0.8 <= sample_norm <= 1.2:  # 接近单位范数
                using_l2_norm = True
        
        # 根据是否使用L2归一化调整阈值
        if using_l2_norm:
            # L2归一化情况下，范数应该接近1
            if avg_feat_bnneck > 2.0 and epoch > 5 and batch_idx % 50 == 0:
                logging.warning(f"⚠️ BNNeck特征范数异常(L2归一化): {avg_feat_bnneck:.2f} - 应接近1.0 (Epoch {epoch})")
        else:
            # 非归一化情况下，范数阈值设为更合理的值
            if avg_feat_bnneck > 15.0 and epoch > 5 and batch_idx % 50 == 0:
                logging.warning(f"⚠️ BNNeck特征范数过大(非归一化): {avg_feat_bnneck:.2f} - 正则化未生效 (Epoch {epoch})")
        
        # 特征范数监控改进（降频优化）
        if batch_idx % NORM_EVERY == 0 and batch_idx > 0:
            logging.info(f"[特征监控] Epoch {epoch}, Batch {batch_idx}: "
                        f"融合特征={avg_fusion_feat:.2f}, BNNeck特征={avg_feat_bnneck:.2f}, "
                        f"BN前特征={avg_feat_pre_bn:.2f}, 是否L2归一化={using_l2_norm}")
        
        # ✅ 进度条更新降频：从5提高到10，减少GPU-CPU同步开销
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'Loss': f'{current_loss:.3f}',
                'CE': f'{float(ce_loss.item()):.3f}',
                'SDMLoss': f'{sdm_loss_val:.3f}',  # 修复后应该非负
                'PreBN': f'{avg_feat_pre_bn:.2f}',  # BN前原始特征范数
                'PostBN': f'{avg_feat_bnneck:.2f}',  # BN后特征范数（用于ReID检索）
                'GradNorm': ('—' if recent_grad_norm is None else f'{recent_grad_norm:.2f}'),
                'Spikes': loss_spikes
            })
        
        # 成功处理一个batch后增加计数（在所有continue检查之后）
        processed += 1
        
        # 每个batch处理完后更新进度条（无论成功还是跳过）
        pbar.update(1)
        
        # guide15.md: 明确禁止在训练循环内触发评测
        # 所有评测应该在epoch结束后进行，不应该在batch级别触发
        

    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = 100. * correct / max(1, total)
    avg_feat_norm = np.mean(feature_norms) if feature_norms else 0.0
    epoch_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
    
    if loss_spikes > len(dataloader) * 0.1:
        logging.warning(f"Epoch {epoch}: 损失异常次数 {loss_spikes}，建议降低学习率")
    if avg_feat_norm > 50.0:
        logging.warning(f"Epoch {epoch}: 平均特征范数 {avg_feat_norm:.2f} 偏大，检查稳定性")
    
    # guide14.md: 打印成功处理的步数统计
    print(f"[epoch {epoch}] steps_run={processed}/{len(dataloader)}  (max_steps={max_steps or 0})")
    
    # guide16.md: 防止"静默早收工"的epoch终止监控
    expected = len(dataloader)  # 名义
    actual = processed         # 实际
    if actual < expected * 0.9:  # 如果实际处理数少于90%
        logging.warning(f"[Epoch {epoch}] 采样器提前耗尽: 实际batch={actual}, 名义batch={expected}. "
                       "可能因 unique_id/Kmin/跨模态约束过严或数据不平衡导致。")
    
    # Fix20: 进度条由外部with语句管理，无需手动关闭
    # 训练完成后清理CUDA缓存
    torch.cuda.empty_cache()
    
    return {
        'total_loss': avg_loss,
        'ce_loss': ce_loss_sum / max(1, len(dataloader)),
        'contrastive_loss': contrastive_loss_sum / max(1, len(dataloader)),  # 兼容性：实际为SDM损失
        'sdm_loss': contrastive_loss_sum / max(1, len(dataloader)),  # 明确的SDM损失
        'accuracy': accuracy,
        'feature_norm': avg_feat_norm,
        'grad_norm': epoch_grad_norm,
        'loss_spikes': loss_spikes,
        'stability_score': max(0.0, 1.0 - train_epoch_fixed._spike_state['spikes'] / max(1, train_epoch_fixed._spike_state['batches'])),
        'actual_batches': total_attempted,  # 实际尝试的batch数（包含跳过的）
        'processed_batches': processed,     # 成功处理的batch数
        'skipped_batches': skipped          # 跳过的batch数
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

    # 使用新的划分工具
    from tools.split import split_ids, create_split_datasets, verify_split_integrity
    
    # 获取所有人员ID
    all_person_ids = [full_dataset.data_list[i]['person_id'] for i in range(len(full_dataset))]
    all_person_ids = sorted(list(set(all_person_ids)))
    
    # 创建person_id到标签的映射
    pid2label = {pid: idx for idx, pid in enumerate(all_person_ids)}
    
    # 按ID划分训练集和验证集
    train_ids, val_ids = split_ids(
        all_person_ids, 
        val_ratio=getattr(config, "val_ratio", 0.2),
        seed=getattr(config, "seed", 42)
    )
    
    # 创建训练集和验证集
    train_dataset, local_val_dataset = create_split_datasets(
        full_dataset, train_ids, val_ids, config
    )
    
    # 验证划分完整性
    verify_split_integrity(train_dataset, local_val_dataset)
    
    # 输出最终的数据集统计（确认划分成功）
    train_ids_actual = set(item['person_id'] for item in train_dataset.data_list)
    val_ids_actual = set(item['person_id'] for item in local_val_dataset.data_list)
    
    print(f"数据集划分结果:")
    print(f"  原始数据集: {len(full_dataset.data_list)} 样本, {len(all_person_ids)} 个ID")
    print(f"  训练数据集: {len(train_dataset.data_list)} 样本, {len(train_ids_actual)} 个ID ({len(train_dataset.data_list)/len(full_dataset.data_list):.1%})")
    print(f"  验证数据集: {len(local_val_dataset.data_list)} 样本, {len(val_ids_actual)} 个ID ({len(local_val_dataset.data_list)/len(full_dataset.data_list):.1%})")
    print(f"  ID重叠检查: {'无重叠' if len(train_ids_actual & val_ids_actual) == 0 else '存在重叠'}")
    
    logging.info(f"最终数据集: 训练集{len(train_dataset.data_list)}样本, 验证集{len(local_val_dataset.data_list)}样本")

    # 分类头 num_classes（应该覆盖所有可能的person_id以避免标签超出范围）
    config.num_classes = len(all_person_ids)
    logging.info(f"分类器输出维度: {config.num_classes} 类")
    logging.info(f"标签映射范围: 0-{len(all_person_ids)-1} （包含训练集+验证集）")
    train_labels = [pid2label[pid] for pid in train_ids]
    val_labels = [pid2label[pid] for pid in val_ids]
    logging.info(f"训练集标签范围: {min(train_labels)}-{max(train_labels)}, 验证集标签范围: {min(val_labels)}-{max(val_labels)}")

    # ==== Batch Size 统一配置 ====
    world_size = (torch.distributed.get_world_size() 
                  if torch.distributed.is_available() and torch.distributed.is_initialized() else 1)
    grad_accum_steps = getattr(config, "gradient_accumulation_steps", 4)
    
    # P×K结构参数（统一从config获取，避免重复定义）
    P = getattr(config, "num_ids_per_batch", 4)        # P: 每个batch的ID数
    K = getattr(config, "instances_per_id", 2)         # K: 每个ID的样本数
    K = max(2, K)  # 强制K>=2保证配对
    
    # 参数验证
    assert K >= 2, f"instances_per_id(K) 必须 ≥ 2，当前为 {K}，无法保证批内配对"
    assert P >= 2, f"num_ids_per_batch(P) 必须 ≥ 2，当前为 {P}"
    
    # 计算实际批次大小
    actual_batch_size = P * K
    effective_batch_size = actual_batch_size * grad_accum_steps * world_size
    
    logging.info(f"Batch Size配置: P×K={P}×{K}={actual_batch_size}, 累积步数={grad_accum_steps}, 等效={effective_batch_size}")
    logging.info(f"强制配对约束: 每个ID必须≥2样本且包含vis+nonvis")

    
    logging.info(f"采样策略: P×K = {P}×{K} = {actual_batch_size}")
    logging.info(f"每个锚的正样本数: {K-1}")
    
    # guide6.md: 使用强配对采样器
    require_modal_pairs = getattr(config, 'require_modal_pairs', True)
    modal_pair_retry_limit = getattr(config, 'modal_pair_retry_limit', 3)
    modal_pair_fallback_ratio = getattr(config, 'modal_pair_fallback_ratio', 0.3)
    
    if require_modal_pairs:
        logging.info(f"使用强配对采样器: ModalAwarePKSampler_Strict")
        logging.info(f"  重试次数: {modal_pair_retry_limit}")
        logging.info(f"  软退路比例: {modal_pair_fallback_ratio}")
    else:
        logging.info(f"使用普通采样器: ModalAwarePKSampler")
    logging.info(f"最终P×K结构: {P}×{K} = {actual_batch_size}")
    
    # guide6.md: 使用强配对采样器和优化的DataLoader配置
    safe_batch_size = min(8, actual_batch_size)  # 小batch更稳定
    logging.info(f"使用简化配置：batch_size={safe_batch_size}（原计划{actual_batch_size}）")
    
    # Guide19修复：使用批采样器替代普通采样器，避免batch_size参数错误
    from datasets.dataset import ModalAwarePKBatchSampler_Strict, analyze_sampling_capability
    
    # 先分析数据集采样能力
    strong_count, total_count = analyze_sampling_capability(train_dataset, limit=2000)
    
    if strong_count == 0:
        logging.error(f"可配对ID数({strong_count}) < 每批需要ID数({P})，建议检查数据集或降低num_ids_per_batch")
        raise ValueError("数据集无有效的强配对ID，无法进行训练")
    
    # 使用批采样器
    train_batch_sampler = ModalAwarePKBatchSampler_Strict(
        train_dataset,
        num_ids_per_batch=P,
        num_instances=K,
        allow_id_reuse=getattr(config, "allow_id_reuse", True),
        include_text=True,
        min_modal_coverage=getattr(config, "min_modal_coverage", 0.6)
    )
    logging.info("✅ 强配对批采样器创建成功")
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=getattr(config, "num_workers", 2),  # guide6.md: 适中的工作进程数
        pin_memory=getattr(config, "pin_memory", True),  # 开启内存锁定配合non_blocking加速传输
        collate_fn=compatible_collate_fn,
        persistent_workers=getattr(config, "persistent_workers", True),  # guide6.md: 保持工作进程，避免重复创建
        prefetch_factor=getattr(config, "prefetch_factor", 2)  # guide6.md: 预取因子，平衡内存和性能
    )

    # 简化验证配置，减少验证开销
    gallery_loader, query_loaders = build_eval_loaders_by_rule(
        local_val_dataset, list(range(len(local_val_dataset))),
        batch_size=min(4, safe_batch_size),  # 进一步减少验证batch_size
        num_workers=0,  # 验证时不用多进程
        pin_memory=False  # 简化配置
    )

    # 模型：CLIP+MER架构
    model = CLIPBasedMultiModalReIDModel(config).to(device)
    
    # 设置分类器（动态设置ID类别数）
    num_classes = getattr(config, 'num_classes', None)
    if num_classes is not None:
        model.set_num_classes(num_classes)
        logging.info(f"设置分类器：{num_classes} 个ID类别")
    else:
        logging.warning("config中未找到num_classes，请确保在数据加载后设置")

    # 显存优化：冻结主干，只训练 LoRA 和特定模块
    freeze_backbone = getattr(config, 'freeze_backbone', True)
    if freeze_backbone:
        logging.info("冻结 CLIP 主干，只训练 LoRA 和特定模块")
        for name, param in model.named_parameters():
            if 'loras' in name or 'feature_mixture' in name or 'bn_neck' in name or 'null_tokens' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    # 优化器 - 支持CLIP+MER分层学习率
    param_groups = model.get_learnable_params()
    
    # guide6.md: 分类头LR降档，防权重爆涨
    head_lr = getattr(config, 'head_learning_rate', 3e-3)
    head_lr_warmup_epochs = getattr(config, 'head_lr_warmup_epochs', 2)
    
    # 过滤掉冻结的参数，并调整分类头学习率
    filtered_param_groups = []
    for group in param_groups:
        trainable_params = [p for p in group['params'] if p.requires_grad]
        if trainable_params:  # 只有包含可训练参数的组才添加
            group['params'] = trainable_params
            
            # guide6.md: 从Epoch 2起把head LR调到3e-3（在训练循环中动态调整）
            if 'classifier' in group.get('name', ''):
                logging.info(f"分类头初始学习率: {group['lr']:.2e}，将在Epoch {head_lr_warmup_epochs}后降档到 {head_lr:.2e}")
            
            filtered_param_groups.append(group)
    
    # 日志输出各参数组的学习率
    total_trainable = 0
    for group in filtered_param_groups:
        group_name = group.get('name', 'unknown')
        group_lr = group['lr']
        num_params = len(group['params'])
        total_trainable += sum(p.numel() for p in group['params'])
        logging.info(f"{group_name}: {num_params} 参数, 学习率: {group_lr:.2e}")
    
    logging.info(f"可训练参数总数: {total_trainable:,}")
    
    optimizer = AdamW(filtered_param_groups, weight_decay=getattr(config, "weight_decay", 1e-4))
    
    # 快速测试：验证模型能正常前向传播
    logging.info("执行模型前向传播测试...")
    try:
        test_batch = next(iter(train_loader))
        test_batch = move_batch_to_device(test_batch, device)
        with torch.no_grad():
            test_outputs = call_model_with_batch(model, test_batch)
            logging.info(f"✅ 前向传播测试成功，输出keys: {list(test_outputs.keys())}")
            if 'logits' in test_outputs:
                logging.info(f"   logits shape: {test_outputs['logits'].shape}")
            if 'bn_features' in test_outputs:
                logging.info(f"   bn_features shape: {test_outputs['bn_features'].shape}")
    except Exception as e:
        logging.error(f"❌ 前向传播测试失败: {e}")
        raise e

    # AMP 优化：使用 bfloat16 + 梯度累积
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = (autocast_dtype == torch.float16)  # bfloat16 不需要 GradScaler
    scaler = GradScaler(enabled=use_scaler) if use_scaler else None
    
    # 调整梯度累积以补偿小batch_size
    target_effective_batch = 16  # 目标有效batch大小
    accum_steps = max(1, target_effective_batch // safe_batch_size)  # 动态调整累积步数
    effective_batch_size = safe_batch_size * accum_steps
    logging.info(f"梯度累积调整: {safe_batch_size} × {accum_steps} = {effective_batch_size}")
    
    logging.info(f"混合精度: {autocast_dtype}, 梯度累积: {accum_steps} 步")
    logging.info(f"实际 batch_size: {actual_batch_size}, 等效 batch_size: {effective_batch_size}")

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
        base_lr = getattr(config, 'base_learning_rate', 1e-5)
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
    # guide11.md: 评测触发条件（每个epoch都评）
    eval_start_epoch = getattr(config, 'eval_start_epoch', 1)
    eval_every_n_epoch = getattr(config, 'eval_every_n_epoch', 1)
    eval_freq = eval_every_n_epoch  # 每个epoch都评测
    save_dir = getattr(config, "save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # 改进方案：自适应进度条 + 动态batch数学习
    estimated_batches = len(train_loader) if hasattr(train_loader, '__len__') else 1000
    actual_batches_per_epoch = []  # 记录每轮的真实batch数
    
    logging.info(f"=== 训练配置总结 ===")
    logging.info(f"计划训练轮次: {num_epochs}")
    logging.info(f"采样器估算batch数: {estimated_batches}")
    logging.info(f"注意: 第1轮为自适应学习，第2轮起进度条将基于实际数据")
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # 每个epoch开始前清除文本缓存（仅在微调文本时清除，冻结时保持缓存）
        if hasattr(model, "text_cache") and not getattr(model, "freeze_text", True):
            model.text_cache.clear()
            if epoch == 1:  # 第一个epoch提醒
                logging.info("文本编码器微调模式：每epoch清除缓存")

        # 自适应进度条：第1轮学习，第2轮起使用真实数据
        if epoch == 1:
            # 第1轮：使用估算值 + 动态扩展
            progress_total = estimated_batches
            progress_desc = f"Epoch {epoch}/{num_epochs} [Learning]"
        elif len(actual_batches_per_epoch) > 0:
            # 第2轮起：使用前一轮的真实batch数
            progress_total = actual_batches_per_epoch[-1]  # 使用最近一轮的实际数据
            progress_desc = f"Epoch {epoch}/{num_epochs} [Actual: {progress_total}]"
        else:
            # 备用方案
            progress_total = estimated_batches
            progress_desc = f"Epoch {epoch}/{num_epochs} [Est: {progress_total}]"
        
        with tqdm(total=progress_total, 
                  desc=progress_desc, 
                  leave=False, ncols=130,
                  dynamic_ncols=True) as epoch_pbar:
            
            adaptive_clip = getattr(config, "adaptive_gradient_clip", True)
            train_metrics = train_epoch_fixed(
                model, train_loader, optimizer, device, epoch, 
                scaler, adaptive_clip, accum_steps, autocast_dtype, config, 
                epoch_pbar  # 传递进度条
            )
        
        # 记录本轮的真实batch数，用于下轮进度条精确显示
        actual_batches_this_epoch = train_metrics.get('actual_batches', estimated_batches)
        actual_batches_per_epoch.append(actual_batches_this_epoch)
        
        # 统计监控：估算vs实际的差异
        if epoch == 1:
            batch_accuracy = actual_batches_this_epoch / estimated_batches
            logging.info(f"第1轮batch数学习完成: 估算={estimated_batches}, 实际={actual_batches_this_epoch}, 准确率={batch_accuracy:.1%}")
            if abs(batch_accuracy - 1.0) > 0.2:  # 如果误差超过20%
                logging.warning(f"⚠️  采样器估算偏差较大({batch_accuracy:.1%})，建议检查数据分布")
        else:
            expected = actual_batches_per_epoch[-2] if len(actual_batches_per_epoch) > 1 else estimated_batches
            if abs(actual_batches_this_epoch - expected) > max(10, expected * 0.1):
                logging.warning(f"⚠️  Epoch {epoch} batch数波动: 预期≈{expected}, 实际={actual_batches_this_epoch}")
        
        # 动态调整进度条到正确的完成状态
        if epoch_pbar.total != actual_batches_this_epoch:
            epoch_pbar.total = actual_batches_this_epoch
            epoch_pbar.n = actual_batches_this_epoch
            epoch_pbar.refresh()
        
        # guide6.md: 分类头学习率动态调整
        head_lr = getattr(config, 'head_learning_rate', 3e-3)
        head_lr_warmup_epochs = getattr(config, 'head_lr_warmup_epochs', 2)
        
        if epoch >= head_lr_warmup_epochs:
            # 从Epoch 2起把head LR调到3e-3
            for param_group in optimizer.param_groups:
                if 'classifier' in param_group.get('name', ''):
                    if param_group['lr'] != head_lr:
                        old_lr = param_group['lr']
                        param_group['lr'] = head_lr
                        logging.info(f"Epoch {epoch}: 分类头学习率降档 {old_lr:.2e} -> {head_lr:.2e}")
        
        # === SDM调度器更新（基于训练指标） ===
        if hasattr(model, 'sdm_scheduler'):
            # 更新SDM参数
            effective_cont_w, current_temp = model.sdm_scheduler.get_parameters(epoch, train_metrics, comp_metrics if 'comp_metrics' in locals() else None)
            
            # 检查是否可以增加权重
            if model.sdm_scheduler.can_increase_weight(epoch, train_metrics, comp_metrics if 'comp_metrics' in locals() else None):
                if model.sdm_scheduler.increase_weight():
                    effective_cont_w = model.sdm_scheduler.weight_scheduler.current_weight
            
            # 检查是否需要降低权重（异常情况）
            current_sdm_loss = train_metrics.get('sdm_loss', train_metrics.get('contrastive_loss', 0.0))
            if current_sdm_loss > 5.0 or current_sdm_loss < 0:
                model.sdm_scheduler.decrease_weight(f"SDM损失异常: {current_sdm_loss:.4f}")
                effective_cont_w = model.sdm_scheduler.weight_scheduler.current_weight
        
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


        # guide15.md: 确保评测只在epoch结束时触发，不在训练循环内
        should_eval = (
            epoch >= eval_start_epoch and 
            ((epoch - eval_start_epoch) % eval_every_n_epoch == 0) and
            getattr(config, "do_eval", True) and
            getattr(config, "eval_every_n_steps", 0) == 0  # 确保没有步数级评测
        )
        
        if should_eval:
            print(f"[INFO] 开始第{epoch}轮评测（仅在epoch结束时触发）")
            sample_ratio = getattr(config, "eval_sample_ratio", 0.3)
            comp_metrics = validate_competition_style(model, gallery_loader, query_loaders, device, k_map=100, sample_ratio=sample_ratio, cfg=config, epoch=epoch)

            # guide14.md: 评测结果打印已在validate_competition_style中处理
            # 这里可以添加额外的日志记录或保存best model逻辑

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

        # 改进的调度器步进 - 基于实际step数而非估算
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                if epoch >= eval_start_epoch and ((epoch - eval_start_epoch) % eval_every_n_epoch == 0):
                    current_map = comp_metrics['map_avg2'] if 'comp_metrics' in locals() else 0.0
                    scheduler.step(current_map)
            elif isinstance(scheduler, LambdaLR) and len(actual_batches_per_epoch) > 0:
                # 对于基于epoch的调度器，使用实际batch数信息
                # 但仍然按epoch步进，只是在日志中记录实际的step信息
                scheduler.step()
                
                # 额外信息：计算实际训练的总step数
                total_actual_steps = sum(actual_batches_per_epoch)
                avg_steps_per_epoch = sum(actual_batches_per_epoch) / len(actual_batches_per_epoch)
                logging.info(f"Epoch {epoch}: 累计实际steps={total_actual_steps}, 本轮steps={actual_batches_this_epoch}, 平均={avg_steps_per_epoch:.1f}")
            else:
                scheduler.step()

            if epoch % 20 == 0:  # 减少查询频率：10->20
                lrs = [pg['lr'] for pg in optimizer.param_groups]
                logging.info(f"Epoch {epoch}: 当前学习率 = {', '.join([f'{lr:.2e}' for lr in lrs])}")

    # 训练完成后全量评估
    logging.info("训练完成，开始本地划分验证集的完整评估...")
    final_metrics = validate_competition_style(model, gallery_loader, query_loaders, device, k_map=100, sample_ratio=1.0, cfg=config, epoch=config.num_epochs)
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
        'val_ids': val_ids
    }
    with open(os.path.join(save_dir, 'dataset_split.pkl'), 'wb') as f:
        pickle.dump(split_info, f)

    # 训练完成统计
    if actual_batches_per_epoch:
        avg_batches = sum(actual_batches_per_epoch) / len(actual_batches_per_epoch)
        min_batches = min(actual_batches_per_epoch)
        max_batches = max(actual_batches_per_epoch)
        logging.info(f"=== 训练完成统计 ===")
        logging.info(f"采样器初始估算: {estimated_batches} batch/epoch")
        logging.info(f"实际batch数统计: 均值={avg_batches:.1f}, 范围=[{min_batches}, {max_batches}]")
        logging.info(f"估算准确率: {avg_batches/estimated_batches:.1%} (理想值: 90%-110%)")
        
        # 分析采样稳定性
        if len(actual_batches_per_epoch) > 1:
            batch_variance = sum((b - avg_batches)**2 for b in actual_batches_per_epoch) / len(actual_batches_per_epoch)
            batch_cv = (batch_variance**0.5) / avg_batches if avg_batches > 0 else 0
            logging.info(f"批次稳定性: 变异系数={batch_cv:.3f} ({'稳定' if batch_cv < 0.1 else '波动较大'})")
    
    logging.info(f"最佳(四类平均) mAP: {best_map:.4f}")
    logging.info("=====================")

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
