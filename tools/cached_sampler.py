# datasets/cached_sampler.py
# 缓存优化版的ModalAwarePKSampler - 一次性预计算所有meta信息

from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Sampler
import logging

logger = logging.getLogger(__name__)

RGB_TAGS = {'rgb', 'vis', 'VIS'}

def _fast_meta(ds, i):
    """
    返回 (pid, modality)，尽量避免解码图像。
    适配多种常见数据集写法，最终兜底为 ds[i]（可能会触发一次真实加载）。
    """
    # 1) 常见：ds.data_list 直接访问元数据
    if hasattr(ds, 'data_list') and len(ds.data_list) > i:
        rec = ds.data_list[i]
        if isinstance(rec, dict) and 'person_id' in rec:
            pid = rec['person_id']
            # 推断模态 - 优先从modality字段，然后从文件路径
            if 'modality' in rec:
                mod = rec['modality']
            else:
                # 从infer_modalities_of_sample推断
                from datasets.dataset import infer_modalities_of_sample
                mods = infer_modalities_of_sample(ds, i)
                mod = 'rgb' if 'vis' in mods else (list(mods)[0] if mods else 'unknown')
            return pid, mod
    
    # 2) 常见：ds.meta / ds.annotations / ds.items 里有轻量字段
    for attr in ('meta', 'metadata', 'annotations', 'items', 'index', 'records'):
        if hasattr(ds, attr):
            try:
                rec = getattr(ds, attr)[i]
                if isinstance(rec, dict) and ('person_id' in rec and 'modality' in rec):
                    return rec['person_id'], rec['modality']
            except (IndexError, KeyError, AttributeError):
                continue
                
    # 3) 某些自定义 getter
    for attr in ('get_meta', 'peek', 'meta_at'):
        if hasattr(ds, attr):
            try:
                rec = getattr(ds, attr)(i)
                if isinstance(rec, dict) and ('person_id' in rec and 'modality' in rec):
                    return rec['person_id'], rec['modality']
            except (IndexError, KeyError, AttributeError):
                continue
                
    # 4) 兜底：调用 __getitem__（可能会慢，但只发生在初始化这一遍）
    try:
        rec = ds[i]
        if isinstance(rec, dict):
            if 'person_id' in rec and 'modality' in rec:
                return rec['person_id'], rec['modality']
            elif 'person_id' in rec:
                # 没有modality字段，推断一下
                from datasets.dataset import infer_modalities_of_sample
                mods = infer_modalities_of_sample(ds, i)
                mod = 'rgb' if 'vis' in mods else (list(mods)[0] if mods else 'unknown')
                return rec['person_id'], mod
                
        # 再兜底：常见 tuple 结构
        # e.g. (image_tensor, pid, modality, ...)
        if isinstance(rec, (list, tuple)) and len(rec) >= 3:
            return rec[1], rec[2]
    except Exception as e:
        logger.warning(f"Failed to get meta from ds[{i}]: {e}")
        
    raise RuntimeError(f"无法从数据集中提取 pid/modality: idx={i}")

class CachedModalAwarePKSampler(Sampler):
    """
    缓存优化版的ModalAwarePKSampler
    
    核心优化：在初始化时一次性缓存所有样本的(pid, modality)信息，
    后续采样只做字典查询和随机抽样，避免反复调用infer_modalities_of_sample
    """
    
    def __init__(self, dataset, batch_size, num_instances=4,
                 ensure_rgb=True, prefer_complete=True, seed=42):
        """
        Args:
            dataset: 数据集对象（可能是Subset）
            batch_size: 批次大小
            num_instances: 每个ID的实例数量（K）
            ensure_rgb: 确保每个ID至少有1张RGB
            prefer_complete: 优先选择有RGB+非RGB组合的ID
            seed: 随机种子
        """
        # 处理数据集（可能是Subset）
        if hasattr(dataset, 'dataset'):
            self.base_dataset = dataset.dataset
            self.indices = dataset.indices
        else:
            self.base_dataset = dataset
            self.indices = list(range(len(dataset)))
            
        self.batch_size = int(batch_size)
        self.K = int(num_instances)
        assert self.batch_size % self.K == 0, f"batch_size({batch_size}) must be divisible by num_instances({num_instances})"
        self.P = self.batch_size // self.K

        self.ensure_rgb = ensure_rgb
        self.prefer_complete = prefer_complete
        self.rng = np.random.default_rng(seed)

        logger.info(f"开始缓存数据集元信息，共{len(self.indices)}个样本...")
        
        # === 核心：一次性缓存 meta，并分桶 ===
        self.pid_to_rgb = defaultdict(list)
        self.pid_to_non = defaultdict(list)
        self.pids_all = set()
        
        cache_start_time = __import__('time').time()
        
        for subset_idx, orig_idx in enumerate(self.indices):
            try:
                pid, mod = _fast_meta(self.base_dataset, orig_idx)
                
                # 确保pid是int类型
                if isinstance(pid, torch.Tensor):
                    pid = int(pid.item())
                else:
                    pid = int(pid)
                    
                self.pids_all.add(pid)
                
                # 根据模态分桶（注意：这里存储的是subset_idx，供DataLoader使用）
                if mod in RGB_TAGS or 'vis' in str(mod).lower():
                    self.pid_to_rgb[pid].append(subset_idx)
                else:
                    self.pid_to_non[pid].append(subset_idx)
                    
            except Exception as e:
                logger.warning(f"跳过样本{orig_idx}: {e}")
                continue
        
        cache_time = __import__('time').time() - cache_start_time
        
        # 可"完备配对"的 pid（RGB+非RGB）
        self.pids_pairable = [p for p in self.pids_all
                              if self.pid_to_rgb[p] and self.pid_to_non[p]]
        
        logger.info(f"✅ 元信息缓存完成 ({cache_time:.2f}s):")
        logger.info(f"  总ID数: {len(self.pids_all)}")
        logger.info(f"  可配对ID数: {len(self.pids_pairable)} ({len(self.pids_pairable)/len(self.pids_all):.1%})")
        logger.info(f"  预期batch数: {len(self)}")
        
        if len(self.pids_pairable) < self.P:
            logger.warning(f"可配对ID数({len(self.pids_pairable)}) < 每批需要ID数({self.P})")
            logger.warning("可能导致某些batch无法生成，建议减少batch_size或num_instances")

    def _pick_K(self, pid):
        """为指定pid选择K个样本，优先保证RGB+非RGB组合"""
        rgbs = self.pid_to_rgb[pid]
        nons = self.pid_to_non[pid]
        
        if self.ensure_rgb and rgbs and nons:
            # 至少1张RGB + 剩余的非RGB
            k1 = min(1, len(rgbs))
            k2 = min(self.K - k1, len(nons))
            
            # 如果还不够K张，从所有样本中补充
            remaining = self.K - k1 - k2
            if remaining > 0:
                all_samples = rgbs + nons
                available = [s for s in all_samples]  # 可以重复选择
            else:
                available = []
            
            # 选择样本
            selected = []
            selected.extend(self.rng.choice(rgbs, size=k1, replace=(len(rgbs) < k1)).tolist())
            selected.extend(self.rng.choice(nons, size=k2, replace=(len(nons) < k2)).tolist())
            
            if remaining > 0 and available:
                selected.extend(self.rng.choice(available, size=remaining, replace=(len(available) < remaining)).tolist())
            
            # 如果还是不够，重复选择已有的
            while len(selected) < self.K:
                if rgbs + nons:
                    selected.append(self.rng.choice(rgbs + nons))
                else:
                    break
                    
            return selected[:self.K]
        else:
            # 兜底：该 pid 模态不全，随机凑 K
            pool = rgbs + nons
            if not pool:
                logger.warning(f"ID {pid} 没有可用样本")
                return []
            return self.rng.choice(pool, size=self.K, replace=(len(pool) < self.K)).tolist()

    def __iter__(self):
        """生成批次索引"""
        pids = list(self.pids_pairable) if self.prefer_complete else list(self.pids_all)
        
        if len(pids) < self.P:
            logger.warning(f"可用ID数({len(pids)}) < 每批需要ID数({self.P})")
            # 补充ID以达到最小要求
            while len(pids) < self.P:
                pids.extend(list(self.pids_all))
        
        self.rng.shuffle(pids)

        batch = []
        for i in range(0, len(pids), self.P):
            batch_pids = pids[i:i+self.P]
            
            if len(batch_pids) < self.P:
                break  # 不足一个完整批次，丢弃
                
            batch_indices = []
            for pid in batch_pids:
                chosen = self._pick_K(pid)
                if chosen:
                    batch_indices.extend(chosen)
                    
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                
    def __len__(self):
        """估计批次数量"""
        available_pids = len(self.pids_pairable) if self.prefer_complete else len(self.pids_all)
        return max(1, available_pids // self.P)