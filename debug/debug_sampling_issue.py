#!/usr/bin/env python3
# debug_sampling_issue.py  
# 诊断采样器导致的"无正样本"问题

import torch
import numpy as np
from collections import defaultdict
from train import train_multimodal_reid, MultiModalDataset
from configs.config import TrainingConfig
from datasets.dataset import ModalAwarePKSampler, MultiModalBalancedSampler, compatible_collate_fn
from torch.utils.data import DataLoader

def analyze_sampler_composition(dataset, sampler_class, **sampler_kwargs):
    """分析采样器生成的batch组成"""
    
    print(f"\n=== 分析 {sampler_class.__name__} ===")
    
    # 创建采样器
    sampler = sampler_class(dataset, **sampler_kwargs)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=compatible_collate_fn)
    
    # 分析前5个batch
    batch_stats = []
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 5:  # 只分析前5个batch
            break
            
        labels = batch['person_id']
        modality_mask = batch.get('modality_mask', {})
        
        # 统计ID分布
        unique_ids, id_counts = torch.unique(labels, return_counts=True)
        num_ids = len(unique_ids)
        avg_samples_per_id = float(id_counts.float().mean())
        
        # 统计每个ID的模态分布
        id_modality_stats = defaultdict(lambda: {'vis': 0, 'non_vis': 0, 'total': 0})
        
        for i, pid in enumerate(labels):
            pid = int(pid.item())
            id_modality_stats[pid]['total'] += 1
            
            # 检查vis模态
            if 'vis' in modality_mask:
                vis_mask = modality_mask['vis']
                if isinstance(vis_mask, torch.Tensor) and bool(vis_mask[i]):
                    id_modality_stats[pid]['vis'] += 1
            
            # 检查非vis模态
            has_non_vis = False
            for mod in ['nir', 'sk', 'cp', 'text']:
                if mod in modality_mask:
                    mod_mask = modality_mask[mod]
                    if isinstance(mod_mask, torch.Tensor) and bool(mod_mask[i]):
                        has_non_vis = True
                        break
            
            if has_non_vis:
                id_modality_stats[pid]['non_vis'] += 1
        
        # 统计可配对的ID数量
        pairable_ids = 0
        for pid, stats in id_modality_stats.items():
            if stats['vis'] > 0 and stats['non_vis'] > 0:
                pairable_ids += 1
        
        batch_stat = {
            'batch_idx': batch_idx,
            'batch_size': len(labels),
            'num_ids': num_ids,
            'avg_samples_per_id': avg_samples_per_id,
            'pairable_ids': pairable_ids,
            'pairable_ratio': pairable_ids / num_ids if num_ids > 0 else 0,
            'id_stats': dict(id_modality_stats)
        }
        batch_stats.append(batch_stat)
        
        print(f"Batch {batch_idx}: {batch_stat['batch_size']}样本, {num_ids}ID, "
              f"可配对ID: {pairable_ids}/{num_ids} ({batch_stat['pairable_ratio']:.1%})")
        
        # 详细显示每个ID的模态构成
        for pid, stats in list(id_modality_stats.items())[:3]:  # 只显示前3个ID
            print(f"  ID {pid}: 总{stats['total']}, vis={stats['vis']}, non_vis={stats['non_vis']}")
    
    return batch_stats

def diagnose_sampling_issues():
    """诊断采样问题并提供修复建议"""
    
    print("=== SDM采样诊断工具 ===")
    
    config = TrainingConfig()
    dataset = MultiModalDataset(config, split='train')
    
    print(f"数据集总体信息:")
    print(f"  总样本数: {len(dataset)}")
    
    # 统计数据集中每个ID的模态分布
    from datasets.dataset import infer_modalities_of_sample
    
    id_modality_distribution = defaultdict(lambda: {'vis': 0, 'nir': 0, 'sk': 0, 'cp': 0, 'text': 0, 'total': 0})
    
    for idx in range(len(dataset)):
        sample = dataset.data_list[idx]
        pid = sample['person_id']
        modalities = infer_modalities_of_sample(dataset, idx)
        
        id_modality_distribution[pid]['total'] += 1
        for mod in modalities:
            if mod in id_modality_distribution[pid]:
                id_modality_distribution[pid][mod] += 1
    
    # 统计可配对的ID
    pairable_ids = []
    vis_only_ids = []
    non_vis_only_ids = []
    
    for pid, stats in id_modality_distribution.items():
        has_vis = stats['vis'] > 0
        has_non_vis = any(stats[m] > 0 for m in ['nir', 'sk', 'cp', 'text'])
        
        if has_vis and has_non_vis:
            pairable_ids.append(pid)
        elif has_vis:
            vis_only_ids.append(pid)
        elif has_non_vis:
            non_vis_only_ids.append(pid)
    
    print(f"  可配对ID数: {len(pairable_ids)} ({len(pairable_ids)/len(id_modality_distribution):.1%})")
    print(f"  仅vis ID数: {len(vis_only_ids)}")
    print(f"  仅非vis ID数: {len(non_vis_only_ids)}")
    
    if len(pairable_ids) < 50:
        print("❌ 可配对ID数量过少，将导致频繁的'无正样本'警告")
        return False
    
    # 测试不同采样器的效果
    batch_size = 32
    num_instances = 4
    
    # 测试 ModalAwarePKSampler
    modal_stats = analyze_sampler_composition(
        dataset, 
        ModalAwarePKSampler,
        batch_size=batch_size,
        num_instances=num_instances,
        prefer_complete=True,
        ensure_rgb=True
    )
    
    # 测试 MultiModalBalancedSampler  
    balanced_stats = analyze_sampler_composition(
        dataset,
        MultiModalBalancedSampler,
        batch_size=batch_size,
        num_instances=num_instances
    )
    
    # 比较采样器效果
    print(f"\n=== 采样器对比 ===")
    
    modal_pairable_avg = np.mean([s['pairable_ratio'] for s in modal_stats])
    balanced_pairable_avg = np.mean([s['pairable_ratio'] for s in balanced_stats])
    
    print(f"ModalAwarePKSampler 平均可配对率: {modal_pairable_avg:.1%}")
    print(f"MultiModalBalancedSampler 平均可配对率: {balanced_pairable_avg:.1%}")
    
    if balanced_pairable_avg > 0.8:
        print(f"✅ 推荐使用 MultiModalBalancedSampler")
        return True
    elif modal_pairable_avg > 0.6:
        print(f"⚠️ 推荐使用 ModalAwarePKSampler，但需要调优参数")
        return True
    else:
        print(f"❌ 两种采样器效果都不理想，需要检查数据集")
        return False

if __name__ == "__main__":
    diagnose_sampling_issues()