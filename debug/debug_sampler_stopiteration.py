#!/usr/bin/env python3
# debug_sampler_stopiteration.py
# 诊断MultiModalBalancedSampler的StopIteration问题

import torch
from datasets.dataset import MultiModalDataset, MultiModalBalancedSampler, infer_modalities_of_sample
from configs.config import TrainingConfig
from tools.split import split_ids, create_split_datasets
from collections import defaultdict

def diagnose_sampler_stopiteration():
    """诊断采样器StopIteration问题"""
    
    print("=== MultiModalBalancedSampler 诊断 ===")
    
    # 加载配置和数据集
    config = TrainingConfig()
    full_dataset = MultiModalDataset(config, split='train')
    
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
    
    # 创建训练集
    train_dataset, _ = create_split_datasets(
        full_dataset, train_ids, val_ids, config
    )
    
    print(f"数据集信息:")
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  训练集ID数: {len(train_ids)}")
    
    # 分析训练集中每个ID的模态分布
    print(f"\n=== 分析训练集模态分布 ===")
    
    id_modality_stats = defaultdict(lambda: {'vis': 0, 'nir': 0, 'sk': 0, 'cp': 0, 'text': 0, 'total': 0})
    
    for idx in range(len(train_dataset)):
        sample = train_dataset.data_list[idx]
        pid = sample['person_id']
        modalities = infer_modalities_of_sample(train_dataset, idx)
        
        id_modality_stats[pid]['total'] += 1
        for mod in modalities:
            if mod in id_modality_stats[pid]:
                id_modality_stats[pid][mod] += 1
    
    # 统计可配对的ID
    pairable_ids = []
    vis_only_ids = []
    non_vis_only_ids = []
    empty_ids = []
    
    for pid in train_ids:  # 只检查训练集的ID
        if pid in id_modality_stats:
            stats = id_modality_stats[pid]
            has_vis = stats['vis'] > 0
            has_non_vis = any(stats[m] > 0 for m in ['nir', 'sk', 'cp', 'text'])
            
            if has_vis and has_non_vis:
                pairable_ids.append(pid)
            elif has_vis:
                vis_only_ids.append(pid)
            elif has_non_vis:
                non_vis_only_ids.append(pid)
            else:
                empty_ids.append(pid)
        else:
            empty_ids.append(pid)
    
    print(f"训练集模态统计:")
    print(f"  可配对ID数 (vis+非vis): {len(pairable_ids)}")
    print(f"  仅vis ID数: {len(vis_only_ids)}")
    print(f"  仅非vis ID数: {len(non_vis_only_ids)}")
    print(f"  无模态ID数: {len(empty_ids)}")
    print(f"  可配对率: {len(pairable_ids)/len(train_ids):.1%}")
    
    # 显示前几个ID的详细信息
    print(f"\n=== 前10个ID的模态详情 ===")
    for i, pid in enumerate(train_ids[:10]):
        if pid in id_modality_stats:
            stats = id_modality_stats[pid]
            has_vis = stats['vis'] > 0
            has_non_vis = any(stats[m] > 0 for m in ['nir', 'sk', 'cp', 'text'])
            status = "可配对" if (has_vis and has_non_vis) else "不可配对"
            print(f"  ID {pid}: 总{stats['total']}, vis={stats['vis']}, "
                  f"nir={stats['nir']}, sk={stats['sk']}, cp={stats['cp']}, "
                  f"text={stats['text']} - {status}")
        else:
            print(f"  ID {pid}: 无数据")
    
    # 现在测试采样器
    print(f"\n=== 测试MultiModalBalancedSampler ===")
    
    batch_size = 32
    num_instances = 4
    
    if len(pairable_ids) == 0:
        print("❌ 没有可配对的ID，采样器无法工作！")
        print("\n🔧 建议修复方案:")
        print("1. 检查数据集中的模态标注是否正确")
        print("2. 使用ModalAwarePKSampler作为备选")
        print("3. 调整modality inference逻辑")
        return False
    
    num_pids_per_batch = batch_size // num_instances
    if len(pairable_ids) < num_pids_per_batch:
        print(f"❌ 可配对ID数({len(pairable_ids)}) < 每批需要的ID数({num_pids_per_batch})")
        print("采样器无法生成完整的batch")
        print(f"\n🔧 建议修复方案:")
        print(f"1. 减少batch_size到 {len(pairable_ids) * num_instances}")
        print(f"2. 减少num_instances")
        print(f"3. 使用更宽松的采样策略")
        return False
    
    try:
        sampler = MultiModalBalancedSampler(
            train_dataset, 
            batch_size, 
            num_instances=num_instances
        )
        
        print(f"采样器创建成功:")
        print(f"  valid_pids数量: {len(sampler.valid_pids)}")
        print(f"  预期batch数量: {len(sampler)}")
        
        # 尝试生成第一个batch
        batch_iter = iter(sampler)
        first_batch = next(batch_iter)
        print(f"  首批生成成功，大小: {len(first_batch)}")
        print("✅ MultiModalBalancedSampler工作正常")
        return True
        
    except StopIteration:
        print("❌ 采样器立即抛出StopIteration")
        return False
    except Exception as e:
        print(f"❌ 采样器创建/使用失败: {e}")
        return False

if __name__ == "__main__":
    success = diagnose_sampler_stopiteration()
    if not success:
        print("\n" + "="*50)
        print("🔧 立即修复建议：在train.py中使用更稳健的采样器切换逻辑")