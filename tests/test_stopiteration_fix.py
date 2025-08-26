#!/usr/bin/env python3
# test_stopiteration_fix.py
# 快速验证StopIteration修复效果

import torch
import sys
from datasets.dataset import MultiModalDataset, ModalAwarePKSampler, compatible_collate_fn
from torch.utils.data import DataLoader
from configs.config import TrainingConfig
from tools.split import split_ids, create_split_datasets

def test_fix():
    """测试修复后的采样器是否工作"""
    
    print("🧪 测试StopIteration修复效果")
    print("=" * 40)
    
    try:
        # 加载配置和数据
        config = TrainingConfig()
        full_dataset = MultiModalDataset(config, split='train')
        
        # 数据集划分（模拟train.py的逻辑）
        all_person_ids = [full_dataset.data_list[i]['person_id'] for i in range(len(full_dataset))]
        all_person_ids = sorted(list(set(all_person_ids)))
        train_ids, val_ids = split_ids(all_person_ids, val_ratio=0.2, seed=42)
        train_dataset, _ = create_split_datasets(full_dataset, train_ids, val_ids, config)
        
        print(f"数据集信息:")
        print(f"  训练集: {len(train_dataset)} 样本, {len(train_ids)} ID")
        
        # 测试参数
        actual_batch_size = 32
        num_instances = 4
        
        print(f"批次参数: batch_size={actual_batch_size}, num_instances={num_instances}")
        
        # 参数校验
        assert actual_batch_size % num_instances == 0, \
            f"批次大小必须能被实例数整除: {actual_batch_size} % {num_instances} != 0"
        P = actual_batch_size // num_instances
        print(f"P×K结构: {P}×{num_instances} = {actual_batch_size}")
        
        # 创建采样器（按修复后的逻辑）
        print(f"\n🔧 创建ModalAwarePKSampler...")
        train_sampler = ModalAwarePKSampler(
            dataset=train_dataset,               
            batch_size=actual_batch_size,        
            num_instances=num_instances,         
            ensure_rgb=True,                     
            prefer_complete=True,                
            seed=42,
        )
        print(f"✅ 采样器创建成功")
        
        # 创建DataLoader
        print(f"\n🔧 创建DataLoader...")
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,         
            num_workers=0,  # 测试时设为0避免多进程问题
            pin_memory=False,
            collate_fn=compatible_collate_fn     
        )
        print(f"✅ DataLoader创建成功")
        
        # 测试生成batch
        print(f"\n🧪 测试batch生成...")
        batch_count = 0
        pairable_stats = []
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # 只测试前3个batch
                break
                
            labels = batch['person_id']
            modality_mask = batch.get('modality_mask', {})
            
            # 统计可配对性
            unique_ids = torch.unique(labels)
            pairable_count = 0
            
            for pid in unique_ids:
                pid_indices = (labels == pid)
                has_vis = modality_mask.get('vis', torch.zeros_like(labels))[pid_indices].any()
                has_non_vis = any(modality_mask.get(m, torch.zeros_like(labels))[pid_indices].any() 
                                 for m in ['nir', 'sk', 'cp', 'text'])
                if has_vis and has_non_vis:
                    pairable_count += 1
            
            pairable_ratio = pairable_count / len(unique_ids) if len(unique_ids) > 0 else 0
            pairable_stats.append(pairable_ratio)
            
            print(f"  Batch {batch_idx}: {len(labels)}样本, {len(unique_ids)}ID, "
                  f"可配对: {pairable_count}/{len(unique_ids)} ({pairable_ratio:.1%})")
            
            batch_count += 1
        
        avg_pairable_ratio = sum(pairable_stats) / len(pairable_stats) if pairable_stats else 0
        
        print(f"\n📊 测试结果:")
        print(f"  成功生成batch数: {batch_count}")
        print(f"  平均可配对率: {avg_pairable_ratio:.1%}")
        
        # 判断修复效果
        if batch_count > 0:
            print(f"✅ StopIteration问题已修复！")
            if avg_pairable_ratio >= 0.7:
                print(f"✅ 可配对率良好，SDM损失应该正常工作")
                return True
            else:
                print(f"⚠️ 可配对率偏低，可能仍有部分SDM损失为0")
                return True
        else:
            print(f"❌ 仍然无法生成batch")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fix()
    if success:
        print(f"\n🚀 可以运行 python train.py 开始训练了!")
    else:
        print(f"\n🔍 需要进一步诊断问题")
        sys.exit(1)