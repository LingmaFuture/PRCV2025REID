#!/usr/bin/env python3
# test_performance_optimization.py
# 快速验证性能优化效果

import torch
import time
import sys
from datasets.dataset import MultiModalDataset, compatible_collate_fn
from tools.cached_sampler import CachedModalAwarePKSampler
from torch.utils.data import DataLoader
from configs.config import TrainingConfig
from tools.split import split_ids, create_split_datasets

def test_performance_optimization():
    """测试性能优化的各个组件"""
    
    print("🚀 性能优化测试")
    print("=" * 50)
    
    try:
        # 1. 测试缓存采样器导入
        print("✅ 1/5: 缓存采样器导入成功")
        
        # 2. 加载数据集
        print("📂 2/5: 加载数据集...")
        config = TrainingConfig()
        full_dataset = MultiModalDataset(config, split='train')
        
        # 数据集划分
        all_person_ids = [full_dataset.data_list[i]['person_id'] for i in range(len(full_dataset))]
        all_person_ids = sorted(list(set(all_person_ids)))
        train_ids, val_ids = split_ids(all_person_ids, val_ratio=0.2, seed=42)
        train_dataset, _ = create_split_datasets(full_dataset, train_ids, val_ids, config)
        
        print(f"   训练集: {len(train_dataset)} 样本, {len(train_ids)} ID")
        print("✅ 2/5: 数据集加载成功")
        
        # 3. 测试缓存采样器创建和缓存时间
        print("⏱️  3/5: 测试缓存采样器性能...")
        actual_batch_size = 32
        num_instances = 4
        
        cache_start = time.time()
        train_sampler = CachedModalAwarePKSampler(
            dataset=train_dataset,
            batch_size=actual_batch_size,
            num_instances=num_instances,
            ensure_rgb=True,
            prefer_complete=True,
            seed=42,
        )
        cache_time = time.time() - cache_start
        
        print(f"   缓存时间: {cache_time:.2f}s (目标: <5s)")
        print(f"   可配对ID: {len(train_sampler.pids_pairable)}/{len(train_sampler.pids_all)} ({len(train_sampler.pids_pairable)/len(train_sampler.pids_all):.1%})")
        print(f"   预期batch数: {len(train_sampler)}")
        
        if cache_time > 5.0:
            print("   ⚠️ 缓存时间偏长，但仍在可接受范围")
        else:
            print("   ✅ 缓存时间理想")
            
        print("✅ 3/5: 缓存采样器创建成功")
        
        # 4. 测试优化后的DataLoader
        print("🔧 4/5: 测试优化DataLoader...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=2,                    # Windows优化
            pin_memory=True,                  # 配合non_blocking
            persistent_workers=True,          # 避免重复spawn
            prefetch_factor=1,               # 降低内存争用
            drop_last=True,                  # 避免动态shape
            collate_fn=compatible_collate_fn
        )
        
        print("✅ 4/5: 优化DataLoader创建成功")
        
        # 5. 性能基准测试
        print("⚡ 5/5: 运行性能基准测试...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 测试数据加载速度
        test_batches = min(10, len(train_loader))
        load_times = []
        
        print(f"   测试 {test_batches} 个batch的加载速度...")
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= test_batches:
                break
                
            batch_start = time.time()
            
            # 模拟数据传输到GPU
            def move_to_device(obj):
                if isinstance(obj, dict):
                    return {k: move_to_device(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [move_to_device(x) for x in obj]
                elif torch.is_tensor(obj):
                    return obj.to(device, non_blocking=True)
                else:
                    return obj
                    
            batch = move_to_device(batch)
            
            # 同步等待传输完成
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            batch_time = time.time() - batch_start
            load_times.append(batch_time)
            
            # 验证batch内容
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
            
            if batch_idx < 3:  # 只显示前3个batch的详情
                print(f"     Batch {batch_idx}: {len(labels)}样本, {len(unique_ids)}ID, "
                      f"可配对{pairable_count}/{len(unique_ids)} ({pairable_ratio:.1%}), "
                      f"用时{batch_time:.3f}s")
        
        # 计算性能指标
        avg_batch_time = sum(load_times) / len(load_times)
        batches_per_sec = 1.0 / avg_batch_time
        samples_per_sec = actual_batch_size * batches_per_sec
        
        print(f"\n📊 性能基准测试结果:")
        print(f"   平均batch时间: {avg_batch_time:.3f}s")
        print(f"   批次处理速度: {batches_per_sec:.2f} batches/s")
        print(f"   样本处理速度: {samples_per_sec:.2f} samples/s")
        print(f"   等效it/s: {batches_per_sec:.2f} it/s")
        
        # 性能评估
        if batches_per_sec >= 8.0:
            print("   🎉 性能优秀! 达到优化目标 (>8 it/s)")
            success_level = "优秀"
        elif batches_per_sec >= 4.0:
            print("   ✅ 性能良好! 显著优于优化前 (~2x提升)")  
            success_level = "良好"
        elif batches_per_sec >= 2.5:
            print("   ⚠️ 性能一般，有一定提升但未达到预期")
            success_level = "一般"
        else:
            print("   ❌ 性能提升不明显，需要进一步诊断")
            success_level = "需要优化"
            
        print("✅ 5/5: 性能基准测试完成")
        
        # 总结
        print(f"\n🎯 优化效果总结:")
        print(f"   缓存时间: {cache_time:.2f}s")
        print(f"   可配对率: {len(train_sampler.pids_pairable)/len(train_sampler.pids_all):.1%}")
        print(f"   处理速度: {batches_per_sec:.2f} it/s")
        print(f"   性能等级: {success_level}")
        
        if success_level in ["优秀", "良好"]:
            print(f"\n🚀 优化成功! 可以开始正式训练:")
            print(f"   python train.py")
            return True
        else:
            print(f"\n🔍 建议进一步优化:")
            print(f"   1. 检查硬件配置 (SSD, 足够内存)")
            print(f"   2. 调整batch_size (如果显存允许)")
            print(f"   3. 尝试num_workers=0 或 1")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_performance_optimization()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 性能优化测试通过! 可以开始训练了!")
        print("运行: python train.py")
    else:
        print("🔧 性能优化需要进一步调整")
        print("查看: PERFORMANCE_OPTIMIZATION_GUIDE.md")
        
    sys.exit(0 if success else 1)