#!/usr/bin/env python3
"""
测试模态识别修复效果
验证数据集是否能正确识别各模态
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from collections import Counter, defaultdict
from datasets.dataset import MultiModalDataset
from configs.config import Config

def test_modality_identification():
    """测试模态识别功能"""
    print("=" * 60)
    print("测试模态识别修复效果")
    print("=" * 60)
    
    # 加载配置
    config = Config()
    
    # 创建数据集
    print("正在加载训练数据集...")
    train_dataset = MultiModalDataset(config, split='train')
    print(f"数据集加载完成: {len(train_dataset)} 个样本")
    
    # 统计模态分布
    modality_counter = Counter()
    pid_modality_stats = defaultdict(lambda: defaultdict(int))
    
    print("\n开始分析样本模态分布...")
    for i in range(min(1000, len(train_dataset))):  # 只分析前1000个样本
        try:
            sample = train_dataset[i]
            modality = sample.get('modality', 'unknown')
            modality_counter[modality] += 1
            
            # 统计每个ID的模态分布
            person_id = sample['person_id'].item()
            pid_modality_stats[person_id][modality] += 1
            
            if i < 10:  # 打印前10个样本的详细信息
                print(f"样本 {i}: ID={person_id}, 模态={modality}, "
                      f"模态掩码={dict(sample.get('modality_mask', {}))}")
                
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            continue
    
    # 输出统计结果
    print(f"\n模态分布统计 (前{min(1000, len(train_dataset))}个样本):")
    for modality, count in modality_counter.most_common():
        percentage = count / min(1000, len(train_dataset)) * 100
        print(f"  {modality}: {count} ({percentage:.1f}%)")
    
    # 分析可配对ID
    print(f"\n分析可配对ID (有vis+非vis模态组合):")
    pairable_ids = []
    for pid, mod_stats in pid_modality_stats.items():
        has_vis = mod_stats.get('vis', 0) > 0
        has_nonvis = any(mod_stats.get(m, 0) > 0 for m in ['nir', 'sk', 'cp', 'text'])
        if has_vis and has_nonvis:
            pairable_ids.append(pid)
    
    print(f"  可配对ID数: {len(pairable_ids)} / {len(pid_modality_stats)} ({len(pairable_ids)/len(pid_modality_stats)*100:.1f}%)")
    
    if pairable_ids:
        print(f"  前5个可配对ID: {pairable_ids[:5]}")
        # 显示一个可配对ID的详细模态分布
        example_pid = pairable_ids[0]
        print(f"  示例ID {example_pid} 的模态分布: {dict(pid_modality_stats[example_pid])}")
    
    # 检查是否有问题
    if modality_counter['unknown'] > 0:
        print(f"\n⚠️  警告: 发现 {modality_counter['unknown']} 个样本被标记为 'unknown' 模态")
    
    if len(pairable_ids) == 0:
        print(f"\n❌ 错误: 没有找到可配对的ID，采样器将无法正常工作")
    else:
        print(f"\n✅ 修复成功: 找到 {len(pairable_ids)} 个可配对ID，采样器应该能正常工作")
    
    return len(pairable_ids) > 0

if __name__ == "__main__":
    success = test_modality_identification()
    if success:
        print("\n🎉 模态识别修复测试通过！")
    else:
        print("\n❌ 模态识别修复测试失败，需要进一步调试")
