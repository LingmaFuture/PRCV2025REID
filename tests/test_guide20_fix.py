#!/usr/bin/env python3
"""
测试Guide20修复效果：验证多模态样本构建和检索功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.dataset import MultiModalDataset, quick_scan
from configs.config import TrainingConfig

def test_guide20_fix():
    """测试Guide20修复后的多模态样本构建"""
    print("=" * 60)
    print("测试Guide20修复：多模态样本构建和检索")
    print("=" * 60)
    
    try:
        # 加载配置
        config = TrainingConfig()
        
        # 创建数据集（抑制打印）
        dataset = MultiModalDataset(config, split='train')
        dataset._suppress_print = True
        
        print(f"✅ 数据集加载成功，共{len(dataset)}个样本")
        
        # 测试样本结构
        print("\n📊 样本结构验证:")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  样本字段: {list(sample.keys())}")
            if 'images' in sample:
                print(f"  images字段: {list(sample['images'].keys())}")
            if 'modality_mask' in sample:
                print(f"  modality_mask: {sample['modality_mask']}")
            if 'text' in sample or 'text_description' in sample:
                print(f"  文本长度: {len(sample.get('text', sample.get('text_description', [''])[0]))}")
        
        # 测试模态检测功能
        print("\n🔍 模态检测功能测试:")
        if hasattr(dataset, 'infer_modalities_of_sample'):
            sample_mods = dataset.infer_modalities_of_sample(0)
            print(f"  样本0模态: {sample_mods}")
        
        # 快速扫描
        print(f"\n🚀 快速扫描前{min(50, len(dataset))}个样本:")
        quick_scan(dataset, n=min(50, len(dataset)))
        
        print("\n✅ Guide20修复测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_guide20_fix()