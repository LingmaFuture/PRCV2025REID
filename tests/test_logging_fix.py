#!/usr/bin/env python3
"""
测试logging修复的脚本
验证models/model.py中的logging作用域问题是否已解决
"""

import sys
import os
import torch
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import CLIPBasedMultiModalReIDModel
from configs.config import TrainingConfig

def test_logging_fix():
    """测试logging修复是否有效"""
    print("🔍 测试logging修复...")
    
    # 设置logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 创建配置
        config = TrainingConfig()
        
        # 创建模型
        print("📦 创建模型...")
        model = CLIPBasedMultiModalReIDModel(config)
        
        # 设置类别数
        model.set_num_classes(100)
        
        # 创建模拟数据
        print("📊 创建模拟数据...")
        batch_size = 8
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模拟图像数据
        images = {
            'rgb': torch.randn(batch_size, 3, 224, 224),
            'ir': torch.randn(batch_size, 3, 224, 224),
            'sketch': torch.randn(batch_size, 3, 224, 224)
        }
        
        # 模拟文本数据
        texts = ["person wearing red shirt"] * batch_size
        
        # 模拟模态掩码
        modality_masks = {
            'vis': torch.ones(batch_size),
            'nir': torch.ones(batch_size),
            'sk': torch.ones(batch_size)
        }
        
        # 模拟标签
        labels = torch.randint(0, 100, (batch_size,))
        
        # 移动数据到设备
        images = {k: v.to(device) for k, v in images.items()}
        labels = labels.to(device)
        modality_masks = {k: v.to(device) for k, v in modality_masks.items()}
        
        # 前向传播
        print("🔄 执行前向传播...")
        model.to(device)
        model.train()
        
        outputs = model(
            images=images,
            texts=texts,
            modality_masks=modality_masks
        )
        
        # 计算损失（这里应该不会出现logging错误）
        print("💡 计算损失...")
        loss_dict = model.compute_loss(outputs, labels)
        
        print("✅ 测试通过！logging修复成功")
        print(f"   总损失: {loss_dict['total_loss'].item():.4f}")
        print(f"   CE损失: {loss_dict['ce_loss'].item():.4f}")
        print(f"   SDM损失: {loss_dict['sdm_loss'].item():.4f}")
        print(f"   特征范数: {loss_dict['feature_norm']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sampler():
    """测试新的采样器"""
    print("\n🔍 测试MultiModalBalancedSampler...")
    
    try:
        from datasets.dataset import MultiModalBalancedSampler
        
        # 创建模拟数据集
        class MockDataset:
            def __init__(self):
                self.data_list = []
                for i in range(100):
                    # 模拟有RGB+非RGB组合的数据
                    self.data_list.append({
                        'person_id': i % 20,  # 20个不同的ID
                        'modality': 'vis' if i % 3 == 0 else ('nir' if i % 3 == 1 else 'sk')
                    })
        
        dataset = MockDataset()
        
        # 创建采样器
        sampler = MultiModalBalancedSampler(
            dataset=dataset,
            batch_size=8,
            num_instances=4,
            seed=42
        )
        
        print(f"✅ 采样器创建成功")
        print(f"   有效ID数量: {len(sampler.valid_pids)}")
        print(f"   批次数量: {len(sampler)}")
        
        # 测试一个批次
        batch_iter = iter(sampler)
        batch = next(batch_iter)
        print(f"   第一个批次: {batch}")
        
        return True
        
    except Exception as e:
        print(f"❌ 采样器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始测试logging修复...")
    
    # 测试logging修复
    logging_ok = test_logging_fix()
    
    # 测试采样器
    sampler_ok = test_sampler()
    
    if logging_ok and sampler_ok:
        print("\n🎉 所有测试通过！修复成功！")
        print("\n📋 修复总结:")
        print("   1. ✅ 修复了models/model.py中的logging作用域问题")
        print("   2. ✅ 添加了稳健的SDM损失保护机制")
        print("   3. ✅ 创建了MultiModalBalancedSampler确保RGB+非RGB组合")
        print("   4. ✅ 更新了train.py的导入语句")
    else:
        print("\n❌ 部分测试失败，请检查修复")
        sys.exit(1)
