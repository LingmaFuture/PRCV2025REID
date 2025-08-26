#!/usr/bin/env python3
"""
测试按模态批量前向传播修复
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets.dataset import modal_batched_collate_fn
from models.model import CLIPBasedMultiModalReIDModel
from configs.config import TrainingConfig
from train import batched_modal_forward

def test_batched_forward():
    """测试修复后的按模态批量前向传播"""
    print("测试按模态批量前向传播修复...")
    
    # 配置
    config = TrainingConfig()
    config.num_classes = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    try:
        model = CLIPBasedMultiModalReIDModel(config).to(device)
        model.set_num_classes(10)
        print("√ 模型创建成功")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False
    
    # 创建测试数据（模拟真实的不完整模态情况）
    test_batch = {
        'person_id': torch.randint(0, 10, (4,)),
        'batch_size': 4,
        'modal_buckets': {
            'vis': {
                'data': torch.randn(2, 3, 224, 224),  # 只有2个样本的vis数据
                'indices': [0, 2],  # 样本0和2有vis数据
                'batch_size': 2
            },
            'text': {
                'data': ['person walking'],  # 只有1个样本的文本数据
                'indices': [1],  # 只有样本1有text数据
                'batch_size': 1
            }
            # 样本3没有任何模态数据，将使用零特征
        }
    }
    
    # 测试前向传播
    try:
        model.eval()
        # 不使用AMP上下文，避免dtype混乱
        with torch.no_grad():
            outputs = batched_modal_forward(model, test_batch, device, return_features=True)
        
        print(f"√ 前向传播成功")
        print(f"  - features shape: {outputs['features'].shape}")
        print(f"  - 输出keys: {list(outputs.keys())}")
        
        # 验证维度一致性
        expected_batch_size = test_batch['batch_size']
        actual_batch_size = outputs['features'].shape[0]
        
        if actual_batch_size == expected_batch_size:
            print(f"  ✓ 批次大小一致: {actual_batch_size}")
        else:
            print(f"  ✗ 批次大小不一致: 期望{expected_batch_size}, 实际{actual_batch_size}")
            return False
        
        if 'logits' in outputs:
            print(f"  - logits shape: {outputs['logits'].shape}")
            if outputs['logits'].shape[0] == expected_batch_size:
                print(f"  ✓ logits批次大小一致")
            else:
                print(f"  ✗ logits批次大小不一致")
                return False
        
        return True
        
    except Exception as e:
        print(f"X 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batched_forward()
    if success:
        print("\n! 按模态批量前向传播修复成功！")
    else:
        print("\nX 仍有问题需要修复")
