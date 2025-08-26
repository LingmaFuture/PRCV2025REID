#!/usr/bin/env python3
# debug_ce_issue.py
# 诊断CE损失为什么卡在5.95不下降

import torch
import torch.nn as nn
import numpy as np
from train import train_multimodal_reid, MultiModalDataset, CLIPBasedMultiModalReIDModel
from configs.config import TrainingConfig

def diagnose_ce_issue():
    """诊断CE损失问题的根本原因"""
    
    print("=== CE损失诊断工具 ===")
    
    # 快速配置
    config = TrainingConfig()
    
    # 加载少量数据进行测试
    dataset = MultiModalDataset(config, split='train')
    
    # 假设有N个ID
    all_person_ids = list(set(item['person_id'] for item in dataset.data_list))
    config.num_classes = len(all_person_ids)
    
    print(f"数据集信息:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  ID数量: {config.num_classes}")
    print(f"  理论随机CE: {np.log(config.num_classes):.3f}")
    
    # 创建模型
    model = CLIPBasedMultiModalReIDModel(config)
    model.set_num_classes(config.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 获取一个小batch测试
    from torch.utils.data import DataLoader
    from datasets.dataset import compatible_collate_fn
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=compatible_collate_fn)
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)
    labels = batch['person_id']
    
    print(f"\n=== 批次信息 ===")
    print(f"batch_size: {labels.shape[0]}")
    print(f"labels范围: {labels.min().item()} - {labels.max().item()}")
    print(f"labels类型: {labels.dtype}")
    print(f"唯一ID数: {len(torch.unique(labels))}")
    
    # ✅ 关键检查1：标签连续性
    unique_labels = torch.unique(labels).cpu().numpy()
    expected_range = np.arange(unique_labels.min(), unique_labels.max() + 1)
    if not np.array_equal(unique_labels, expected_range):
        print(f"❌ 标签不连续! 唯一标签: {unique_labels}")
        print(f"   期望连续范围: {expected_range}")
        print("   修复方案: 使用LabelEncoder重映射标签")
        return False
    
    # ✅ 关键检查2：标签是否超出num_classes范围
    if labels.max().item() >= config.num_classes:
        print(f"❌ 标签超出范围! max_label={labels.max().item()}, num_classes={config.num_classes}")
        print("   修复方案: 重新设置num_classes或修正标签映射")
        return False
    
    # ✅ 关键检查3：分类器权重是否在优化器中
    model.train()
    
    # 创建简单的优化器测试
    params_with_grad = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_with_grad.append((name, param.numel()))
            if 'bn_neck.classifier' in name or 'classifier' in name:
                classifier_params.append((name, param.numel()))
    
    print(f"\n=== 参数检查 ===")
    print(f"可训练参数组: {len(params_with_grad)}")
    print(f"分类器参数组: {len(classifier_params)}")
    
    if not classifier_params:
        print("❌ 分类器参数不在可训练参数中!")
        return False
    
    for name, count in classifier_params:
        print(f"  {name}: {count:,} 参数")
    
    # ✅ 关键检查4：前向传播和梯度
    print(f"\n=== 前向传播测试 ===")
    
    try:
        # 前向传播
        with torch.amp.autocast('cuda', enabled=True):
            from train import call_model_with_batch
            outputs = call_model_with_batch(model, batch, return_features=False)
        
        if 'logits' not in outputs:
            print("❌ 模型输出缺少logits!")
            return False
        
        logits = outputs['logits']
        print(f"logits形状: {logits.shape}")
        print(f"logits范围: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        print(f"logits均值: {logits.mean().item():.3f}")
        print(f"logits标准差: {logits.std().item():.3f}")
        
        # 计算CE损失
        ce_loss = nn.CrossEntropyLoss()(logits, labels)
        print(f"当前CE损失: {ce_loss.item():.4f}")
        
        # ✅ 关键检查5：梯度反向传播
        ce_loss.backward()
        
        # 检查分类器的梯度
        classifier_grad_norms = []
        for name, param in model.named_parameters():
            if 'classifier' in name and param.grad is not None:
                grad_norm = param.grad.norm().item()
                classifier_grad_norms.append(grad_norm)
                print(f"  {name} 梯度范数: {grad_norm:.6f}")
        
        if not classifier_grad_norms:
            print("❌ 分类器没有梯度!")
            return False
        elif all(g < 1e-6 for g in classifier_grad_norms):
            print("❌ 分类器梯度过小，可能被其他损失项压制")
            return False
        
        print("✅ CE损失计算和梯度反向传播正常")
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False

def move_batch_to_device(batch, device):
    """简化版的batch移动函数"""
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(x, device) for x in batch]
    elif torch.is_tensor(batch):
        return batch.to(device)
    else:
        return batch

if __name__ == "__main__":
    diagnose_ce_issue()