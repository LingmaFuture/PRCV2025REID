#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试轻量特征混合器的兼容性和功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from configs.config import TrainingConfig
from models.model import LightweightFeatureMixer, MultiModalReIDModel


def test_lightweight_mixer_standalone():
    """测试轻量特征混合器的独立功能"""
    print("=== 测试轻量特征混合器独立功能 ===")
    
    # 创建测试配置
    embed_dim = 768
    batch_size = 4
    device = torch.device('cpu')  # 使用CPU测试以避免CUDA依赖
    
    # 初始化轻量特征混合器
    mixer = LightweightFeatureMixer(
        embed_dim=embed_dim,
        num_heads=8,
        mlp_ratio=2.0,
        dropout=0.1
    ).to(device)
    
    print(f"轻量特征混合器参数量: {sum(p.numel() for p in mixer.parameters()):,}")
    
    # 创建测试输入
    modality_features = {
        'vis': torch.randn(batch_size, embed_dim, device=device),
        'nir': torch.randn(batch_size, embed_dim, device=device), 
        'sk': torch.randn(batch_size, embed_dim, device=device),
        'cp': torch.randn(batch_size, embed_dim, device=device),
        'text': torch.randn(batch_size, embed_dim, device=device),
    }
    
    modality_mask = {
        'vis': True,
        'nir': True,
        'sk': True, 
        'cp': True,
        'text': True,
    }
    
    # 前向传播测试
    print("\n--- 完整模态测试 ---")
    with torch.no_grad():
        output = mixer(modality_features, modality_mask)
        print(f"输入特征形状: {[(k, v.shape) for k, v in modality_features.items()]}")
        print(f"输出特征形状: {output.shape}")
        print(f"输出特征范数: {output.norm(dim=1).mean().item():.4f}")
    
    # 部分模态测试
    print("\n--- 部分模态测试 ---")
    partial_mask = {
        'vis': True,
        'nir': False,
        'sk': True,
        'cp': False,
        'text': True,
    }
    
    with torch.no_grad():
        output_partial = mixer(modality_features, partial_mask)
        print(f"部分模态输出形状: {output_partial.shape}")
        print(f"部分模态输出范数: {output_partial.norm(dim=1).mean().item():.4f}")
    
    # 单模态测试
    print("\n--- 单模态测试 ---")
    single_modal_features = {'vis': modality_features['vis']}
    single_modal_mask = {'vis': True}
    
    with torch.no_grad():
        output_single = mixer(single_modal_features, single_modal_mask)
        print(f"单模态输出形状: {output_single.shape}")
        print(f"单模态输出范数: {output_single.norm(dim=1).mean().item():.4f}")
    
    print("✓ 轻量特征混合器独立功能测试通过")
    return True


def test_model_integration():
    """测试与整个模型的集成"""
    print("\n=== 测试模型集成 ===")
    
    # 创建配置
    config = TrainingConfig()
    config.num_classes = 100  # 设置一个测试用的类别数
    
    # 设置轻量特征混合器参数
    config.fusion_num_heads = 8
    config.fusion_mlp_ratio = 2.0
    config.fusion_dropout = 0.1
    
    try:
        # 初始化模型（CPU模式）
        model = MultiModalReIDModel(config)
        model.eval()
        
        print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"融合模块参数量: {sum(p.numel() for p in model.fusion_module.parameters()):,}")
        
        # 创建测试批次
        batch_size = 2
        batch = {
            'images': {
                'vis': torch.randn(batch_size, 3, 224, 224),
                'nir': torch.randn(batch_size, 3, 224, 224),
                'sk': torch.randn(batch_size, 3, 224, 224),
                'cp': torch.randn(batch_size, 3, 224, 224),
            },
            'text_description': ['person wearing red shirt', 'person in blue jeans'],
            'modality_mask': {
                'vis': [1.0, 1.0],
                'nir': [1.0, 0.0],  # 第二个样本没有NIR
                'sk': [1.0, 1.0],
                'cp': [0.0, 1.0],   # 第一个样本没有CP
                'text': [1.0, 1.0],
            }
        }
        
        # 前向传播测试
        with torch.no_grad():
            outputs = model(batch, return_features=False)
            
            print(f"模型输出键: {list(outputs.keys())}")
            print(f"logits形状: {outputs['logits'].shape}")
            print(f"features形状: {outputs['features'].shape}")
            print(f"reid_features形状: {outputs['reid_features'].shape}")
            
        print("✓ 模型集成测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 模型集成测试失败: {str(e)}")
        return False


def compare_with_original_fusion():
    """比较轻量特征混合器与原层次化融合的参数量"""
    print("\n=== 参数量对比 ===")
    
    embed_dim = 768
    
    # 原层次化融合参数量
    from models.model import HierarchicalMultiModalFusion
    original_fusion = HierarchicalMultiModalFusion(embed_dim, num_layers=3, num_heads=8)
    original_params = sum(p.numel() for p in original_fusion.parameters())
    
    # 轻量特征混合器参数量
    lightweight_fusion = LightweightFeatureMixer(embed_dim, num_heads=8, mlp_ratio=2.0)
    lightweight_params = sum(p.numel() for p in lightweight_fusion.parameters())
    
    print(f"原层次化融合参数量: {original_params:,}")
    print(f"轻量特征混合器参数量: {lightweight_params:,}")
    print(f"参数量减少: {(original_params - lightweight_params) / original_params * 100:.2f}%")
    print(f"参数量比率: {lightweight_params / original_params:.3f}")
    
    return True


def main():
    """主测试函数"""
    print("轻量特征混合器测试套件")
    print("=" * 50)
    
    tests = [
        ("轻量特征混合器独立功能", test_lightweight_mixer_standalone),
        ("模型集成", test_model_integration),
        ("参数量对比", compare_with_original_fusion),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name}测试失败: {str(e)}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("测试总结:")
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n总体结果: {'✓ 全部通过' if all_passed else '✗ 有测试失败'}")
    
    return all_passed


if __name__ == "__main__":
    main()
