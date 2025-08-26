#!/usr/bin/env python3
"""
测试修复后的SDM损失在模型中的使用
验证是否真正使用了非负的sdm_loss_stable
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_fixed_sdm_in_model():
    """测试模型中的修复后SDM损失"""
    print("[TEST] 测试模型中修复后的SDM损失...")
    
    try:
        from configs.config import TrainingConfig
        from models.model import CLIPBasedMultiModalReIDModel
        
        # 创建配置和模型
        config = TrainingConfig()
        config.num_classes = 100  # 设置类别数
        
        print(f"配置中的SDM温度: {getattr(config, 'sdm_temperature', 'NOT_FOUND')}")
        
        # 创建模型
        model = CLIPBasedMultiModalReIDModel(config)
        model.set_num_classes(config.num_classes)
        
        print(f"模型中的SDM温度: {getattr(model, 'sdm_temperature', 'NOT_FOUND')}")
        
        # 模拟输入
        batch_size = 16
        feature_dim = 768
        
        # 模拟模型输出
        outputs = {
            'logits': torch.randn(batch_size, config.num_classes),
            'features': torch.randn(batch_size, feature_dim),
            'bn_features': torch.randn(batch_size, feature_dim),
            'raw_modality_features': {
                'rgb': torch.randn(batch_size, feature_dim),
                'nir': torch.randn(batch_size, feature_dim),
                'sk': torch.randn(batch_size, feature_dim),
                'cp': torch.randn(batch_size, feature_dim),
            }
        }
        
        # 模拟标签
        labels = torch.randint(0, config.num_classes, (batch_size,))
        
        # 计算损失
        loss_dict = model.compute_loss(outputs, labels)
        
        print("[RESULT] 损失计算结果:")
        for key, value in loss_dict.items():
            loss_val = float(value.item()) if torch.is_tensor(value) else value
            if key == 'sdm_loss':
                status = "[OK] 非负" if loss_val >= 0 else "[FAIL] 负值"
                print(f"  {key}: {loss_val:.6f} {status}")
            else:
                print(f"  {key}: {loss_val:.6f}")
        
        # 验证SDM损失非负
        sdm_loss = loss_dict.get('sdm_loss', 0.0)
        if torch.is_tensor(sdm_loss):
            sdm_loss = float(sdm_loss.item())
        
        assert sdm_loss >= 0.0, f"SDM损失仍为负值: {sdm_loss}"
        assert torch.isfinite(torch.tensor(sdm_loss)), f"SDM损失为NaN/Inf: {sdm_loss}"
        
        print(f"[SUCCESS] SDM损失修复成功: {sdm_loss:.6f} >= 0")
        return True
        
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fixed_sdm_in_model()