# tools/test_clip_mer_integration.py
"""
CLIP+MER架构集成测试脚本
验证新架构的核心功能是否正常工作
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
from configs.config import TrainingConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_clip_mer_integration():
    """测试CLIP+MER架构的基本功能"""
    
    print("🚀 开始CLIP+MER架构集成测试...")
    
    # 1. 测试配置加载
    print("\n📋 测试1: 配置加载")
    try:
        config = TrainingConfig()
        
        # 检查新添加的配置项
        required_configs = [
            'clip_model_name', 'modalities', 'mer_lora_rank', 
            'base_learning_rate', 'mer_learning_rate'
        ]
        
        for cfg_name in required_configs:
            if hasattr(config, cfg_name):
                value = getattr(config, cfg_name)
                print(f"  ✅ {cfg_name}: {value}")
            else:
                print(f"  ❌ 缺少配置项: {cfg_name}")
                return False
        
        print("  ✅ 配置加载成功")
        
    except Exception as e:
        print(f"  ❌ 配置加载失败: {e}")
        return False
    
    # 2. 测试模型导入和实例化
    print("\n🏗️ 测试2: 模型导入和实例化")
    try:
        from models.model import CLIPBasedMultiModalReIDModel
        
        # 创建模型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.device = device
        
        model = CLIPBasedMultiModalReIDModel(config)
        model = model.to(device)
        
        # 设置分类器
        model.set_num_classes(100)  # 假设100个ID
        
        print(f"  ✅ 模型创建成功，设备: {device}")
        print(f"  ✅ 支持模态: {model.modalities}")
        print(f"  ✅ 融合维度: {model.fusion_dim}")
        
    except Exception as e:
        print(f"  ❌ 模型实例化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 测试前向传播
    print("\n⚡ 测试3: 前向传播")
    try:
        batch_size = 4
        
        # 准备测试数据 (注意通道数匹配)
        test_images = {
            'rgb': torch.randn(batch_size, 3, 224, 224, device=device),      # RGB: 3通道
            'ir': torch.randn(batch_size, 1, 224, 224, device=device),       # IR: 1通道  
            'cpencil': torch.randn(batch_size, 3, 224, 224, device=device),  # 彩铅: 3通道
            'sketch': torch.randn(batch_size, 1, 224, 224, device=device),   # 素描: 1通道
        }
        
        test_texts = [
            "A person walking in the street",
            "一个行人在路上行走",
            "Person wearing red jacket",
            "穿红色外套的人"
        ]
        
        # 前向传播
        with torch.no_grad():
            # 测试视觉+文本
            outputs = model(images=test_images, texts=test_texts)
            print(f"  ✅ 视觉+文本前向传播成功")
            print(f"    - Features shape: {outputs['features'].shape}")
            print(f"    - Logits shape: {outputs['logits'].shape}")
            
            # 测试仅视觉 (仅测试RGB模态)
            outputs_vis = model(images={'rgb': test_images['rgb']}, texts=None)
            print(f"  ✅ 仅视觉前向传播成功")
            print(f"    - Features shape: {outputs_vis['features'].shape}")
            
            # 测试单个IR模态
            outputs_ir = model(images={'ir': test_images['ir']}, texts=None)
            print(f"  ✅ IR模态前向传播成功")
            print(f"    - Features shape: {outputs_ir['features'].shape}")
            
            # 测试仅文本
            outputs_text = model(images=None, texts=test_texts)
            print(f"  ✅ 仅文本前向传播成功")
            print(f"    - Features shape: {outputs_text['features'].shape}")
        
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试损失计算
    print("\n💸 测试4: 损失计算")
    try:
        # 准备标签
        labels = torch.randint(0, 100, (batch_size,), device=device)
        
        # 计算损失
        loss_dict = model.compute_loss(outputs, labels)
        
        required_losses = ['total_loss', 'ce_loss', 'sdm_loss', 'feat_penalty']
        for loss_name in required_losses:
            if loss_name in loss_dict:
                loss_value = loss_dict[loss_name].item()
                print(f"  ✅ {loss_name}: {loss_value:.4f}")
            else:
                print(f"  ❌ 缺少损失项: {loss_name}")
                return False
        
    except Exception as e:
        print(f"  ❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 测试参数分组
    print("\n⚙️ 测试5: 参数分组")
    try:
        param_groups = model.get_learnable_params()
        
        total_params = 0
        for group in param_groups:
            group_name = group.get('name', 'unknown')
            group_lr = group['lr']
            num_params = len(group['params'])
            total_params += num_params
            print(f"  ✅ {group_name}: {num_params} 参数, LR: {group_lr:.2e}")
        
        print(f"  ✅ 总参数组数: {len(param_groups)}, 总参数数: {total_params}")
        
    except Exception as e:
        print(f"  ❌ 参数分组失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 测试训练脚本的兼容性函数
    print("\n🔄 测试6: 训练脚本兼容性")
    try:
        # 导入训练脚本中的辅助函数
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from train import convert_batch_for_clip_model, call_model_with_batch, move_batch_to_device
        
        # 模拟数据集batch格式
        mock_batch = {
            'images': {
                'vis': torch.randn(2, 3, 224, 224),
                'nir': torch.randn(2, 1, 224, 224),
            },
            'text_description': ['Person walking', 'A man in blue shirt'],
            'person_id': torch.tensor([1, 2])
        }
        
        # 测试batch转换
        images, texts = convert_batch_for_clip_model(mock_batch)
        print(f"  ✅ Batch转换成功: {list(images.keys())} + {len(texts) if texts else 0} texts")
        
        # 测试设备移动
        mock_batch = move_batch_to_device(mock_batch, device)
        print(f"  ✅ 设备移动成功")
        
        # 测试模型调用
        outputs = call_model_with_batch(model, mock_batch, return_features=True)
        print(f"  ✅ 兼容性模型调用成功: {outputs['features'].shape}")
        
    except Exception as e:
        print(f"  ❌ 训练脚本兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 所有测试通过！CLIP+MER架构集成成功！")
    print("\n📝 架构总结:")
    print("  - ✅ CLIP-B/16统一编码器")
    print("  - ✅ 多模态非共享tokenizer (RGB/IR/cpencil/sketch/text)")
    print("  - ✅ MER模态路由LoRA (rank=4)")
    print("  - ✅ SDM语义分离模块")
    print("  - ✅ 特征融合器")
    print("  - ✅ 分层学习率优化")
    print("  - ✅ 训练脚本兼容性")
    
    return True


if __name__ == "__main__":
    success = test_clip_mer_integration()
    if success:
        print("\n✨ 可以开始训练了！运行: python train.py")
    else:
        print("\n❌ 请修复上述错误后再继续")
        sys.exit(1)
