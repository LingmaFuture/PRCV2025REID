#!/usr/bin/env python3
"""
测试同名映射的完整性
验证Dataset -> Model的模态名称一致性
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_same_name_mapping():
    """测试同名映射配置"""
    print("🧪 测试同名映射配置")
    print("=" * 50)
    
    # 测试映射函数
    try:
        from train import MODALITY_MAPPING, map_modality_name
        
        print("📋 当前MODALITY_MAPPING:")
        for k, v in sorted(MODALITY_MAPPING.items()):
            identity = "✅" if k == v else "🔄"
            print(f"  {identity} '{k}' -> '{v}'")
        
        # 测试主要模态映射
        main_modalities = ['vis', 'nir', 'sk', 'cp', 'text']
        print(f"\n🔍 主要模态映射测试:")
        all_identity = True
        for mod in main_modalities:
            mapped = map_modality_name(mod)
            is_identity = (mod == mapped)
            status = "✅" if is_identity else "❌"
            print(f"  {status} {mod} -> {mapped}")
            if not is_identity:
                all_identity = False
        
        if all_identity:
            print("🎉 所有主要模态都是恒等映射！")
        else:
            print("⚠️  存在非恒等映射")
            
    except Exception as e:
        print(f"❌ 映射测试失败: {e}")
        return False
    
    # 测试配置文件
    try:
        from configs.config import TrainingConfig
        config = TrainingConfig()
        
        print(f"\n📝 模型配置测试:")
        print(f"  config.modalities = {config.modalities}")
        
        expected = ['vis', 'nir', 'sk', 'cp', 'text']
        if config.modalities == expected:
            print("✅ 配置与dataset命名一致")
        else:
            print(f"❌ 配置不匹配，期望: {expected}")
            return False
            
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False
    
    # 测试模型兼容性
    try:
        from models.model import CLIPBasedMultiModalReIDModel
        model = CLIPBasedMultiModalReIDModel(config)
        
        print(f"\n🤖 模型配置测试:")
        print(f"  model.modalities = {model.modalities}")
        print(f"  model.vision_modalities = {model.vision_modalities}")
        
        if 'vis' in model.modalities and 'vis' in model.vision_modalities:
            print("✅ 模型支持vis模态")
        else:
            print("❌ 模型不支持vis模态")
            return False
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 同名映射测试通过！")
    print("\n✅ 验证结果:")
    print("- 映射逻辑: 恒等映射")
    print("- 配置一致: Dataset ←→ Model")
    print("- 架构简化: 无需命名转换")
    
    return True

def test_model_forward():
    """测试模型前向传播"""
    try:
        print("\n🔄 模型前向传播测试:")
        
        from configs.config import TrainingConfig
        from models.model import CLIPBasedMultiModalReIDModel
        import torch
        
        config = TrainingConfig()
        model = CLIPBasedMultiModalReIDModel(config)
        model.set_num_classes(10)
        
        # 使用同名模态构建输入
        images = {
            'vis': torch.randn(2, 3, 224, 224),
            'nir': torch.randn(2, 3, 224, 224),
            'sk': torch.randn(2, 3, 224, 224),
            'cp': torch.randn(2, 3, 224, 224)
        }
        
        texts = ["person walking", "一个行人"]
        
        modality_masks = {
            'vis': torch.ones(2),
            'nir': torch.ones(2),
            'sk': torch.ones(2),
            'cp': torch.ones(2),
            'text': torch.ones(2)
        }
        
        # 前向传播
        with torch.no_grad():
            outputs = model(images=images, texts=texts, modality_masks=modality_masks)
            
        print(f"✅ 前向传播成功")
        print(f"  输出keys: {list(outputs.keys())}")
        if 'logits' in outputs:
            print(f"  logits shape: {outputs['logits'].shape}")
        if 'features' in outputs:
            print(f"  features shape: {outputs['features'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_same_name_mapping()
    success2 = test_model_forward()
    
    if success1 and success2:
        print(f"\n🌟 完整测试通过！同名映射工作正常。")
        sys.exit(0)
    else:
        print(f"\n💥 测试失败，需要检查配置")
        sys.exit(1)
