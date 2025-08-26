#!/usr/bin/env python3
"""
测试统一的模态映射逻辑
验证MODALITY_MAPPING修改后的正确性
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import MODALITY_MAPPING, map_modality_name

def test_modality_mapping():
    """测试模态映射的各种情况"""
    
    print("🧪 测试统一的模态映射逻辑")
    print("=" * 50)
    
    # 测试用例：原始输入 -> 期望输出
    test_cases = [
        # 数据集格式映射
        ('vis', 'rgb'),
        ('nir', 'ir'),
        ('sk', 'sketch'),
        ('cp', 'cp'),          # 修改后应该是cp而不是cpencil
        ('text', 'text'),
        ('cpencil', 'cp'),     # 支持旧版本
        
        # 标准格式恒等映射
        ('rgb', 'rgb'),
        ('ir', 'ir'),
        ('sketch', 'sketch'),
        ('txt', 'text'),       # 简写支持
        
        # 大小写测试
        ('VIS', 'rgb'),
        ('NIR', 'ir'),
        ('CP', 'cp'),
        
        # 未知模态
        ('unknown', 'unknown'),
        ('xyz', 'xyz')
    ]
    
    print("📋 测试用例结果:")
    all_passed = True
    
    for input_mod, expected in test_cases:
        result = map_modality_name(input_mod)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_mod}' -> '{result}' (期望: '{expected}')")
        
        if result != expected:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过！统一映射逻辑工作正常")
    else:
        print("⚠️  部分测试失败，需要检查映射逻辑")
    
    # 显示当前的MODALITY_MAPPING
    print(f"\n📊 当前MODALITY_MAPPING内容:")
    for k, v in sorted(MODALITY_MAPPING.items()):
        print(f"  '{k}' -> '{v}'")
    
    return all_passed

def test_batch_extraction_simulation():
    """模拟批次模态提取测试"""
    print(f"\n🔄 模拟批次模态提取测试:")
    
    # 模拟不同的模态输入
    mock_modalities = ['vis', 'cp', 'nir', 'sk', 'text', 'cpencil', 'RGB', 'ir']
    
    print("原始模态 -> 标准化结果:")
    for mod in mock_modalities:
        standardized = map_modality_name(mod.lower())
        print(f"  {mod:8} -> {standardized}")

if __name__ == "__main__":
    success = test_modality_mapping()
    test_batch_extraction_simulation()
    
    if success:
        print(f"\n✅ 模态映射统一修改成功！")
        print("- 删除了冗余的MOD_MAP")
        print("- 统一使用MODALITY_MAPPING")
        print("- cp模态映射已修正为简洁命名")
        print("- 支持向后兼容和大小写不敏感")
    else:
        print(f"\n❌ 测试失败，请检查映射配置")
        sys.exit(1)
