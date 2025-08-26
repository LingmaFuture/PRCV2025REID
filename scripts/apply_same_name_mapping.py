#!/usr/bin/env python3
"""
应用同名映射：统一使用dataset原生命名，消除映射复杂性
将所有'rgb'引用改为'vis'，实现完全的同名映射
"""
import os
import re

def apply_same_name_mapping():
    """应用同名映射的完整修改"""
    
    print("🔧 应用同名映射方案")
    print("=" * 50)
    
    # 需要修改的文件和对应的替换规则
    modifications = [
        {
            'file': 'models/model.py',
            'changes': [
                # 修改硬编码的反向映射表
                (r"for orig, new in \[\('vis', 'rgb'\), \('nir', 'ir'\), \('sk', 'sketch'\), \('cp', 'cpencil'\)\]:",
                 "# 同名映射后不需要反向查找，直接使用原名"),
                
                # 修改模态dropout中的'rgb'引用
                (r"# 永不drop 'rgb'，优先保留主模态", 
                 "# 永不drop 'vis'，优先保留主模态"),
                (r"if mod == 'rgb' or torch.rand\(1\)\.item\(\) > modality_dropout:",
                 "if mod == 'vis' or torch.rand(1).item() > modality_dropout:"),
                
                # 修改SDM损失计算中的RGB特征获取
                (r"rgb_features = raw_modality_features\.get\('rgb', None\)",
                 "vis_features = raw_modality_features.get('vis', None)"),
                (r"rgb_mask = feature_masks\.get\('rgb', None\)",
                 "vis_mask = feature_masks.get('vis', None)"),
                
                # 修改条件判断
                (r"if rgb_features is None or rgb_mask is None:",
                 "if vis_features is None or vis_mask is None:"),
                (r"# 回退：如果没有RGB特征或mask，跳过SDM对齐",
                 "# 回退：如果没有vis特征或mask，跳过SDM对齐"),
                
                # 修改变量名
                (r"rgb_valid_idx", "vis_valid_idx"),
                (r"rgb_valid_feat", "vis_valid_feat"), 
                (r"rgb_valid_labels", "vis_valid_labels"),
                
                # 修改注释和日志
                (r"# 没有有效RGB样本，跳过对齐", "# 没有有效vis样本，跳过对齐"),
                (r"# 过滤出有效的RGB特征和标签", "# 过滤出有效的vis特征和标签"),
                (r"# 扩展当前batch的RGB特征和标签", "# 扩展当前batch的vis特征和标签"),
                (r"# 缓存当前batch的RGB特征", "# 缓存当前batch的vis特征"),
                (r"# 缓存有效的RGB特征", "# 缓存有效的vis特征"),
                
                # 修改循环中的模态名比较
                (r"if mod_name == 'rgb':", "if mod_name == 'vis':"),
                
                # 修改SDM损失计算的注释
                (r"# 模态特征 -> RGB特征的SDM对齐", "# 模态特征 -> vis特征的SDM对齐"),
                
                # 修改测试用例
                (r"modalities = \['rgb', 'ir', 'cpencil', 'sketch', 'text'\]",
                 "modalities = ['vis', 'nir', 'sk', 'cp', 'text']"),
                (r"'rgb': torch\.randn\(2, 3, 224, 224\)",
                 "'vis': torch.randn(2, 3, 224, 224)")
            ]
        },
        {
            'file': 'train.py', 
            'changes': [
                # 修改反向映射逻辑，简化为直接使用原名
                (r"# 需要从原始模态名映射回来.*?original_modality = orig.*?break",
                 "# 同名映射：直接使用模态名\n                        original_modality = modality", 
                 re.DOTALL)
            ]
        }
    ]
    
    # 执行修改
    for mod in modifications:
        file_path = mod['file']
        print(f"\n📝 修改文件: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"⚠️  文件不存在: {file_path}")
            continue
            
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 应用所有替换
            original_content = content
            for pattern, replacement, *flags in mod['changes']:
                if flags and re.DOTALL in flags:
                    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
                else:
                    content = content.replace(pattern, replacement)
            
            # 保存修改
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ {file_path} 修改完成")
            else:
                print(f"➡️  {file_path} 无需修改")
                
        except Exception as e:
            print(f"❌ {file_path} 修改失败: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 同名映射应用完成！")
    print("\n✅ 修改总结:")
    print("- Dataset使用: ['vis', 'nir', 'sk', 'cp', 'text']") 
    print("- Model使用:   ['vis', 'nir', 'sk', 'cp', 'text']")
    print("- 映射逻辑:    恒等映射，无需转换")
    print("- 架构简化:    消除两套命名系统")
    
    print("\n📋 后续验证:")
    print("1. 运行训练确认无模态匹配错误")
    print("2. 检查SDM损失计算是否正常")
    print("3. 验证评测流程中的模态识别")

if __name__ == "__main__":
    apply_same_name_mapping()
