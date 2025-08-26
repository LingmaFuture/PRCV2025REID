#!/usr/bin/env python3
"""
项目目录重组脚本
将当前混乱的test/目录重新组织为清晰的目录结构
"""
import os
import shutil
from pathlib import Path

# 目录映射规则
REORGANIZE_RULES = {
    # 正式单元测试 -> tests/
    "tests/": [
        "test_batched_forward.py",
        "test_clip_mer_integration.py", 
        "test_eval_protocol.py",
        "test_sampler.py",
        "test_sdm_complete.py",
        "test_modality_fix.py"
    ],
    
    # 调试脚本 -> debug/
    "debug/": [
        "debug_ce_convergence.py",
        "debug_ce_issue.py",
        "debug_guide18.py", 
        "debug_infer.py",
        "debug_modality_fixed.py",
        "debug_modality.py",
        "debug_sampler_stopiteration.py",
        "debug_sampling_issue.py",
        "debug_sampling.py",
        "debug_vis.py"
    ],
    
    # 生产工具 -> tools/
    "tools/": [
        "export_submission.py",
        "generate_submission.py",
        "model_backup.py", 
        "eval_mm_protocol.py",
        "integrate_eval_to_train.py",
        "cached_sampler.py"
    ],
    
    # 检查和快速脚本 -> scripts/ 
    "scripts/": [
        "check_clip_layers.py",
        "check_clip_projections.py", 
        "check_clip_tokens.py",
        "check_env.py",
        "quick_start.py",
        "quick_test.py",
        "quickfix_test.py",
        "inspect_clip_structure.py"
    ],
    
    # 实验验证脚本 -> experiments/
    "experiments/": [
        "test_conservative_scheduler.py",
        "test_fixed_sdm.py",
        "test_fixes.py", 
        "test_guide20_fix.py",
        "test_lightweight_mixer.py",
        "test_logging_fix.py",
        "test_performance_optimization.py",
        "test_sdm_fix.py", 
        "test_stability_fix.py",
        "test_stopiteration_fix.py",
        "sdm_head.py"
    ],
    
    # 特殊处理
    "experiments/evaluation/": [
        "evaluation/orbench_protocol.py"
    ]
}

def create_directories():
    """创建新的目录结构"""
    base_dirs = ["tests", "debug", "tools", "scripts", "experiments", "experiments/evaluation"]
    
    for dir_name in base_dirs:
        os.makedirs(dir_name, exist_ok=True)
        
        # 创建__init__.py让目录成为Python包
        if not dir_name.endswith("/evaluation"):
            init_file = Path(dir_name) / "__init__.py"
            init_file.touch()
    
    print("✅ 目录结构创建完成")

def move_files_dry_run():
    """干运行：显示将要进行的移动操作"""
    print("🔍 预览重组操作:")
    print("=" * 50)
    
    test_dir = Path("test")
    if not test_dir.exists():
        print("❌ test/ 目录不存在")
        return
    
    moved_files = set()
    
    for target_dir, files in REORGANIZE_RULES.items():
        print(f"\n📁 {target_dir}")
        for file_name in files:
            source_path = test_dir / file_name.replace("evaluation/", "")
            if source_path.exists():
                print(f"   {file_name} -> {target_dir}{file_name}")
                moved_files.add(file_name.replace("evaluation/", ""))
            else:
                print(f"   ⚠️  {file_name} (文件不存在)")
    
    # 显示未分类的文件
    all_files = [f.name for f in test_dir.rglob("*.py")]
    unmoved = set(all_files) - moved_files
    
    if unmoved:
        print(f"\n🗂️ 未分类的文件:")
        for file_name in sorted(unmoved):
            print(f"   {file_name}")

def generate_cleanup_summary():
    """生成清理总结"""
    print("\n📊 重组效果预估:")
    print("=" * 30)
    
    total_files = sum(len(files) for files in REORGANIZE_RULES.values())
    print(f"总共重组文件数: {total_files}")
    
    for target_dir, files in REORGANIZE_RULES.items():
        print(f"{target_dir:20} {len(files):2d} 个文件")
    
    print("\n✅ 重组后的优势:")
    print("- 清晰的目录分类，易于查找")
    print("- 正式测试与调试脚本分离") 
    print("- 生产工具统一管理")
    print("- 临时实验脚本隔离")
    print("- 符合Python项目最佳实践")

if __name__ == "__main__":
    print("🧹 PRCV2025REID 项目重组工具")
    print("=" * 50)
    
    create_directories()
    move_files_dry_run()
    generate_cleanup_summary()
    
    print("\n⚡ 下一步操作:")
    print("1. 检查上述预览结果")
    print("2. 确认无误后运行实际重组")
    print("3. 更新导入路径")
    print("4. 清理空的test/目录")
