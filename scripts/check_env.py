#!/usr/bin/env python3
"""
环境检查脚本 - 确保训练前环境就绪
"""
import os
import sys
import importlib
import torch

def check_pytorch():
    """检查PyTorch环境"""
    print("🔍 检查PyTorch环境...")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory // 1024**3
            print(f"   GPU {i}: {name} ({mem}GB)")
    
def check_required_packages():
    """检查必需包"""
    required_packages = [
        'transformers',
        'datasets', 
        'numpy',
        'pandas',
        'tqdm',
        'sklearn',
        'PIL',
        'torchvision'
    ]
    
    print("\n📦 检查必需包...")
    missing = []
    
    for pkg in required_packages:
        try:
            module = importlib.import_module(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {pkg}: {version}")
        except ImportError:
            print(f"   ❌ {pkg}: 未安装")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️ 缺少包: {', '.join(missing)}")
        print("安装命令: pip install " + ' '.join(missing))
        return False
    
    return True

def check_data_paths():
    """检查数据路径"""
    print("\n📁 检查数据路径...")
    
    # 检查配置中的路径
    try:
        from configs.config import TrainingConfig
        config = TrainingConfig()
        
        paths_to_check = [
            ("数据根目录", config.data_root),
            ("JSON标注文件", config.json_file),
            ("日志目录", config.log_dir),
            ("检查点目录", config.save_dir)
        ]
        
        all_good = True
        for name, path in paths_to_check:
            if os.path.exists(path):
                print(f"   ✅ {name}: {path}")
            else:
                print(f"   ❌ {name}: {path} (不存在)")
                all_good = False
                
        return all_good
        
    except Exception as e:
        print(f"   ❌ 配置检查失败: {e}")
        return False

def check_memory():
    """检查内存状况"""
    print("\n💾 检查内存状况...")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            free = total - reserved
            
            print(f"   GPU {i} 内存:")
            print(f"     总量: {total//1024**3:.1f}GB")
            print(f"     已分配: {allocated//1024**3:.1f}GB") 
            print(f"     已保留: {reserved//1024**3:.1f}GB")
            print(f"     可用: {free//1024**3:.1f}GB")
            
            if free < 2 * 1024**3:  # 小于2GB
                print(f"     ⚠️ GPU内存可能不足，建议清理或降低batch_size")

def main():
    """主检查函数"""
    print("🔧 环境检查开始...\n")
    
    # 各项检查
    checks = [
        ("PyTorch环境", check_pytorch),
        ("必需包", check_required_packages), 
        ("数据路径", check_data_paths),
        ("内存状况", check_memory)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if result is False:
                all_passed = False
        except Exception as e:
            print(f"   ❌ {check_name}检查异常: {e}")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ 环境检查通过! 可以开始训练")
        print("启动命令: python quick_start.py")
    else:
        print("❌ 环境检查失败，请修复上述问题后再试")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)