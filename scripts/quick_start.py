#!/usr/bin/env python3
"""
快速启动脚本 - 按guide.md要求优先跑通训练
"""
import os
import sys
import logging
import traceback

def setup_quick_logging():
    """设置快速日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('./logs/quick_start.log')
        ]
    )

def main():
    """主函数"""
    try:
        # 创建必要目录
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./checkpoints', exist_ok=True)
        
        setup_quick_logging()
        
        logging.info("🚀 开始快速启动训练...")
        logging.info("📋 按guide.md配置：小batch_size + 简化采样器")
        
        # 导入训练函数
        from train import train_multimodal_reid
        
        # 启动训练
        train_multimodal_reid()
        
        logging.info("✅ 训练完成!")
        
    except Exception as e:
        logging.error(f"❌ 训练失败: {str(e)}")
        logging.error("详细错误信息:")
        logging.error(traceback.format_exc())
        
        # 提供修复建议
        logging.info("\n🔧 可能的修复方案:")
        logging.info("1. 检查数据路径是否正确")
        logging.info("2. 确认GPU内存充足")
        logging.info("3. 尝试进一步降低batch_size")
        logging.info("4. 检查依赖库版本")
        
        sys.exit(1)

if __name__ == "__main__":
    main()