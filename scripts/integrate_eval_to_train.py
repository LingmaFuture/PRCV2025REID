# tools/integrate_eval_to_train.py
"""
集成评估协议到训练流程
在训练完成后自动运行多模态评估协议
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.eval_mm_protocol import run_eval
from configs.config import TrainingConfig

def run_post_training_evaluation(
    dataset_root: str = "./data/train",
    model_path: str = "./checkpoints/best_model.pth",
    cache_dir: str = "./eval_cache",
    seed: int = 42,
    device: str = "cuda"
):
    """
    训练后评估函数，可以直接在train.py中调用
    
    Args:
        dataset_root: 数据集根目录
        model_path: 最佳模型权重路径
        cache_dir: 评估缓存目录
        seed: 随机种子（保证可复现）
        device: 计算设备
    
    Returns:
        评估结果字典
    """
    print("\n" + "="*60)
    print("🎯 开始运行多模态评估协议（MM-1/2/3/4）")
    print("="*60)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请确保训练已完成并保存了最佳模型")
        return None
    
    try:
        # 运行评估协议
        results = run_eval(
            dataset_root=dataset_root,
            model_path=model_path,
            cache_dir=cache_dir,
            seed=seed,
            device=device
        )
        
        # 保存评估结果
        import json
        result_path = os.path.join(os.path.dirname(model_path), "mm_evaluation_results.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n📊 评估结果已保存到: {result_path}")
        
        # 打印总结
        print("\n" + "="*60)
        print("📈 评估总结")
        print("="*60)
        
        avg_results = results.get("AVG(1-4)", {})
        print(f"平均 mAP: {avg_results.get('mAP', 0.0):.4f}")
        print(f"平均 R@1: {avg_results.get('R@1', 0.0):.4f}")
        print(f"平均 R@5: {avg_results.get('R@5', 0.0):.4f}")
        print(f"平均 R@10: {avg_results.get('R@10', 0.0):.4f}")
        
        print("\n分模态详细结果:")
        for mode in ["MM-1", "MM-2", "MM-3", "MM-4"]:
            if mode in results:
                mode_res = results[mode]
                print(f"{mode}: mAP={mode_res.get('mAP', 0.0):.4f}, "
                      f"R@1={mode_res.get('R@1', 0.0):.4f}, "
                      f"查询数={mode_res.get('num_queries', 0)}")
        
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"❌ 评估过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def add_evaluation_to_train_script():
    """
    提供代码片段，可以添加到train.py的末尾
    """
    code_snippet = '''
# ===== 训练完成后运行多模态评估协议 =====
def run_final_evaluation():
    """训练完成后的最终评估"""
    print("\\n" + "="*60)
    print("🎯 开始最终多模态评估协议")
    print("="*60)
    
    try:
        from tools.integrate_eval_to_train import run_post_training_evaluation
        
        # 运行评估
        eval_results = run_post_training_evaluation(
            dataset_root=config.data_root,
            model_path=os.path.join(config.save_dir, 'best_model.pth'),
            cache_dir=os.path.join(config.save_dir, 'eval_cache'),
            seed=config.seed,
            device=str(device)
        )
        
        if eval_results:
            # 将评估结果添加到训练历史
            final_eval_summary = {
                'final_mm_avg_map': eval_results.get("AVG(1-4)", {}).get('mAP', 0.0),
                'final_mm_avg_r1': eval_results.get("AVG(1-4)", {}).get('R@1', 0.0),
                'mm1_map': eval_results.get("MM-1", {}).get('mAP', 0.0),
                'mm2_map': eval_results.get("MM-2", {}).get('mAP', 0.0),
                'mm3_map': eval_results.get("MM-3", {}).get('mAP', 0.0),
                'mm4_map': eval_results.get("MM-4", {}).get('mAP', 0.0),
            }
            
            # 保存到CSV
            import pandas as pd
            final_df = pd.DataFrame([final_eval_summary])
            final_df.to_csv(os.path.join(log_dir, 'final_mm_evaluation.csv'), index=False)
            
            logging.info(f"✅ 最终多模态评估完成！平均mAP: {final_eval_summary['final_mm_avg_map']:.4f}")
            
        else:
            logging.warning("⚠️ 最终评估失败，请手动运行评估脚本")
            
    except Exception as e:
        logging.error(f"❌ 最终评估出错: {e}")

# 在train_multimodal_reid()函数的最后添加：
if __name__ == "__main__":
    # 原有的训练代码...
    train_multimodal_reid()
    
    # 训练完成后运行最终评估
    try:
        run_final_evaluation()
    except Exception as e:
        print(f"最终评估失败: {e}")
        print("可以手动运行: python tools/eval_mm_protocol.py --model_path ./checkpoints/best_model.pth")
'''
    
    return code_snippet

if __name__ == "__main__":
    # 示例：运行评估
    import argparse
    
    parser = argparse.ArgumentParser(description="训练后评估")
    parser.add_argument("--dataset_root", type=str, default="./data/train", help="数据集根目录")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pth", help="模型权重路径")
    parser.add_argument("--cache_dir", type=str, default="./eval_cache", help="缓存目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    args = parser.parse_args()
    
    # 运行评估
    results = run_post_training_evaluation(
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        seed=args.seed,
        device=args.device
    )
    
    if results:
        print("🎉 评估完成！")
    else:
        print("❌ 评估失败！")
