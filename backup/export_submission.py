# tools/export_submission.py
"""
导出Kaggle提交CSV文件
基于多模态评估协议生成竞赛提交格式
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.eval_mm_protocol import (
    build_index, build_gallery, build_queries, 
    FeatureExtractor, extract_gallery_feats,
    extract_query_feat, cosine_sim, l2n
)
from datasets.dataset import MultiModalDataset
from models.model import CLIPBasedMultiModalReIDModel
from configs.config import TrainingConfig

def generate_submission_csv(
    dataset_root: str,
    model_path: str,
    output_csv: str = "./submission.csv",
    cache_dir: str = "./eval_cache",
    seed: int = 42,
    top_k: int = 100,
    weight_cfg: Dict[str, float] = None,
    device: str = "cuda"
):
    """
    生成Kaggle提交CSV文件
    
    Args:
        dataset_root: 数据集根目录
        model_path: 模型权重路径
        output_csv: 输出CSV文件路径
        cache_dir: 缓存目录
        seed: 随机种子
        top_k: 每个查询返回前K个结果
        weight_cfg: 模态权重配置
        device: 计算设备
    """
    print("🎯 开始生成Kaggle提交文件")
    print("=" * 50)
    
    # 设置随机种子
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    
    # 默认权重配置
    if weight_cfg is None:
        weight_cfg = {"ir": 1.0, "cpencil": 1.0, "sketch": 1.0, "text": 1.2}
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    config = TrainingConfig()
    model = CLIPBasedMultiModalReIDModel(config).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'num_classes' in checkpoint:
        model.set_num_classes(checkpoint['num_classes'])
    
    model.eval()
    
    # 创建特征提取器
    extractor = FeatureExtractor(model, device)
    
    # 加载数据集
    print("加载数据集...")
    config.data_root = dataset_root
    dataset = MultiModalDataset(config, split='val')
    
    # 构建数据索引
    index = build_index(dataset)
    
    # 构建Gallery
    gallery = build_gallery(index)
    print(f"Gallery大小: {len(gallery)} 张RGB图像")
    
    # 提取Gallery特征
    g_feats, g_meta = extract_gallery_feats(gallery, extractor, cache_dir)
    g_feats = l2n(g_feats)
    
    # 收集所有查询
    all_queries = []
    print("\n构建所有查询...")
    
    for k in [1, 2, 3, 4]:
        queries = build_queries(index, mode_k=k, rng=rng, main_mod_choice="lexi_first")
        print(f"MM-{k}: {len(queries)} 个查询")
        
        for q in queries:
            q['mode'] = f"MM-{k}"
            all_queries.append(q)
    
    print(f"总查询数: {len(all_queries)}")
    
    # 生成提交数据
    submission_data = []
    
    print("\n生成提交数据...")
    for q in tqdm(all_queries, desc="处理查询"):
        try:
            # 提取查询特征
            q_feat = extract_query_feat(q, extractor, weight_cfg).view(1, -1)
            
            # 计算相似度
            sims = cosine_sim(q_feat, g_feats).squeeze(0)  # [G]
            
            # 排序并取前K
            ranks = torch.argsort(sims, descending=True)[:top_k].tolist()
            
            # 生成query_key
            pid = q["pid"]
            mode = q["mode"]
            mods = "+".join(sorted(q["modalities"]))
            
            # 构建查询样本ID
            sample_ids = []
            for sample in q["samples"].values():
                if "img_id" in sample and sample["img_id"] is not None:
                    sample_ids.append(str(sample["img_id"]))
            
            query_key = f"{pid}|{mode}|{mods}|{'+'.join(sample_ids)}"
            
            # 获取排序后的gallery img_id
            ranked_gallery_ids = []
            for rank_idx in ranks:
                g_item = g_meta[rank_idx]
                g_img_id = g_item.get("img_id", "unknown")
                if g_img_id is not None:
                    ranked_gallery_ids.append(str(g_img_id))
            
            # 确保有足够的结果
            while len(ranked_gallery_ids) < top_k:
                ranked_gallery_ids.append("unknown")
            
            ranked_gallery_ids_str = " ".join(ranked_gallery_ids[:top_k])
            
            submission_data.append({
                "query_key": query_key,
                "ranked_gallery_ids": ranked_gallery_ids_str
            })
            
        except Exception as e:
            print(f"处理查询失败 {q}: {e}")
            continue
    
    # 保存CSV
    print(f"\n保存提交文件: {output_csv}")
    df = pd.DataFrame(submission_data)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    
    print(f"✅ 提交文件生成完成！")
    print(f"   文件路径: {output_csv}")
    print(f"   查询条目数: {len(submission_data)}")
    print(f"   每个查询返回: {top_k} 个结果")
    
    # 显示一些示例
    if len(submission_data) > 0:
        print(f"\n示例提交条目:")
        for i, item in enumerate(submission_data[:3]):
            print(f"  {i+1}. Query: {item['query_key']}")
            gallery_preview = " ".join(item['ranked_gallery_ids'].split()[:5])
            print(f"     Gallery (前5): {gallery_preview}...")
    
    return output_csv

def validate_submission_format(csv_path: str):
    """
    验证提交CSV格式是否正确
    
    Args:
        csv_path: CSV文件路径
    """
    print(f"\n🔍 验证提交文件格式: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # 检查必需的列
        required_cols = ["query_key", "ranked_gallery_ids"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ 缺少必需列: {missing_cols}")
            return False
        
        print(f"✅ 列名检查通过: {list(df.columns)}")
        
        # 检查数据
        print(f"✅ 提交条目数: {len(df)}")
        
        # 检查是否有空值
        null_queries = df['query_key'].isnull().sum()
        null_galleries = df['ranked_gallery_ids'].isnull().sum()
        
        if null_queries > 0:
            print(f"⚠️ 发现 {null_queries} 个空查询key")
        
        if null_galleries > 0:
            print(f"⚠️ 发现 {null_galleries} 个空gallery结果")
        
        # 检查查询key格式
        sample_queries = df['query_key'].head(5).tolist()
        print(f"✅ 示例查询key:")
        for i, qk in enumerate(sample_queries):
            print(f"   {i+1}. {qk}")
        
        # 检查gallery结果格式
        sample_gallery = df['ranked_gallery_ids'].iloc[0]
        gallery_count = len(sample_gallery.split())
        print(f"✅ 每个查询的gallery结果数: {gallery_count}")
        
        print(f"✅ 提交文件格式验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成Kaggle提交CSV")
    parser.add_argument("--dataset_root", type=str, default="./data/train", help="数据集根目录")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pth", help="模型权重路径")
    parser.add_argument("--output_csv", type=str, default="./submission.csv", help="输出CSV文件路径")
    parser.add_argument("--cache_dir", type=str, default="./eval_cache", help="缓存目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--top_k", type=int, default=100, help="每个查询返回前K个结果")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--validate", action="store_true", help="验证生成的CSV格式")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.dataset_root):
        print(f"❌ 数据集目录不存在: {args.dataset_root}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        sys.exit(1)
    
    # 生成提交文件
    try:
        output_path = generate_submission_csv(
            dataset_root=args.dataset_root,
            model_path=args.model_path,
            output_csv=args.output_csv,
            cache_dir=args.cache_dir,
            seed=args.seed,
            top_k=args.top_k,
            device=args.device
        )
        
        # 验证格式（如果需要）
        if args.validate:
            validate_submission_format(output_path)
            
    except Exception as e:
        print(f"❌ 生成提交文件失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
