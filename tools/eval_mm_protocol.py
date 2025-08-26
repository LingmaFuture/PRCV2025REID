# tools/eval_mm_protocol.py
"""
多模态人员重识别评估协议实现
基于用户提供的评估细则，适配CLIP+MER架构

评估协议要点：
- Gallery: 全部RGB图像
- Query: MM-1/2/3/4 (1/2/3/4种模态组合)
- 特征提取: CLIP+MER统一编码
- 融合: FeatureFusion或简单加权
- 评估: mAP、CMC@1/5/10
"""

import os
import json
import math
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 导入项目模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset import MultiModalDataset
from models.model import CLIPBasedMultiModalReIDModel
from configs.config import TrainingConfig


# ====== 全局配置 ======
ALL_NON_RGB = ["ir", "cpencil", "sketch", "text"]

# 模态名称映射（兼容数据集）
DATASET_TO_MODEL_MAPPING = {
    'vis': 'rgb',
    'nir': 'ir', 
    'sk': 'sketch',
    'cp': 'cpencil',
    'text': 'text'
}

def l2n(x: torch.Tensor) -> torch.Tensor:
    """L2归一化"""
    return F.normalize(x, dim=-1)

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """余弦相似度（已L2归一化的特征）"""
    # a: [Q, D], b: [G, D]
    return a @ b.T

def pick_one(items: List[dict], rng: random.Random) -> dict:
    """随机选择一个样本"""
    return rng.choice(items)

def combos(mods: List[str], k: int) -> List[Tuple[str, ...]]:
    """生成模态组合"""
    from itertools import combinations
    return list(combinations(mods, k))


# ====== 数据索引构建 ======
def build_index(dataset: MultiModalDataset) -> Dict[int, Dict[str, List[dict]]]:
    """
    构建数据索引：{pid: {modality: [sample_items]}}
    适配现有MultiModalDataset
    
    Returns:
        index[pid]["rgb"] = [{"img_path": "...", "pid": pid, "img_id": "...", "camid": 1}, ...]
        index[pid]["ir"] = [{"img_path": "...", "pid": pid, "img_id": "..."}]
        index[pid]["text"] = [{"text": "...", "pid": pid, "img_id": "对应rgb图像id(若有)"}]
    """
    print("构建数据索引...")
    index = defaultdict(lambda: defaultdict(list))
    
    for sample_idx, data_item in enumerate(dataset.data_list):
        person_id = data_item['person_id']
        person_id_str = data_item['person_id_str']
        text_desc = data_item['text_description']
        
        # 获取该person_id的所有图像
        cache = dataset.image_cache.get(person_id_str, {})
        
        # 处理各个模态的图像
        for dataset_modality, image_paths in cache.items():
            if not image_paths:  # 跳过空的模态
                continue
                
            # 映射模态名称
            model_modality = DATASET_TO_MODEL_MAPPING.get(dataset_modality, dataset_modality)
            
            for img_path in image_paths:
                # 生成img_id（从文件路径提取）
                img_id = os.path.splitext(os.path.basename(img_path))[0]
                
                sample_item = {
                    "img_path": img_path,
                    "pid": person_id,
                    "img_id": img_id,
                    "camid": None  # 当前数据集没有相机ID信息
                }
                index[person_id][model_modality].append(sample_item)
        
        # 处理文本模态
        if text_desc and isinstance(text_desc, str) and text_desc.strip():
            text_item = {
                "text": text_desc,
                "pid": person_id,
                "img_id": f"{person_id_str}_text"  # 文本的唯一ID
            }
            index[person_id]["text"].append(text_item)
    
    # 打印统计信息
    total_pids = len(index)
    modality_stats = defaultdict(int)
    for pid, mods_map in index.items():
        for mod, items in mods_map.items():
            if items:
                modality_stats[mod] += 1
    
    print(f"数据索引构建完成:")
    print(f"  总身份数: {total_pids}")
    for mod, count in modality_stats.items():
        print(f"  {mod}模态覆盖身份数: {count}")
    
    return dict(index)


# ====== 特征提取接口 ======
class FeatureExtractor:
    """特征提取器，封装CLIP+MER模型"""
    
    def __init__(self, model: CLIPBasedMultiModalReIDModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def encode_rgb(self, img_path: str) -> torch.Tensor:
        """编码RGB图像"""
        return self._encode_image(img_path, 'rgb')
    
    def encode_ir(self, img_path: str) -> torch.Tensor:
        """编码红外图像"""
        return self._encode_image(img_path, 'ir')
    
    def encode_cpencil(self, img_path: str) -> torch.Tensor:
        """编码彩铅图像"""
        return self._encode_image(img_path, 'cpencil')
    
    def encode_sketch(self, img_path: str) -> torch.Tensor:
        """编码素描图像"""
        return self._encode_image(img_path, 'sketch')
    
    def encode_text(self, text_str: str) -> torch.Tensor:
        """编码文本"""
        with torch.no_grad():
            outputs = self.model(images=None, texts=[text_str])
            features = outputs['features']  # [1, fusion_dim]
            return features.squeeze(0)  # [fusion_dim]
    
    def _encode_image(self, img_path: str, modality: str) -> torch.Tensor:
        """通用图像编码方法"""
        from PIL import Image
        import torchvision.transforms as T
        
        # 图像预处理
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            
            # 处理单通道模态（IR、Sketch）
            if modality in ['ir', 'sketch']:
                # 转换为灰度然后扩展为3通道
                image = image.convert('L').convert('RGB')
            
            # 预处理
            image_tensor = transform(image).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
            
            # 模型推理
            with torch.no_grad():
                images = {modality: image_tensor}
                outputs = self.model(images=images, texts=None)
                features = outputs['features']  # [1, fusion_dim]
                return features.squeeze(0)  # [fusion_dim]
                
        except Exception as e:
            print(f"错误：无法加载图像 {img_path}: {e}")
            # 返回零特征作为后备
            return torch.zeros(self.model.fusion_dim, device=self.device)
    
    def fuse_features_if_any(self, modal_feats: List[torch.Tensor], modalities: List[str]) -> Optional[torch.Tensor]:
        """
        特征融合（如果有融合器）
        Args:
            modal_feats: 模态特征列表
            modalities: 模态名称列表
        Returns:
            融合后特征，如果没有融合器则返回None
        """
        if len(modal_feats) <= 1:
            return modal_feats[0] if modal_feats else None
        
        # 使用模型内置的特征融合器
        try:
            with torch.no_grad():
                fused = self.model.feature_fusion(modal_feats)
                return fused
        except Exception as e:
            print(f"特征融合失败: {e}")
            return None


# ====== Query构建 ======
def build_queries(index: Dict[int, Dict[str, List[dict]]],
                  mode_k: int,
                  rng: random.Random,
                  main_mod_choice="lexi_first") -> List[dict]:
    """
    构建查询列表
    Args:
        index: 数据索引
        mode_k: 模态数量（1/2/3/4）
        rng: 随机数生成器
        main_mod_choice: 主模态选择策略
    Returns:
        查询列表，每个查询包含：{"pid": int, "modalities": tuple, "samples": dict}
    """
    queries = []
    for pid, mods_map in index.items():
        # 获取可用的非RGB模态
        avail = [m for m in ALL_NON_RGB if m in mods_map and len(mods_map[m]) > 0]
        
        # 生成所有k元组合
        for S in combos(avail, mode_k):
            S = tuple(sorted(S))
            
            # 选择主模态
            if main_mod_choice == "lexi_first":
                main_m = S[0]
            else:
                main_m = rng.choice(S)
            
            # 检查主模态样本是否存在
            if len(mods_map[main_m]) == 0:
                continue
            
            # 选择主模态样本
            samples = {main_m: pick_one(mods_map[main_m], rng)}
            
            # 为其他模态选择样本
            ok = True
            for m in S:
                if m == main_m:
                    continue
                if m not in mods_map or len(mods_map[m]) == 0:
                    ok = False
                    break
                samples[m] = pick_one(mods_map[m], rng)
            
            if ok:
                queries.append({
                    "pid": pid,
                    "modalities": S,
                    "samples": samples
                })
    
    return queries


# ====== Gallery构建 ======
def build_gallery(index: Dict[int, Dict[str, List[dict]]]) -> List[dict]:
    """构建Gallery（全部RGB图像）"""
    gallery = []
    for pid, mods_map in index.items():
        rgb_samples = mods_map.get("rgb", [])
        for sample in rgb_samples:
            gallery.append(sample)
    return gallery


# ====== 特征提取与缓存 ======
def extract_gallery_feats(gallery: List[dict], extractor: FeatureExtractor, cache_dir: str):
    """提取Gallery特征并缓存"""
    os.makedirs(cache_dir, exist_ok=True)
    feat_path = os.path.join(cache_dir, "rgb_feats.npy")
    meta_path = os.path.join(cache_dir, "rgb_meta.json")
    
    # 检查缓存
    if os.path.exists(feat_path) and os.path.exists(meta_path):
        print("从缓存加载Gallery特征...")
        feats = np.load(feat_path)
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        return torch.from_numpy(feats).float(), meta
    
    print("提取Gallery特征...")
    feats, meta = [], []
    
    for item in tqdm(gallery, desc="提取RGB特征"):
        # 提取特征
        f = extractor.encode_rgb(item["img_path"])
        f = l2n(f.float().view(1, -1)).squeeze(0)  # L2归一化
        feats.append(f.cpu().numpy())
        
        # 保存元数据
        meta.append({
            "img_id": item.get("img_id", None),
            "pid": int(item["pid"]),
            "camid": item.get("camid", None)
        })
    
    # 保存缓存
    feats = np.stack(feats, 0)
    np.save(feat_path, feats)
    json.dump(meta, open(meta_path, "w", encoding="utf-8"))
    
    return torch.from_numpy(feats).float(), meta


def extract_query_feat(q: dict, extractor: FeatureExtractor, weight_cfg: Dict[str, float]) -> torch.Tensor:
    """
    提取查询特征
    Args:
        q: 查询样本
        extractor: 特征提取器
        weight_cfg: 模态权重配置
    Returns:
        融合后的查询特征
    """
    feats, mods = [], []
    
    # 提取各模态特征
    for m, sample in q["samples"].items():
        if m == "ir":
            f = extractor.encode_ir(sample["img_path"])
        elif m == "cpencil":
            f = extractor.encode_cpencil(sample["img_path"])
        elif m == "sketch":
            f = extractor.encode_sketch(sample["img_path"])
        elif m == "text":
            f = extractor.encode_text(sample["text"])
        else:
            raise ValueError(f"未知模态: {m}")
        
        feats.append(l2n(f.float().view(1, -1)).squeeze(0))
        mods.append(m)
    
    # 尝试使用模型内置融合器
    fused = extractor.fuse_features_if_any(feats, mods)
    if fused is not None:
        return l2n(fused.view(1, -1)).squeeze(0)
    
    # 简单加权融合
    weights = torch.tensor([weight_cfg.get(m, 1.0) for m in mods], 
                          dtype=torch.float32, device=feats[0].device)
    weighted_feats = torch.stack(feats, 0) * weights[:, None]
    return l2n(weighted_feats.sum(0, keepdim=True)).squeeze(0)


# ====== 排名与评估指标 ======
def rank_and_metrics(queries: List[dict],
                     gallery_feats: torch.Tensor, 
                     gallery_meta: List[dict],
                     extractor: FeatureExtractor,
                     weight_cfg: Dict[str, float],
                     ignore_same_img=True, 
                     cross_camera=False) -> Dict[str, float]:
    """
    排名和指标计算
    Args:
        queries: 查询列表
        gallery_feats: Gallery特征 [G, D]
        gallery_meta: Gallery元数据
        extractor: 特征提取器
        weight_cfg: 模态权重配置
        ignore_same_img: 是否忽略同图像
        cross_camera: 是否跨相机（当前未使用）
    Returns:
        评估指标字典
    """
    # 预处理Gallery信息
    g_pids = torch.tensor([m["pid"] for m in gallery_meta], dtype=torch.long)
    g_imgid = [m.get("img_id", None) for m in gallery_meta]
    g_camid = [m.get("camid", None) for m in gallery_meta]
    
    APs, cmc_hits_1, cmc_hits_5, cmc_hits_10 = [], [], [], []
    
    for q in tqdm(queries, desc="评估查询"):
        # 提取查询特征
        q_feat = extract_query_feat(q, extractor, weight_cfg).view(1, -1)
        
        # 计算相似度
        sims = cosine_sim(q_feat, gallery_feats)  # [1, G]
        sims = sims.squeeze(0)  # [G]
        
        # 构造掩码：剔除无效项
        mask = torch.ones_like(sims, dtype=torch.bool)
        
        # 同图像剔除（如果启用）
        if ignore_same_img:
            # 获取查询的img_id集合
            q_imgids = set()
            for sample in q["samples"].values():
                if "img_id" in sample and sample["img_id"] is not None:
                    q_imgids.add(sample["img_id"])
            
            if q_imgids:
                for i, gid in enumerate(g_imgid):
                    if gid in q_imgids:
                        mask[i] = False
        
        # 相似度排序（保留掩码）
        sims_masked = sims.clone()
        sims_masked[~mask] = -1e9
        ranks = torch.argsort(sims_masked, descending=True)
        
        # 构建正样本集合
        pid = q["pid"]
        is_pos = (g_pids == pid) & mask
        pos_indices = torch.nonzero(is_pos).flatten().tolist()
        
        if len(pos_indices) == 0:
            # 该pid在Gallery中不存在，跳过
            continue
        
        # CMC计算
        topk = ranks[:10].tolist()
        hit1 = int(any(i in topk[:1] for i in pos_indices))
        hit5 = int(any(i in topk[:5] for i in pos_indices))
        hit10 = int(any(i in topk[:10] for i in pos_indices))
        cmc_hits_1.append(hit1)
        cmc_hits_5.append(hit5)
        cmc_hits_10.append(hit10)
        
        # AP计算
        num_pos = len(pos_indices)
        pos_set = set(pos_indices)
        hit, prec_sum = 0, 0.0
        
        for rank_idx, gidx in enumerate(ranks.tolist(), start=1):
            if gidx in pos_set:
                hit += 1
                prec_sum += hit / rank_idx
            if hit == num_pos:
                break
        
        APs.append(prec_sum / num_pos)
    
    # 计算平均指标
    mAP = float(np.mean(APs)) if APs else 0.0
    CMC1 = float(np.mean(cmc_hits_1)) if cmc_hits_1 else 0.0
    CMC5 = float(np.mean(cmc_hits_5)) if cmc_hits_5 else 0.0
    CMC10 = float(np.mean(cmc_hits_10)) if cmc_hits_10 else 0.0
    
    return {
        "mAP": mAP,
        "R@1": CMC1,
        "R@5": CMC5,
        "R@10": CMC10,
        "num_queries": len(APs)
    }


# ====== 主评估函数 ======
def run_eval(dataset_root: str, 
             model_path: str = None,
             model: CLIPBasedMultiModalReIDModel = None,
             cache_dir: str = "./eval_cache",
             seed: int = 42,
             weight_cfg: Dict[str, float] = None,
             ignore_same_img=True,
             cross_camera=False,
             device: str = "cuda"):
    """
    运行多模态评估协议
    Args:
        dataset_root: 数据集根目录
        model_path: 模型权重路径（可选）
        model: 已加载的模型（可选）
        cache_dir: 缓存目录
        seed: 随机种子
        weight_cfg: 模态权重配置
        ignore_same_img: 是否忽略同图像
        cross_camera: 是否跨相机
        device: 设备
    Returns:
        评估结果字典
    """
    # 设置随机种子
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 默认权重配置
    if weight_cfg is None:
        weight_cfg = {"ir": 1.0, "cpencil": 1.0, "sketch": 1.0, "text": 1.2}
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    if model is None:
        if model_path is None:
            raise ValueError("必须提供model_path或model参数")
        print(f"加载模型: {model_path}")
        config = TrainingConfig()
        model = CLIPBasedMultiModalReIDModel(config).to(device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 设置分类器
        if 'num_classes' in checkpoint:
            model.set_num_classes(checkpoint['num_classes'])
    
    model.eval()
    
    # 创建特征提取器
    extractor = FeatureExtractor(model, device)
    
    # 加载数据集
    print("加载数据集...")
    config = TrainingConfig()
    config.data_root = dataset_root
    dataset = MultiModalDataset(config, split='val')  # 使用验证模式，关闭增强
    
    # 构建数据索引
    index = build_index(dataset)
    
    # 构建Gallery
    gallery = build_gallery(index)
    print(f"Gallery大小: {len(gallery)} 张RGB图像")
    
    # 提取Gallery特征
    g_feats, g_meta = extract_gallery_feats(gallery, extractor, cache_dir)
    g_feats = l2n(g_feats)  # 确保L2归一化
    
    # 运行各模态组合评估
    results = {}
    print("\n开始多模态评估...")
    
    for k in [1, 2, 3, 4]:
        print(f"\n=== MM-{k}评估 ===")
        queries = build_queries(index, mode_k=k, rng=rng, main_mod_choice="lexi_first")
        print(f"查询数量: {len(queries)}")
        
        if len(queries) == 0:
            print(f"MM-{k}: 无有效查询，跳过")
            results[f"MM-{k}"] = {"mAP": 0.0, "R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "num_queries": 0}
            continue
        
        # 计算指标
        res = rank_and_metrics(
            queries, g_feats, g_meta, extractor, weight_cfg,
            ignore_same_img=ignore_same_img, cross_camera=cross_camera
        )
        results[f"MM-{k}"] = res
        
        print(f"MM-{k}: mAP={res['mAP']:.4f}, R@1={res['R@1']:.4f}, "
              f"R@5={res['R@5']:.4f}, R@10={res['R@10']:.4f}")
    
    # 计算平均指标
    valid_results = [results[f"MM-{k}"] for k in [1,2,3,4] if results[f"MM-{k}"]["num_queries"] > 0]
    if valid_results:
        avg = {
            "mAP": np.mean([r["mAP"] for r in valid_results]),
            "R@1": np.mean([r["R@1"] for r in valid_results]),
            "R@5": np.mean([r["R@5"] for r in valid_results]),
            "R@10": np.mean([r["R@10"] for r in valid_results])
        }
        results["AVG(1-4)"] = avg
    else:
        results["AVG(1-4)"] = {"mAP": 0.0, "R@1": 0.0, "R@5": 0.0, "R@10": 0.0}
    
    # 打印最终结果
    print("\n" + "="*50)
    print("最终评估结果:")
    print("="*50)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    return results


# ====== 导出提交文件（可选） ======
def export_submission_csv(queries: List[dict],
                         gallery_feats: torch.Tensor,
                         gallery_meta: List[dict], 
                         extractor: FeatureExtractor,
                         weight_cfg: Dict[str, float],
                         output_path: str,
                         top_k: int = 100):
    """
    导出Kaggle提交CSV文件
    Args:
        queries: 查询列表
        gallery_feats: Gallery特征
        gallery_meta: Gallery元数据
        extractor: 特征提取器
        weight_cfg: 模态权重配置
        output_path: 输出路径
        top_k: 取前K个结果
    """
    print(f"导出提交文件: {output_path}")
    
    submission_data = []
    
    for q in tqdm(queries, desc="生成提交"):
        # 提取查询特征
        q_feat = extract_query_feat(q, extractor, weight_cfg).view(1, -1)
        
        # 计算相似度
        sims = cosine_sim(q_feat, gallery_feats).squeeze(0)
        
        # 排序
        ranks = torch.argsort(sims, descending=True)[:top_k].tolist()
        
        # 生成query_key
        pid = q["pid"]
        mods = "+".join(sorted(q["modalities"]))
        sample_ids = []
        for sample in q["samples"].values():
            if "img_id" in sample:
                sample_ids.append(sample["img_id"])
        query_key = f"{pid}|{mods}|{'+'.join(sample_ids)}"
        
        # 获取排序后的gallery img_id
        ranked_gallery_ids = [gallery_meta[i]["img_id"] for i in ranks]
        ranked_gallery_ids_str = " ".join(str(gid) for gid in ranked_gallery_ids if gid is not None)
        
        submission_data.append({
            "query_key": query_key,
            "ranked_gallery_ids": ranked_gallery_ids_str
        })
    
    # 保存CSV
    import pandas as pd
    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    print(f"提交文件已保存: {output_path}")


if __name__ == "__main__":
    # 示例使用
    import argparse
    
    parser = argparse.ArgumentParser(description="多模态ReID评估协议")
    parser.add_argument("--dataset_root", type=str, default="./data/train", help="数据集根目录")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pth", help="模型权重路径")
    parser.add_argument("--cache_dir", type=str, default="./eval_cache", help="缓存目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--export_csv", type=str, default=None, help="导出提交CSV路径")
    
    args = parser.parse_args()
    
    # 检查环境
    if not os.path.exists(args.dataset_root):
        print(f"错误：数据集目录不存在 {args.dataset_root}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"错误：模型文件不存在 {args.model_path}")
        sys.exit(1)
    
    # 运行评估
    results = run_eval(
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        seed=args.seed,
        device=args.device
    )
    
    # 可选：导出提交文件
    if args.export_csv:
        print(f"\n导出提交文件功能需要额外实现，请参考export_submission_csv函数")
