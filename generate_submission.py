# tools/generate_submission.py - 生成比赛提交文件的推理脚本
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List, Dict, Union, Tuple

from models.advanced_model import AdvancedMultiModalReIDModel
from configs.config import TrainingConfig
from datasets.dataset import compatible_collate_fn


class ValQueryDataset(Dataset):
    """验证集查询数据集"""
    
    def __init__(self, val_root: str, queries_json: str, transform=None):
        """
        初始化验证集查询数据集
        
        Args:
            val_root: 验证集根目录
            queries_json: 查询标注文件路径
            transform: 图像预处理函数
        """
        self.val_root = val_root
        self.transform = transform
        
        # 加载查询标注
        with open(queries_json, 'r', encoding='utf-8') as f:
            self.queries = json.load(f)
            
        # 初始化文本编码器
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.text_encoder.eval()
        
        # 缓存所有唯一的查询类型
        self.query_types = sorted(list(set(q['query_type'] for q in self.queries)))
        
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """获取单个查询样本"""
        query = self.queries[idx]
        query_type = query['query_type']
        contents = query['content']
        
        # 初始化返回字典
        sample = {
            'query_idx': query['query_idx'],
            'query_type': query_type,
            'modalities': {}
        }
        
        # 处理不同模态的内容
        for content in contents:
            if isinstance(content, str):
                if content.endswith('.jpg'):  # 图像文件
                    modality = content.split('/')[0]  # nir, cp, sk
                    img_path = os.path.join(self.val_root, content)
                    img = Image.open(img_path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    sample['modalities'][modality] = img
                else:  # 文本描述
                    # 编码文本
                    tokens = self.tokenizer(content, 
                                         padding='max_length',
                                         truncation=True,
                                         max_length=128,
                                         return_tensors='pt')
                    with torch.no_grad():
                        text_features = self.text_encoder(**tokens).last_hidden_state.mean(dim=1)
                    sample['modalities']['text'] = text_features.squeeze(0)
        
        return sample


class GalleryDataset(Dataset):
    """验证集Gallery数据集"""
    
    def __init__(self, gallery_root: str, transform=None):
        """
        初始化Gallery数据集
        
        Args:
            gallery_root: gallery图像目录路径
            transform: 图像预处理函数
        """
        self.gallery_root = gallery_root
        self.transform = transform
        
        # 获取所有gallery图像路径
        self.image_paths = []
        self.image_ids = []
        for img_name in sorted(os.listdir(gallery_root)):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(gallery_root, img_name))
                self.image_ids.append(int(img_name.split('.')[0]))
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        """获取单个gallery样本"""
        img_path = self.image_paths[idx]
        img_id = self.image_ids[idx]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return {
            'image': img,
            'image_id': img_id
        }


def extract_gallery_features(
    model: torch.nn.Module,
    gallery_loader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, List[int]]:
    """
    提取gallery特征
    
    Args:
        model: ReID模型
        gallery_loader: gallery数据加载器
        device: 计算设备
    
    Returns:
        gallery_features: gallery特征矩阵
        gallery_ids: gallery图像ID列表
    """
    model.eval()
    gallery_features = []
    gallery_ids = []
    
    with torch.no_grad():
        for batch in tqdm(gallery_loader, desc='提取Gallery特征'):
            images = batch['image'].to(device)
            ids = batch['image_id']
            
            # 提取特征
            features = model.extract_features(images)
            features = F.normalize(features, p=2, dim=1)
            
            gallery_features.append(features.cpu())
            gallery_ids.extend(ids)
    
    gallery_features = torch.cat(gallery_features, dim=0)
    return gallery_features, gallery_ids


def generate_rankings(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_ids: List[int],
    top_k: int = 100
) -> List[List[int]]:
    """
    生成排序结果
    
    Args:
        query_features: 查询特征矩阵
        gallery_features: gallery特征矩阵
        gallery_ids: gallery图像ID列表
        top_k: 返回的最相似图像数量
    
    Returns:
        rankings: 每个查询的top-k排序结果
    """
    # 计算相似度矩阵
    similarity = torch.mm(query_features, gallery_features.t())
    
    # 获取top-k索引
    _, indices = torch.topk(similarity, k=top_k, dim=1)
    
    # 转换为图像ID
    rankings = []
    for idx_list in indices:
        ranking = [gallery_ids[idx] for idx in idx_list.tolist()]
        rankings.append(ranking)
    
    return rankings


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 加载配置
    config = TrainingConfig()
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')
    
    # 创建数据集
    val_root = 'data/val'
    queries_json = os.path.join(val_root, 'val_queries.json')
    gallery_root = os.path.join(val_root, 'gallery')
    
    # 创建数据加载器
    val_dataset = ValQueryDataset(val_root, queries_json, transform=config.val_transform)
    gallery_dataset = GalleryDataset(gallery_root, transform=config.val_transform)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.inference_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=compatible_collate_fn
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config.inference_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 加载模型
    model = AdvancedMultiModalReIDModel(config).to(device)
    checkpoint = torch.load(config.best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 提取gallery特征
    gallery_features, gallery_ids = extract_gallery_features(model, gallery_loader, device)
    
    # 处理每个查询
    results = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='处理查询'):
            query_idx = batch['query_idx']
            query_type = batch['query_type']
            
            # 提取查询特征
            query_features = model.extract_query_features(batch)
            query_features = F.normalize(query_features, p=2, dim=1)
            
            # 生成排序结果
            rankings = generate_rankings(query_features, gallery_features, gallery_ids)
            
            # 保存结果
            for idx, ranking in zip(query_idx, rankings):
                results.append({
                    'query_idx': idx,
                    'query_type': query_type,
                    'ranking_list_idx': ranking
                })
    
    # 保存为CSV文件
    df = pd.DataFrame(results)
    # 将ranking_list_idx转换为字符串格式
    df['ranking_list_idx'] = df['ranking_list_idx'].apply(lambda x: str(x))
    output_path = 'submission.csv'
    df.to_csv(output_path, index=False)
    logging.info(f'已保存提交文件到: {output_path}')


if __name__ == '__main__':
    main()
