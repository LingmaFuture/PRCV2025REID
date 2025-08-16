#!/usr/bin/env python3
"""
PRCV2025全模态行人重识别竞赛提交文件生成器

生成符合比赛要求的CSV提交文件，支持：
- 单模态查询: onemodal_NIR, onemodal_SK, onemodal_CP, onemodal_TEXT
- 双模态查询: twomodal_*
- 三模态查询: threemodal_*  
- 四模态查询: fourmodal_*
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import ast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from models.model import MultiModalReIDModel
from datasets.dataset import MultiModalDataset, compatible_collate_fn
from configs.config import TrainingConfig


class PRCV2025SubmissionGenerator:
    """PRCV2025竞赛提交文件生成器"""
    
    def __init__(self, config_path: str = None, model_path: str = None):
        """
        初始化提交文件生成器
        
        Args:
            config_path: 配置文件路径
            model_path: 模型权重路径
        """
        # 加载配置
        self.config = TrainingConfig() if config_path is None else self._load_config(config_path)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 验证集路径设置
        self.val_data_root = "./data/val"
        self.gallery_data_root = "./data/val/gallery"  # 画廊数据(可见光图像)
        self.query_file = "./data/val/val_queries.csv"
        
        # 输出路径
        self.output_dir = "./submissions"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _load_config(self, config_path: str):
        """加载配置文件"""
        # 这里可以扩展为从文件加载配置
        return TrainingConfig()
    
    def _load_model(self, model_path: str = None) -> MultiModalReIDModel:
        """加载训练好的模型"""
        if model_path is None:
            model_path = self.config.best_model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 创建模型
        model = MultiModalReIDModel(self.config).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"加载模型权重: {model_path}")
            print(f"训练轮次: {checkpoint.get('epoch', 'Unknown')}")
            print(f"最佳mAP: {checkpoint.get('best_map', 'Unknown')}")
        else:
            state_dict = checkpoint
        
        # 处理类别数量不匹配的问题
        model_state_dict = model.state_dict()
        
        # 检查分类器权重尺寸是否匹配
        if 'classifier.weight' in state_dict and 'classifier.weight' in model_state_dict:
            checkpoint_num_classes = state_dict['classifier.weight'].size(0)
            current_num_classes = model_state_dict['classifier.weight'].size(0)
            
            if checkpoint_num_classes != current_num_classes:
                print(f"警告: 检查点中的类别数量 ({checkpoint_num_classes}) 与当前配置 ({current_num_classes}) 不匹配")
                print("将跳过分类器权重的加载，使用随机初始化的分类器")
                
                # 移除分类器权重，只加载其他层
                state_dict.pop('classifier.weight', None)
                state_dict.pop('classifier.bias', None)
        
        # 加载兼容的权重
        try:
            model.load_state_dict(state_dict, strict=False)
            print("模型权重加载成功")
        except Exception as e:
            print(f"警告: 部分权重加载失败: {e}")
            print("将使用部分加载的权重继续")
        
        model.eval()
        return model
    
    def _load_query_list(self) -> pd.DataFrame:
        """加载查询列表"""
        if not os.path.exists(self.query_file):
            raise FileNotFoundError(f"查询文件不存在: {self.query_file}")
        
        queries_df = pd.read_csv(self.query_file)
        print(f"加载查询数量: {len(queries_df)}")
        
        # 统计各类查询数量
        query_types = queries_df['query_type'].value_counts()
        print("查询类型分布:")
        for qtype, count in query_types.items():
            print(f"  {qtype}: {count}")
        
        return queries_df
    
    def _parse_query_content(self, content_str: str) -> List[str]:
        """解析查询内容字符串"""
        try:
            # 使用ast.literal_eval安全解析列表字符串
            content_list = ast.literal_eval(content_str)
            return content_list if isinstance(content_list, list) else [content_list]
        except Exception as e:
            print(f"解析查询内容失败: {content_str}, 错误: {e}")
            return []
    
    def _extract_gallery_features(self) -> Tuple[torch.Tensor, List[str]]:
        """提取画廊特征(可见光模态)"""
        print("提取画廊特征...")
        
        gallery_path = Path(self.gallery_data_root)
        if not gallery_path.exists():
            raise FileNotFoundError(f"画廊目录不存在: {gallery_path}")
        
        # 收集所有画廊图像，按文件名数字排序
        gallery_images = []
        for img_path in gallery_path.glob("*.jpg"):
            gallery_images.append(str(img_path))
        
        # 按文件名中的数字排序（确保1.jpg, 2.jpg, ..., 10.jpg的正确顺序）
        def extract_number(path):
            filename = Path(path).stem  # 获取不带扩展名的文件名
            try:
                return int(filename)
            except ValueError:
                return float('inf')  # 非数字文件名排到最后
                
        gallery_images.sort(key=extract_number)
        print(f"画廊图像数量: {len(gallery_images)}")
        print(f"前5个画廊图像: {[Path(p).name for p in gallery_images[:5]]}")
        
        # 创建画廊数据集
        gallery_dataset = GalleryDataset(gallery_images, self.config)
        gallery_loader = DataLoader(
            gallery_dataset,
            batch_size=self.config.inference_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=compatible_collate_fn
        )
        
        # 提取特征
        gallery_features = []
        
        with torch.no_grad():
             for batch in tqdm(gallery_loader, desc="提取画廊特征"):
                 batch = self._move_to_device(batch)
                 
                 with autocast('cuda', enabled=self.device.type == 'cuda'):
                     features = self.model(batch, return_features=True)
                 
                 gallery_features.append(features.cpu())
        
        gallery_features = torch.cat(gallery_features, dim=0)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)
        
        print(f"画廊特征形状: {gallery_features.shape}")
        return gallery_features, gallery_images
    
    def _extract_query_features(self, queries_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """提取所有查询特征"""
        print("提取查询特征...")
        
        query_features = {}
        
        # 按查询类型分组处理
        for query_type, group in queries_df.groupby('query_type'):
            print(f"处理查询类型: {query_type} ({len(group)} 个查询)")
            
            # 创建查询数据集
            query_dataset = QueryDataset(group, self.val_data_root, self.config, query_type)
            query_loader = DataLoader(
                query_dataset,
                batch_size=self.config.inference_batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
                collate_fn=compatible_collate_fn
            )
            
            # 提取特征
            type_features = []
            
            with torch.no_grad():
                for batch in tqdm(query_loader, desc=f"提取{query_type}特征"):
                    batch = self._move_to_device(batch)
                    
                    with autocast('cuda', enabled=self.device.type == 'cuda'):
                        features = self.model(batch, return_features=True)
                    
                    type_features.append(features.cpu())
            
            if type_features:
                type_features = torch.cat(type_features, dim=0)
                type_features = F.normalize(type_features, p=2, dim=1)
                query_features[query_type] = type_features
                
                print(f"{query_type} 特征形状: {type_features.shape}")
        
        return query_features
    
    def _compute_similarities_and_rankings(self, 
                                         query_features: Dict[str, torch.Tensor],
                                         gallery_features: torch.Tensor,
                                         gallery_images: List[str],
                                         queries_df: pd.DataFrame,
                                         top_k: int = 100) -> List[Dict]:
        """计算相似度并生成排序列表"""
        print("计算相似度并生成排序...")
        
        results = []
        
        # 按查询类型处理
        for query_type, group in queries_df.groupby('query_type'):
            if query_type not in query_features:
                print(f"警告: 未找到 {query_type} 的特征")
                continue
            
            qf = query_features[query_type]
            
            # 计算相似度矩阵
            similarity = torch.mm(qf, gallery_features.t())  # (Q, G)
            
            # 获取top-k排序
            _, indices = torch.topk(similarity, top_k, dim=1, largest=True)
            
            # 转换为实际的图像ID（文件名中的数字）
            # 由于画廊图像文件名是1.jpg, 2.jpg, ..., N.jpg，而数组索引是0, 1, ..., N-1
            # 我们需要将数组索引转换为对应的文件名ID
            rankings = []
            for query_indices in indices:
                query_ranking = []
                for idx in query_indices:
                    # 从数组索引转换为文件名ID
                    img_path = gallery_images[idx.item()]
                    img_id = int(Path(img_path).stem)  # 获取文件名中的数字
                    query_ranking.append(img_id)
                rankings.append(query_ranking)
            
            # 为每个查询生成结果
            for i, (_, query_row) in enumerate(group.iterrows()):
                result = {
                    'query_idx': query_row['query_idx'],
                    'query_type': query_type,
                    'ranking_list_idx': str(rankings[i])  # 转为字符串格式
                }
                results.append(result)
        
        print(f"生成排序结果: {len(results)} 个查询")
        return results
    
    def _move_to_device(self, batch):
        """将批次数据移动到设备"""
        if isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [self._move_to_device(x) for x in batch]
        elif torch.is_tensor(batch):
            return batch.to(self.device)
        else:
            return batch
    
    def generate_submission(self, output_filename: str = "submission.csv") -> str:
        """生成提交文件"""
        print("=" * 60)
        print("PRCV2025 多模态人员重识别 - 提交文件生成")
        print("=" * 60)
        
        # 1. 加载查询列表
        queries_df = self._load_query_list()
        
        # 2. 提取画廊特征
        gallery_features, gallery_images = self._extract_gallery_features()
        
        # 3. 提取查询特征
        query_features = self._extract_query_features(queries_df)
        
        # 4. 计算相似度和排序
        results = self._compute_similarities_and_rankings(
            query_features, gallery_features, gallery_images, queries_df
        )
        
        # 5. 生成CSV文件
        output_path = os.path.join(self.output_dir, output_filename)
        results_df = pd.DataFrame(results)
        
        # 确保列顺序正确
        results_df = results_df[['query_idx', 'query_type', 'ranking_list_idx']]
        
        # 按query_idx排序
        results_df = results_df.sort_values('query_idx').reset_index(drop=True)
        
        # 保存文件
        results_df.to_csv(output_path, index=False)
        
        print(f"\n提交文件已生成: {output_path}")
        print(f"总查询数量: {len(results_df)}")
        print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        # 验证文件格式
        self._validate_submission_format(output_path)
        
        return output_path
    
    def _validate_submission_format(self, submission_path: str):
        """验证提交文件格式"""
        print("\n验证提交文件格式...")
        
        try:
            df = pd.read_csv(submission_path)
            
            # 检查列名
            expected_columns = ['query_idx', 'query_type', 'ranking_list_idx']
            if list(df.columns) != expected_columns:
                print(f"警告: 列名不匹配. 期望: {expected_columns}, 实际: {list(df.columns)}")
            
            # 检查数据类型
            print(f"query_idx 范围: {df['query_idx'].min()} - {df['query_idx'].max()}")
            print(f"查询类型: {df['query_type'].nunique()} 种")
            
            # 检查ranking_list_idx格式
            sample_ranking = df['ranking_list_idx'].iloc[0]
            try:
                ranking_list = ast.literal_eval(sample_ranking)
                print(f"排序列表长度: {len(ranking_list)}")
                print(f"排序范围: {min(ranking_list)} - {max(ranking_list)}")
            except:
                print("警告: ranking_list_idx 格式可能有问题")
            
            print("✅ 文件格式验证完成")
            
        except Exception as e:
            print(f"❌ 文件格式验证失败: {e}")


class GalleryDataset(torch.utils.data.Dataset):
    """画廊数据集(仅可见光模态)"""
    
    def __init__(self, image_paths: List[str], config):
        self.image_paths = image_paths
        self.config = config
        
        # 使用验证时的数据变换
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            print(f"加载图像失败: {image_path}, 错误: {e}")
            image_tensor = torch.zeros(3, self.config.image_size, self.config.image_size)
        
        return {
            'person_id': torch.tensor(0, dtype=torch.long),  # 占位符
            'images': {'vis': image_tensor},
            'text_description': [""],
            'modality_mask': {
                'vis': 1.0, 'nir': 0.0, 'sk': 0.0, 'cp': 0.0, 'text': 0.0
            }
        }


class QueryDataset(torch.utils.data.Dataset):
    """查询数据集(支持多模态)"""
    
    def __init__(self, queries_df: pd.DataFrame, data_root: str, config, query_type: str):
        self.queries_df = queries_df.reset_index(drop=True)
        self.data_root = data_root
        self.config = config
        self.query_type = query_type
        
        # 解析查询内容
        self.query_contents = []
        for _, row in self.queries_df.iterrows():
            content = self._parse_query_content(row['content'])
            self.query_contents.append(content)
        
        # 数据变换
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _parse_query_content(self, content_str: str) -> List[str]:
        """解析查询内容"""
        try:
            return ast.literal_eval(content_str)
        except:
            return []
    
    def __len__(self):
        return len(self.queries_df)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        content = self.query_contents[idx]
        
        # 初始化模态数据
        images = {}
        modality_mask = {'vis': 0.0, 'nir': 0.0, 'sk': 0.0, 'cp': 0.0, 'text': 0.0}
        text_description = ""
        
        # 处理图像模态
        for item in content:
            if item.endswith('.jpg') or item.endswith('.png'):
                # 确定模态类型
                if item.startswith('nir/'):
                    modality = 'nir'
                elif item.startswith('sk/'):
                    modality = 'sk'
                elif item.startswith('cp/'):
                    modality = 'cp'
                elif item.startswith('vis/'):
                    modality = 'vis'
                else:
                    continue
                
                # 加载图像
                image_path = os.path.join(self.data_root, item)
                try:
                    image = Image.open(image_path).convert('RGB')
                    images[modality] = self.transform(image)
                    modality_mask[modality] = 1.0
                except Exception as e:
                    print(f"加载图像失败: {image_path}, 错误: {e}")
                    images[modality] = torch.zeros(3, self.config.image_size, self.config.image_size)
                    modality_mask[modality] = 0.0
            else:
                # 文本描述
                text_description = item
                modality_mask['text'] = 1.0
        
        # 填充缺失的模态
        for modality in ['vis', 'nir', 'sk', 'cp']:
            if modality not in images:
                images[modality] = torch.zeros(3, self.config.image_size, self.config.image_size)
        
        return {
            'person_id': torch.tensor(0, dtype=torch.long),  # 占位符
            'images': images,
            'text_description': [text_description],
            'modality_mask': modality_mask
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PRCV2025 提交文件生成器')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型权重路径 (默认使用配置中的best_model_path)')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='输出文件名')
    parser.add_argument('--config_path', type=str, default=None,
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = PRCV2025SubmissionGenerator(
        config_path=args.config_path,
        model_path=args.model_path
    )
    
    # 生成提交文件
    output_path = generator.generate_submission(args.output)
    
    print(f"\n🎉 提交文件生成完成: {output_path}")
    print("现在可以将此文件提交到比赛平台!")


if __name__ == "__main__":
    main()