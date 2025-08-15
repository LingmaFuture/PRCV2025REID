# dataset.py - 数据集
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import cv2

from configs.config import TrainingConfig

class ModalityAugmentation:
    """模态特定的数据增强"""
    
    def __init__(self, modality: str, config: TrainingConfig):
        self.modality = modality
        self.config = config
        
    def get_transform(self, is_training: bool = True):
        """获取模态特定的变换"""
        if is_training:
            transforms_list = [
                transforms.Resize((288, 144)),  # 稍大尺寸用于随机裁剪
                transforms.RandomHorizontalFlip(0.5) if self.config.random_flip else transforms.Lambda(lambda x: x),
            ]
            
            # 针对不同模态的特殊处理
            if self.modality in ['visible', 'painting'] and self.config.color_jitter:
                transforms_list.append(
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                         saturation=0.2, hue=0.1)
                )
            elif self.modality == 'infrared':
                # 红外图像特殊处理
                transforms_list.append(
                    transforms.ColorJitter(brightness=0.1, contrast=0.3)
                )
            elif self.modality == 'sketch':
                # 素描图像增强对比度
                transforms_list.append(
                    transforms.ColorJitter(contrast=0.3)
                )
            
            if self.config.random_crop:
                transforms_list.extend([
                    transforms.Pad(10, padding_mode='reflect'),
                    transforms.RandomCrop((256, 128))
                ])
            else:
                transforms_list.append(transforms.Resize((256, 128)))
                
        else:
            transforms_list = [transforms.Resize((256, 128))]
        
        # 通用变换
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        if is_training and self.config.random_erase > 0:
            transforms_list.append(
                transforms.RandomErasing(p=self.config.random_erase, scale=(0.02, 0.4))
            )
        
        return transforms.Compose(transforms_list)


class BalancedBatchSampler(Sampler):
    """平衡批次采样器，确保每个批次包含多个相同ID的样本"""
    
    def __init__(self, dataset, batch_size, num_instances=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        
        # 按ID分组样本索引
        self.index_pid = defaultdict(list)
        for idx, data in enumerate(dataset.data_list):
            self.index_pid[data['person_id']].append(idx)
        
        self.pids = list(self.index_pid.keys())
        
        # 确保所有ID都有足够的样本
        self.pids = [pid for pid in self.pids if len(self.index_pid[pid]) >= 2]
        
        # 计算epoch长度
        self.length = len(self.pids) // self.num_pids_per_batch * self.batch_size
    
    def __iter__(self):
        """生成批次"""
        random.shuffle(self.pids)
        
        for start_idx in range(0, len(self.pids), self.num_pids_per_batch):
            batch_indices = []
            end_idx = min(start_idx + self.num_pids_per_batch, len(self.pids))
            selected_pids = self.pids[start_idx:end_idx]
            
            for pid in selected_pids:
                indices = self.index_pid[pid]
                if len(indices) >= self.num_instances:
                    selected_indices = random.sample(indices, self.num_instances)
                else:
                    selected_indices = random.choices(indices, k=self.num_instances)
                
                batch_indices.extend(selected_indices)
            
            # 确保批次大小一致
            while len(batch_indices) < self.batch_size:
                pid = random.choice(self.pids)
                indices = self.index_pid[pid]
                batch_indices.append(random.choice(indices))
            
            yield batch_indices[:self.batch_size]
    
    def __len__(self):
        return self.length // self.batch_size


class ImprovedMultiModalDataset(Dataset):
    """改进的多模态数据集"""
    
    def __init__(self, 
                 config: TrainingConfig,
                 split: str = 'train',
                 modality_dropout: float = None):
        
        self.config = config
        self.split = split
        self.modality_dropout = modality_dropout or config.modality_dropout
        
        # 模态文件夹映射
        self.modality_folders = {
            'vis': 'vis',  # 修改为实际目录名
            'nir': 'nir',  # 修改为实际目录名
            'sk': 'sk',    # 修改为实际目录名
            'cp': 'cp'     # 修改为实际目录名
        }
        
        # 加载数据
        self._load_annotations()
        self._prepare_data_split()
        
        # 初始化变换
        self.transforms = {
            modality: ModalityAugmentation(modality, config).get_transform(
                is_training=(split == 'train')
            ) for modality in self.modality_folders.keys()
        }
        
        # 预加载图像路径以提高效率
        self._cache_image_paths()
        
        print(f"{split} dataset: {len(self.data_list)} samples from {len(self.person_ids)} identities")
    
    def _load_annotations(self):
        """加载标注文件并转换为字典格式"""
        with open(self.config.json_file, 'r', encoding='utf-8') as f:
            annotations_list = json.load(f)
        
        # 将列表格式转换为字典格式，以person_id为键
        self.annotations = {}
        for item in annotations_list:
            person_id = item['id']
            person_id_str = f"{person_id:04d}"
            
            if person_id_str not in self.annotations:
                self.annotations[person_id_str] = {
                    'description': item['caption'],
                    'samples': []
                }
            
            # 添加样本信息
            self.annotations[person_id_str]['samples'].append({
                'file_path': item['file_path'],
                'caption': item['caption'],
                'split': item['split']
            })
        
        print(f"Loaded annotations for {len(self.annotations)} identities")
    
    def _prepare_data_split(self):
        """准备数据分割"""
        all_person_ids = list(range(1, 1000))  # 0001-0999
        random.seed(42)  # 固定随机种子确保可复现
        random.shuffle(all_person_ids)
        
        train_end = int(len(all_person_ids) * self.config.train_split)
        val_end = train_end + int(len(all_person_ids) * self.config.val_split)
        
        if self.split == 'train':
            self.person_ids = all_person_ids[:train_end]
        elif self.split == 'val':
            self.person_ids = all_person_ids[train_end:val_end]
        else:  # test
            self.person_ids = all_person_ids[val_end:]
        
        # 构建数据列表
        self.data_list = []
        for person_id in self.person_ids:
            person_id_str = f"{person_id:04d}"
            # 获取该ID的文本描述，如果没有则使用空字符串
            text_desc = self.annotations.get(person_id_str, {}).get('description', '')
            
            # 可以为每个ID创建多个样本（如果有多张图片）
            self.data_list.append({
                'person_id': person_id,
                'person_id_str': person_id_str,
                'text_description': text_desc
            })
    
    def _cache_image_paths(self):
        """缓存图像路径"""
        self.image_cache = {}
        missing_modalities = defaultdict(int)
        
        for data in self.data_list:
            person_id_str = data['person_id_str']
            self.image_cache[person_id_str] = {}
            
            for modality, folder in self.modality_folders.items():
                person_folder = os.path.join(self.config.data_root, folder, person_id_str)
                
                if os.path.exists(person_folder):
                    images = [f for f in os.listdir(person_folder) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    if images:
                        self.image_cache[person_id_str][modality] = [
                            os.path.join(person_folder, img) for img in images
                        ]
                    else:
                        self.image_cache[person_id_str][modality] = []
                        missing_modalities[modality] += 1
                else:
                    self.image_cache[person_id_str][modality] = []
                    missing_modalities[modality] += 1
        
        # 打印缺失统计
        for modality, count in missing_modalities.items():
            print(f"Missing {modality} for {count}/{len(self.data_list)} samples")
    
    def _get_available_modalities(self, person_id_str):
        """获取该ID可用的模态"""
        available = []
        for modality in self.modality_folders.keys():
            if self.image_cache[person_id_str][modality]:
                available.append(modality)
        return available
    
    def _get_random_modality_subset(self, available_modalities):
        """随机选择模态子集（模态dropout）"""
        if self.split == 'train' and random.random() < self.modality_dropout:
            # 训练时随机dropout一些模态
            num_modalities = random.randint(
                max(self.config.min_modalities, 1), 
                len(available_modalities)
            )
            selected_modalities = random.sample(available_modalities, num_modalities)
        else:
            selected_modalities = available_modalities
        
        return selected_modalities
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        person_id = data['person_id']
        person_id_str = data['person_id_str']
        text_desc = data['text_description']
        
        # 获取可用模态
        available_modalities = self._get_available_modalities(person_id_str)
        selected_modalities = self._get_random_modality_subset(available_modalities)
        
        # 准备图像数据
        images = {}
        modality_mask = {}
        
        for modality in self.modality_folders.keys():
            if modality in selected_modalities and self.image_cache[person_id_str][modality]:
                # 随机选择一张图像
                image_paths = self.image_cache[person_id_str][modality]
                selected_path = random.choice(image_paths)
                
                try:
                    image = Image.open(selected_path).convert('RGB')
                    images[modality] = self.transforms[modality](image)
                    modality_mask[modality] = 1.0
                except Exception as e:
                    print(f"Error loading image {selected_path}: {e}")
                    images[modality] = torch.zeros(3, 256, 128)
                    modality_mask[modality] = 0.0
            else:
                images[modality] = torch.zeros(3, 256, 128)
                modality_mask[modality] = 0.0
        
        return {
            'person_id': torch.tensor(person_id - 1, dtype=torch.long),  # 0-based indexing
            'images': images,
            'text_description': text_desc,
            'modality_mask': modality_mask,
            'selected_modalities': selected_modalities
        }
