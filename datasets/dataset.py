# dataset.py
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import random
from collections import defaultdict

class ModalityAugmentation:
    """修复后的数据增强"""
    
    def __init__(self, config, is_training=True):
        self.config = config
        self.is_training = is_training
    
    def get_transform(self):
        if self.is_training:
            return transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2) if self.config.color_jitter else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=self.config.random_erase, scale=(0.02, 0.2)) if self.config.random_erase > 0 else transforms.Lambda(lambda x: x)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

class MultiModalDataset(Dataset):
    """修复后的多模态数据集"""
    
    def __init__(self, config, split='train', person_ids=None):
        self.config = config
        self.split = split
        self.is_training = (split == 'train')
        self.modality_folders = ['vis', 'nir', 'sk', 'cp']
        
        # 加载标注
        self._load_annotations()
        
        # 设置person_ids
        if person_ids is not None:
            self.person_ids = person_ids
        else:
            self.person_ids = self._get_available_person_ids()
        
        self.pid2label = {pid: i for i, pid in enumerate(self.person_ids)}
        # 构建数据列表
        self._build_data_list()
        
        # 数据变换
        self.transform = ModalityAugmentation(config, split == 'train').get_transform()
        
        # 缓存图像路径
        self._cache_image_paths()
        
        if hasattr(self, '_suppress_print') and self._suppress_print:
            pass  # 不打印，避免重复信息
        else:
            print(f"{split} dataset: {len(self.data_list)} samples, {len(self.person_ids)} identities")
    
    def _load_annotations(self):
        """加载标注文件 - 使用file_path中的目录ID作为身份ID"""
        with open(self.config.json_file, 'r', encoding='utf-8') as f:
            annotations_list = json.load(f)
        
        # 按file_path中的目录ID分组文本标注，同时记录文件路径
        self.id_to_annotations = {}
        self.id_to_files = {}
        
        for item in annotations_list:
            file_path = item['file_path']
            caption = item['caption']
            
            # 从file_path中提取目录ID作为身份ID
            parts = file_path.split('/')
            if len(parts) >= 3:
                dir_name = parts[1]  # 0001
                if dir_name.isdigit():
                    person_id = int(dir_name)  # 使用目录ID作为身份ID
                else:
                    continue  # 跳过无效路径
            else:
                continue  # 跳过无效路径
            
            if person_id not in self.id_to_annotations:
                self.id_to_annotations[person_id] = []
                self.id_to_files[person_id] = []
            
            self.id_to_annotations[person_id].append(caption)
            # 移除vis/前缀，匹配实际目录结构
            actual_path = file_path.replace('vis/', '')
            self.id_to_files[person_id].append(actual_path)
        
        # 为每个身份创建合并的文本描述
        self.annotations = {}
        for person_id, captions in self.id_to_annotations.items():
            person_id_str = f"{person_id:04d}"
            # 合并该身份的所有文本描述
            combined_caption = ' '.join(captions)
            self.annotations[person_id_str] = combined_caption
    
    def _get_available_person_ids(self):
        """获取可用的person_ids - 基于json标注"""
        # 使用json中的身份ID作为主要依据
        json_ids = set(self.id_to_annotations.keys())
        
        # 检查这些身份在vis模态中是否有对应的目录（作为gallery）
        final_ids = []
        missing_vis = []
        for pid in json_ids:
            # 检查vis模态是否存在（作为gallery目标）
            vis_path = os.path.join(self.config.data_root, 'vis', f"{pid:04d}")
            if os.path.exists(vis_path):
                final_ids.append(pid)
            else:
                missing_vis.append(pid)
        
        print(f"数据集加载信息:")
        print(f"  json中的身份ID: {len(json_ids)}")
        print(f"  最终可用的身份ID: {len(final_ids)}")
        print(f"  缺少vis目录的身份ID: {len(missing_vis)}")
        
        if missing_vis:
            print(f"  缺少vis目录的示例身份ID: {missing_vis[:10]}")
        
        return sorted(final_ids)
    
    def _build_data_list(self):
        """构建数据列表 - 修复：为每个图片文件创建独立样本"""
        self.data_list = []
        for person_id in self.person_ids:
            person_id_str = f"{person_id:04d}"
            text_desc = self.annotations.get(person_id_str, "unknown person")
            
            # 检查该身份是否有图片文件
            if person_id in self.id_to_files:
                # 为每个图片文件创建一个样本
                for file_path in self.id_to_files[person_id]:
                    self.data_list.append({
                        'person_id': person_id,
                        'person_id_str': person_id_str,
                        'text_description': text_desc,
                        'file_path': file_path  # 添加具体文件路径
                    })
            else:
                # 没有图片文件的身份，创建一个基本样本
                self.data_list.append({
                    'person_id': person_id,
                    'person_id_str': person_id_str,
                    'text_description': text_desc,
                    'file_path': None
                })
    
    def _cache_image_paths(self):
        """缓存图像路径 - 基于json文件路径"""
        self.image_cache = {}
        
        for data in self.data_list:
            person_id = data['person_id']
            person_id_str = data['person_id_str']
            self.image_cache[person_id_str] = {}
            
            # 对于vis模态，使用json中的文件路径
            if person_id in self.id_to_files:
                vis_files = self.id_to_files[person_id]
                self.image_cache[person_id_str]['vis'] = []
                
                for file_path in vis_files:
                    # 移除vis/前缀，构建完整路径
                    actual_path = file_path.replace('vis/', '')
                    full_path = os.path.join(self.config.data_root, 'vis', actual_path)
                    if os.path.exists(full_path):
                        self.image_cache[person_id_str]['vis'].append(full_path)
            
            # 对于其他模态，检查目录是否存在
            for modality in ['nir', 'sk', 'cp']:
                folder_path = os.path.join(self.config.data_root, modality, person_id_str)
                
                if os.path.exists(folder_path):
                    images = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    self.image_cache[person_id_str][modality] = [
                        os.path.join(folder_path, img) for img in images
                    ]
                else:
                    self.image_cache[person_id_str][modality] = []
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        person_id = data['person_id']
        person_id_str = data['person_id_str']
        text_desc = data['text_description']

        # 创建样本字典（统一为 images 字典 + text_description）
        sample = {
            'person_id': torch.tensor(self.pid2label[person_id], dtype=torch.long),  # 转换为0-based
            'text_description': [text_desc],
            'modality_mask': {},
        }

        images = {}
        # 加载图像 - 优先使用指定的file_path，否则随机选择
        file_path = data.get('file_path', None)
        
        for modality in self.modality_folders:
            image_paths = self.image_cache[person_id_str][modality]

            use_drop = self.is_training and (self.config.modality_dropout > 0)
            if image_paths and (not use_drop or random.random() > self.config.modality_dropout):
                # 如果有指定文件路径，尝试使用对应的模态图片
                if file_path and modality == 'vis':
                    # 对于vis模态，尝试使用指定的file_path
                    full_path = os.path.join(self.config.data_root, file_path)
                    if os.path.exists(full_path):
                        selected_path = full_path
                    else:
                        selected_path = random.choice(image_paths)
                else:
                    # 其他模态随机选择
                    selected_path = random.choice(image_paths)
                    
                try:
                    image = Image.open(selected_path).convert('RGB')
                    images[modality] = self.transform(image)
                    sample['modality_mask'][modality] = 1.0
                except Exception as e:
                    print(f"Error loading {selected_path}: {e}")
                    images[modality] = torch.zeros(3, self.config.image_size, self.config.image_size)
                    sample['modality_mask'][modality] = 0.0
            else:
                images[modality] = torch.zeros(3, self.config.image_size, self.config.image_size)
                sample['modality_mask'][modality] = 0.0


        sample['images'] = images
        return sample

class ModalityAwareBatchSampler(Sampler):
    """模态感知批次采样器 - 确保每个批次包含多样化的模态组合"""
    
    def __init__(self, dataset, batch_size, num_instances=4, min_modality_combinations=3):
        """
        Args:
            dataset: 数据集
            batch_size: 批次大小
            num_instances: 每个身份的样本数
            min_modality_combinations: 每个批次最少包含的模态组合数
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.min_modality_combinations = min_modality_combinations
        
        # 处理Subset
        if hasattr(dataset, 'dataset'):
            self.base_dataset = dataset.dataset
            self.indices = dataset.indices
        else:
            self.base_dataset = dataset
            self.indices = list(range(len(dataset)))
        
        # 先计算批次参数
        self.num_pids_per_batch = max(1, self.batch_size // num_instances)
        
        # 分析每个样本的模态情况
        self._analyze_modality_coverage()
        
        # 按身份和模态组合分组
        self._group_by_identity_and_modality()
        
    def _analyze_modality_coverage(self):
        """分析数据集的模态覆盖情况"""
        self.sample_modalities = {}  # {idx: set of available modalities}
        
        for subset_idx, orig_idx in enumerate(self.indices):
            if hasattr(self.base_dataset, 'data_list'):
                data_item = self.base_dataset.data_list[orig_idx]
                person_id_str = data_item['person_id_str']
                
                # 检查该样本的模态可用性
                modalities = set()
                
                # 检查图像模态
                if hasattr(self.base_dataset, 'image_cache') and person_id_str in self.base_dataset.image_cache:
                    cache = self.base_dataset.image_cache[person_id_str]
                    for modality in ['vis', 'nir', 'sk', 'cp']:
                        if len(cache.get(modality, [])) > 0:
                            modalities.add(modality)
                
                # 检查文本模态
                if data_item.get('text_description', '').strip():
                    modalities.add('text')
                
                self.sample_modalities[subset_idx] = modalities
        
        # 统计模态组合
        modality_combo_count = {}
        for modalities in self.sample_modalities.values():
            combo = tuple(sorted(modalities))
            modality_combo_count[combo] = modality_combo_count.get(combo, 0) + 1
        
        print(f"模态组合统计:")
        for combo, count in sorted(modality_combo_count.items(), key=lambda x: -x[1]):
            print(f"  {combo}: {count} 个样本")
    
    def _group_by_identity_and_modality(self):
        """按身份和模态组合分组"""
        self.identity_modality_groups = defaultdict(lambda: defaultdict(list))
        
        for subset_idx, orig_idx in enumerate(self.indices):
            if hasattr(self.base_dataset, 'data_list'):
                person_id = self.base_dataset.data_list[orig_idx]['person_id']
            else:
                person_id = self.base_dataset[orig_idx]['person_id'].item()
            
            modalities = self.sample_modalities[subset_idx]
            modality_key = tuple(sorted(modalities))
            
            self.identity_modality_groups[person_id][modality_key].append(subset_idx)
        
        self.pids = list(self.identity_modality_groups.keys())
        self.length = len(self.pids) // self.num_pids_per_batch * self.batch_size
    
    def __iter__(self):
        random.shuffle(self.pids)
        
        for start_idx in range(0, len(self.pids), self.num_pids_per_batch):
            batch_indices = []
            end_idx = min(start_idx + self.num_pids_per_batch, len(self.pids))
            selected_pids = self.pids[start_idx:end_idx]
            
            # 收集当前批次的模态组合
            current_batch_modalities = set()
            
            for pid in selected_pids:
                modality_groups = self.identity_modality_groups[pid]
                
                # 优先选择能增加模态多样性的组合
                best_modality_key = None
                best_score = -1
                
                for modality_key, indices in modality_groups.items():
                    if len(indices) == 0:
                        continue
                    
                    # 计算选择该模态组合的分数
                    score = 0
                    # 1. 优先选择尚未在批次中出现的模态组合
                    if modality_key not in current_batch_modalities:
                        score += 10
                    
                    # 2. 优先选择包含更多模态的组合
                    score += len(modality_key)
                    
                    # 3. 随机性
                    score += random.random()
                    
                    if score > best_score:
                        best_score = score
                        best_modality_key = modality_key
                
                # 从最佳模态组合中采样
                if best_modality_key and best_modality_key in modality_groups:
                    indices = modality_groups[best_modality_key]
                    if len(indices) >= self.num_instances:
                        selected = random.sample(indices, self.num_instances)
                    else:
                        selected = random.choices(indices, k=self.num_instances)
                    
                    batch_indices.extend(selected)
                    current_batch_modalities.add(best_modality_key)
                else:
                    # 回退策略：从所有可用样本中随机选择
                    all_indices = []
                    for indices in modality_groups.values():
                        all_indices.extend(indices)
                    if all_indices:
                        if len(all_indices) >= self.num_instances:
                            selected = random.sample(all_indices, self.num_instances)
                        else:
                            selected = random.choices(all_indices, k=self.num_instances)
                        batch_indices.extend(selected)
            
            # 补齐批次大小
            while len(batch_indices) < self.batch_size:
                pid = random.choice(self.pids)
                modality_groups = self.identity_modality_groups[pid]
                all_indices = []
                for indices in modality_groups.values():
                    all_indices.extend(indices)
                if all_indices:
                    idx = random.choice(all_indices)
                    batch_indices.append(idx)
            
            # 确保批次大小正确
            batch_indices = batch_indices[:self.batch_size]
            yield batch_indices
    
    def __len__(self):
        return self.length


class BalancedBatchSampler(Sampler):
    """修复后的平衡批次采样器"""
    
    def __init__(self, dataset, batch_size, num_instances=4):  # 增加每个身份的实例数
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        
        # 确保batch size至少为2，避免BatchNorm问题
        if self.batch_size < 2:
            print(f"警告: batch_size {batch_size} 太小，调整为2")
            self.batch_size = 2
            self.num_pids_per_batch = max(1, self.batch_size // num_instances)
        
        # 处理Subset
        if hasattr(dataset, 'dataset'):
            self.base_dataset = dataset.dataset
            self.indices = dataset.indices
        else:
            self.base_dataset = dataset
            self.indices = list(range(len(dataset)))
        
        # 按ID分组
        self.index_pid = defaultdict(list)
        for subset_idx, orig_idx in enumerate(self.indices):
            if hasattr(self.base_dataset, 'data_list'):
                person_id = self.base_dataset.data_list[orig_idx]['person_id']
            else:
                person_id = self.base_dataset[orig_idx]['person_id'].item() + 1  # 转回1-based
            
            self.index_pid[person_id].append(subset_idx)
        
        self.pids = list(self.index_pid.keys())
        self.length = len(self.pids) // self.num_pids_per_batch * self.batch_size
    
    def __iter__(self):
        random.shuffle(self.pids)
        
        for start_idx in range(0, len(self.pids), self.num_pids_per_batch):
            batch_indices = []
            end_idx = min(start_idx + self.num_pids_per_batch, len(self.pids))
            selected_pids = self.pids[start_idx:end_idx]
            
            for pid in selected_pids:
                indices = self.index_pid[pid]
                if len(indices) >= self.num_instances:
                    selected = random.sample(indices, self.num_instances)
                else:
                    selected = random.choices(indices, k=self.num_instances)
                batch_indices.extend(selected)
            
            # 补齐批次 - 确保batch size一致
            while len(batch_indices) < self.batch_size:
                pid = random.choice(self.pids)
                idx = random.choice(self.index_pid[pid])
                batch_indices.append(idx)
            
            # 确保返回的batch size正确
            if len(batch_indices) != self.batch_size:
                print(f"警告: batch indices数量不匹配: {len(batch_indices)} vs {self.batch_size}")
                if len(batch_indices) < self.batch_size:
                    # 重复添加样本直到达到目标batch size
                    while len(batch_indices) < self.batch_size:
                        pid = random.choice(self.pids)
                        idx = random.choice(self.index_pid[pid])
                        batch_indices.append(idx)
                else:
                    # 截断到目标batch size
                    batch_indices = batch_indices[:self.batch_size]
            
            yield batch_indices
    
    def __len__(self):
        # 确保至少返回1个batch，避免空迭代器
        return max(1, self.length // self.batch_size)


def compatible_collate_fn(batch):
    """
    兼容的collate函数，统一输出结构：
    {
        'person_id': Tensor[B],
        'images': {
            'vis'|'nir'|'sk'|'cp': Tensor[B,3,H,W]
        },
        'text_description': List[str],
        'modality_mask': {
            'vis'|'nir'|'sk'|'cp'|'text': Tensor[B]
        }
    }
    支持两种输入样本：
    1) 训练集样本：顶层含 'images' 字典
    2) 旧式样本：顶层直接含各模态键（vis/nir/sk/cp）与 'text_descriptions'
    """
    if not batch:
        return {}

    first_sample = batch[0]
    batch_dict = {}

    # person_id
    if 'person_id' in first_sample:
        batch_dict['person_id'] = torch.stack([sample['person_id'] for sample in batch])

    # 文本统一为 text_description: List[str]
    text_list = []
    if 'text_description' in first_sample:
        for sample in batch:
            td = sample.get('text_description', [""])
            if isinstance(td, list) and len(td) > 0:
                text_list.append(td[0])
            elif isinstance(td, str):
                text_list.append(td)
            else:
                text_list.append("")
    elif 'text_descriptions' in first_sample:
        for sample in batch:
            td = sample.get('text_descriptions', [""])
            text_list.append(td[0] if isinstance(td, list) and len(td) > 0 else "")
    else:
        text_list = [""] * len(batch)
    batch_dict['text_description'] = text_list

    # 组织 images
    images = {}
    modalities = ['vis', 'nir', 'sk', 'cp']
    if 'images' in first_sample and isinstance(first_sample['images'], dict):
        for m in modalities:
            # 支持缺失模态（用零填充）
            tensors = []
            for sample in batch:
                if 'images' in sample and m in sample['images'] and isinstance(sample['images'][m], torch.Tensor):
                    tensors.append(sample['images'][m])
                else:
                    # 推断尺寸，回退为 3xHxW 的零张量
                    h = w = getattr(getattr(sample, 'config', None), 'image_size', 224)
                    tensors.append(torch.zeros(3, h, w))
            images[m] = torch.stack(tensors)
    else:
        # 旧式：顶层各模态
        for m in modalities:
            if m in first_sample and isinstance(first_sample[m], torch.Tensor):
                images[m] = torch.stack([sample[m] for sample in batch])
    batch_dict['images'] = images

    # modality_mask 统一到 Tensor[B]
    mask_out = {}
    for m in modalities + ['text']:
        vals = []
        for sample in batch:
            mv = 0.0
            if 'modality_mask' in sample and isinstance(sample['modality_mask'], dict):
                raw = sample['modality_mask'].get(m, 0.0)
                if isinstance(raw, (float, int)):
                    mv = float(raw)
                elif isinstance(raw, bool):
                    mv = 1.0 if raw else 0.0
            # 若为文本且未显式标注，则依据文本是否为空
            if m == 'text' and mv == 0.0:
                td = sample.get('text_description', sample.get('text_descriptions', [""]))
                if isinstance(td, list):
                    mv = 1.0 if (len(td) > 0 and isinstance(td[0], str) and len(td[0]) > 0) else 0.0
                elif isinstance(td, str):
                    mv = 1.0 if len(td) > 0 else 0.0
            vals.append(mv)
        mask_out[m] = torch.tensor(vals, dtype=torch.float)
    batch_dict['modality_mask'] = mask_out

    return batch_dict