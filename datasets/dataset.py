# datasets/dataset.py
# 多模态人员重识别数据集实现
# 支持四种模态：可见光(vis)、近红外(nir)、骨架图(sk)、彩色分割图(cp)

import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import random
from collections import defaultdict


def infer_modalities_of_sample(ds, idx):
    """
    返回该样本拥有哪些模态的集合，如 {'vis','nir'} 或 {'text'} 等
    兼容两种结构：
      A) ds.data_list[idx].get('modality') 是单模态字符串
      B) ds.data_list[idx].get('images') 是dict，键包含多模态
    """
    rec = ds.data_list[idx]
    mods = set()

    # 常见字段1：单字段
    m = rec.get('modality', None)
    if isinstance(m, str) and m:
        mods.add(m)

    # 常见字段2：图像字典
    imgs = rec.get('images', None)
    if isinstance(imgs, dict):
        for k in imgs.keys():
            if k in ('vis', 'nir', 'sk', 'cp'):
                mods.add(k)

    # 文本
    td = rec.get('text_description', None)
    if isinstance(td, str) and td.strip():
        mods.add('text')

    return mods

class ModalityAugmentation:
    """
    多模态数据增强类
    为不同模态的图像数据提供训练和验证时的数据变换
    """
    def __init__(self, config, is_training=True):
        """
        初始化数据增强配置
        
        Args:
            config: 配置对象，包含图像尺寸、增强参数等
            is_training: 是否为训练模式，决定使用哪种变换策略
        """
        self.config = config
        self.is_training = is_training
    
    def get_transform(self):
        """
        根据训练/验证模式返回相应的数据变换序列
        
        Returns:
            transforms.Compose: PyTorch数据变换组合
        """
        if self.is_training:
            # 训练时的数据增强：包含随机裁剪、翻转、颜色扰动等
            return transforms.Compose([
                # 随机缩放裁剪，保持宽高比的同时增强数据多样性
                transforms.RandomResizedCrop(self.config.image_size, scale=(0.8, 1.0)),
                # 50%概率水平翻转，增加数据变化
                transforms.RandomHorizontalFlip(0.5),
                # 条件颜色扰动：调整亮度和对比度
                transforms.ColorJitter(brightness=0.2, contrast=0.2) if self.config.color_jitter else transforms.Lambda(lambda x: x),
                # 转换为张量格式
                transforms.ToTensor(),
                # 使用ImageNet预训练模型的标准化参数
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # 条件随机擦除：模拟遮挡情况
                transforms.RandomErasing(p=self.config.random_erase, scale=(0.02, 0.2)) if self.config.random_erase > 0 else transforms.Lambda(lambda x: x)
            ])
        else:
            # 验证时的简单变换：仅缩放和标准化
            return transforms.Compose([
                # 直接缩放到目标尺寸
                transforms.Resize((self.config.image_size, self.config.image_size)),
                # 转换为张量格式
                transforms.ToTensor(),
                # 标准化处理
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
            pass
        else:
            print(f"{split} dataset: {len(self.data_list)} samples, {len(self.person_ids)} identities")
    
    def _load_annotations(self):
        """
        加载标注文件 - 使用file_path中的目录ID作为身份ID
        
        从JSON文件中读取图像路径和对应的文本描述，解析出人员ID并构建映射关系
        JSON格式示例：[{"file_path": "vis/0001/xxx.jpg", "caption": "描述文本"}, ...]
        """
        with open(self.config.json_file, 'r', encoding='utf-8') as f:
            annotations_list = json.load(f)
        
        # 初始化存储结构
        self.id_to_annotations = {}  # 人员ID -> 文本描述列表
        self.id_to_files = {}        # 人员ID -> 图像文件路径列表
        
        # 遍历每个标注项
        for item in annotations_list:
            file_path = item['file_path']  # 例如: "vis/0001/xxx.jpg"
            caption = item['caption']      # 文本描述
            
            # 从file_path中提取目录ID作为身份ID
            parts = file_path.split('/')
            if len(parts) >= 3:
                dir_name = parts[1]  # 获取人员ID目录名，如"0001"
                if dir_name.isdigit():
                    person_id = int(dir_name)  # 转换为整数，如1
                else:
                    continue  # 跳过非数字目录名
            else:
                continue  # 跳过路径格式不正确的项
            
            # 为新的person_id初始化列表
            if person_id not in self.id_to_annotations:
                self.id_to_annotations[person_id] = []
                self.id_to_files[person_id] = []
            
            # 添加文本描述和文件路径
            self.id_to_annotations[person_id].append(caption)
            # 移除vis/前缀，匹配实际目录结构（在 __getitem__ 方法中再补回vis/）
            actual_path = file_path.replace('vis/', '')
            self.id_to_files[person_id].append(actual_path)
        
        # 合并每个人员的所有文本描述
        self.annotations = {}
        for person_id, captions in self.id_to_annotations.items():
            person_id_str = f"{person_id:04d}"  # 转换为4位字符串格式
            combined_caption = ' '.join(captions)  # 合并所有描述文本
            self.annotations[person_id_str] = combined_caption
    
    def _get_available_person_ids(self):
        """
        获取可用的person_ids - 基于json标注
        
        检查JSON标注中的人员ID是否在文件系统中有对应的vis目录
        只有同时满足以下条件的人员ID才被认为是可用的：
        1. 在JSON标注文件中存在
        2. 在文件系统的vis目录下有对应的人员文件夹
        
        Returns:
            list: 排序后的可用人员ID列表
        """
        json_ids = set(self.id_to_annotations.keys())  # 从JSON标注获取的所有人员ID
        final_ids, missing_vis = [], []  # 最终可用ID列表，缺少vis目录的ID列表
        
        # 检查每个JSON中的人员ID是否有对应的vis目录
        for pid in json_ids:
            vis_path = os.path.join(self.config.data_root, 'vis', f"{pid:04d}")
            if os.path.exists(vis_path):
                final_ids.append(pid)  # 目录存在，添加到可用列表
            else:
                missing_vis.append(pid)  # 目录不存在，添加到缺失列表
        
        # 打印数据集加载统计信息
        print(f"数据集加载信息:")
        print(f"  json中的身份ID: {len(json_ids)}")
        print(f"  最终可用的身份ID: {len(final_ids)}")
        print(f"  缺少vis目录的身份ID: {len(missing_vis)}")
        if missing_vis:
            print(f"  缺少vis目录的示例身份ID: {missing_vis[:10]}")
        
        return sorted(final_ids)  # 返回排序后的可用人员ID列表
    
    def _build_data_list(self):
        """
        构建数据列表 - 每个图片文件创建独立样本
        
        为每个可用的人员ID创建数据样本：
        - 如果该人员在JSON标注中有对应的图像文件，则为每个文件创建一个样本
        - 如果该人员没有对应的图像文件，则创建一个空的样本（file_path=None）
        
        每个样本包含：person_id, person_id_str, text_description, file_path
        """
        self.data_list = []
        
        for person_id in self.person_ids:
            person_id_str = f"{person_id:04d}"  # 转换为4位字符串格式
            # 获取该人员的文本描述，如果不存在则使用默认描述
            text_desc = self.annotations.get(person_id_str, "unknown person")
            
            # 检查该人员是否在JSON标注中有对应的图像文件
            if person_id in self.id_to_files:
                # 为该人员的每个图像文件创建一个独立的数据样本
                for file_path in self.id_to_files[person_id]:
                    self.data_list.append({
                        'person_id': person_id,           # 人员ID（整数）
                        'person_id_str': person_id_str,   # 人员ID（字符串格式）
                        'text_description': text_desc,    # 文本描述
                        'file_path': file_path            # 对应的图像文件路径
                    })
            else:
                # 如果该人员没有对应的图像文件，创建一个空样本
                self.data_list.append({
                    'person_id': person_id,
                    'person_id_str': person_id_str,
                    'text_description': text_desc,
                    'file_path': None  # 没有对应的图像文件
                })
    
    def _cache_image_paths(self):
        """
        缓存图像路径 - 基于json文件路径
        
        为每个人员的四种模态（vis, nir, sk, cp）缓存所有可用的图像文件路径
        这样可以避免在训练过程中重复进行文件系统查询，提高数据加载效率
        
        缓存结构：
        image_cache[person_id_str][modality] = [path1, path2, ...]
        """
        self.image_cache = {}
        
        # 遍历所有数据样本，为每个人员建立图像路径缓存
        for data in self.data_list:
            person_id_str = data['person_id_str']
            self.image_cache[person_id_str] = {}
            
            # 处理可见光(vis)模态 - 基于JSON标注的文件路径
            pid = int(person_id_str)
            if pid in self.id_to_files:
                vis_files = self.id_to_files[pid]  # 从JSON标注获取的文件路径列表
                self.image_cache[person_id_str]['vis'] = []
                
                for file_path in vis_files:
                    # 移除vis/前缀并重新构建完整路径
                    actual_path = file_path.replace('vis/', '')
                    full_path = os.path.join(self.config.data_root, 'vis', actual_path)
                    # 验证文件是否真实存在
                    if os.path.exists(full_path):
                        self.image_cache[person_id_str]['vis'].append(full_path)
            
            # 处理其他三种模态(nir, sk, cp) - 基于文件系统扫描
            for modality in ['nir', 'sk', 'cp']:
                folder_path = os.path.join(self.config.data_root, modality, person_id_str)
                if os.path.exists(folder_path):
                    # 扫描目录下所有支持的图像文件
                    images = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    # 构建完整的文件路径列表
                    self.image_cache[person_id_str][modality] = [
                        os.path.join(folder_path, img) for img in images
                    ]
                else:
                    # 如果目录不存在，设置为空列表
                    self.image_cache[person_id_str][modality] = []
    
    def __len__(self):
        """返回数据集中样本的总数"""
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含以下字段的样本字典
                - person_id: 人员标签ID (torch.long)
                - text_description: 文本描述列表
                - images: 四种模态的图像张量字典
                - modality_mask: 各模态的可用性掩码
        """
        data = self.data_list[idx]
        person_id = data['person_id']
        person_id_str = data['person_id_str']
        text_desc = data['text_description']

        # 构建基础样本结构
        sample = {
            'person_id': torch.tensor(self.pid2label[person_id], dtype=torch.long),  # 转换为标签ID
            'text_description': [text_desc],  # 文本描述（列表格式）
            'modality_mask': {},  # 模态掩码，标识哪些模态可用
        }

        images = {}
        file_path = data.get('file_path', None)  # 获取指定的文件路径（如果有）
        
        # 处理四种模态的图像数据
        for modality in self.modality_folders:
            image_paths = self.image_cache[person_id_str][modality]  # 获取该模态的缓存路径
            # 检查是否应用模态dropout（仅在训练时）
            use_drop = self.is_training and (self.config.modality_dropout > 0)
            
            # 如果有可用图像且通过dropout检查
            if image_paths and (not use_drop or random.random() > self.config.modality_dropout):
                # 选择图像路径
                if file_path and modality == 'vis':
                    # 对于vis模态，优先使用指定的文件路径
                    full_path = os.path.join(self.config.data_root, 'vis', file_path)
                    selected_path = full_path if os.path.exists(full_path) else random.choice(image_paths)
                else:
                    # 其他模态随机选择一张图像
                    selected_path = random.choice(image_paths)
                
                try:
                    # 加载并处理图像
                    image = Image.open(selected_path).convert('RGB')
                    images[modality] = self.transform(image)  # 应用数据变换
                    sample['modality_mask'][modality] = 1.0  # 标记该模态可用
                except Exception as e:
                    # 图像加载失败，使用零张量
                    print(f"Error loading {selected_path}: {e}")
                    images[modality] = torch.zeros(3, self.config.image_size, self.config.image_size)
                    sample['modality_mask'][modality] = 0.0
            else:
                # 无可用图像或被dropout丢弃，使用零张量
                images[modality] = torch.zeros(3, self.config.image_size, self.config.image_size)
                sample['modality_mask'][modality] = 0.0

        sample['images'] = images
        return sample

class BalancedBatchSampler(Sampler):
    """
    平衡批次采样器
    
    用于人员重识别任务的特殊采样器，确保每个批次中：
    - 包含固定数量的不同人员ID (num_pids_per_batch)
    - 每个人员ID有固定数量的样本实例 (num_instances)
    - 总批次大小 = num_pids_per_batch * num_instances
    
    这种采样方式有利于对比学习和三元组损失的训练
    """
    def __init__(self, dataset, batch_size, num_instances=4):
        """
        初始化平衡批次采样器
        
        Args:
            dataset: 数据集对象或数据集子集
            batch_size: 批次大小
            num_instances: 每个人员ID在批次中的实例数量，默认4
        """
        self.batch_size = batch_size
        self.num_instances = num_instances  # 每个人员ID的实例数
        self.num_pids_per_batch = batch_size // num_instances  # 每批次的人员ID数
        
        # 批次大小检查和调整
        if self.batch_size < 2:
            print(f"警告: batch_size {batch_size} 太小，调整为2")
            self.batch_size = 2
            self.num_pids_per_batch = max(1, self.batch_size // num_instances)
        
        # 处理数据集子集的情况
        if hasattr(dataset, 'dataset'):
            # 如果是Subset对象，获取原始数据集和索引映射
            self.base_dataset = dataset.dataset
            self.indices = dataset.indices
        else:
            # 如果是完整数据集，直接使用
            self.base_dataset = dataset
            self.indices = list(range(len(dataset)))
        
        # 构建人员ID到样本索引的映射
        self.index_pid = defaultdict(list)  # person_id -> [sample_indices]
        for subset_idx, orig_idx in enumerate(self.indices):
            # 从数据集中获取人员ID
            if hasattr(self.base_dataset, 'data_list'):
                person_id = self.base_dataset.data_list[orig_idx]['person_id']
            else:
                person_id = self.base_dataset[orig_idx]['person_id'].item() + 1
            self.index_pid[person_id].append(subset_idx)
        
        self.pids = list(self.index_pid.keys())  # 所有可用的人员ID列表
        # 计算总批次数（确保能构成完整批次的人员ID数量）
        self.length = len(self.pids) // self.num_pids_per_batch * self.batch_size
    
    def __iter__(self):
        """
        迭代生成平衡的批次索引
        
        每个批次确保包含指定数量的不同人员ID，每个人员ID有指定数量的样本实例
        
        Yields:
            list: 批次样本索引列表
        """
        # 随机打乱人员ID顺序，增加训练随机性
        random.shuffle(self.pids)
        
        # 按照每批次的人员ID数量进行分组
        for start_idx in range(0, len(self.pids), self.num_pids_per_batch):
            batch_indices = []
            end_idx = min(start_idx + self.num_pids_per_batch, len(self.pids))
            selected_pids = self.pids[start_idx:end_idx]  # 当前批次选中的人员ID
            
            # 为每个选中的人员ID采样指定数量的实例
            for pid in selected_pids:
                indices = self.index_pid[pid]  # 该人员ID的所有样本索引
                if len(indices) >= self.num_instances:
                    # 如果样本数充足，随机采样不重复的实例
                    selected = random.sample(indices, self.num_instances)
                else:
                    # 如果样本数不足，允许重复采样
                    selected = random.choices(indices, k=self.num_instances)
                batch_indices.extend(selected)
            
            # 确保批次大小正确：如果不足，随机补充样本
            while len(batch_indices) < self.batch_size:
                pid = random.choice(self.pids)
                idx = random.choice(self.index_pid[pid])
                batch_indices.append(idx)
            
            # 最终批次大小调整
            if len(batch_indices) != self.batch_size:
                if len(batch_indices) < self.batch_size:
                    # 不足的情况：继续随机补充
                    while len(batch_indices) < self.batch_size:
                        pid = random.choice(self.pids)
                        idx = random.choice(self.index_pid[pid])
                        batch_indices.append(idx)
                else:
                    # 超出的情况：截断到指定大小
                    batch_indices = batch_indices[:self.batch_size]
            
            yield batch_indices
    
    def __len__(self):
        """返回采样器的总批次数量"""
        return max(1, self.length // self.batch_size)


class ModalAwarePKSampler(Sampler):
    """
    模态感知的P×K采样器
    
    在原有P×K采样基础上，尽量保证每个ID在同一batch里含有vis和非vis（nir/sk/cp/text任一）
    这样可以显著减少SDM损失为0和融合层NaN的问题
    """
    def __init__(self, dataset, batch_size, num_instances=4, seed=42,
                 prefer_complete=True, ensure_rgb=True):
        """
        Args:
            dataset: 数据集对象
            batch_size: 批次大小
            num_instances: 每个ID的实例数量（K）
            seed: 随机种子
            prefer_complete: 优先选择模态齐全的ID
            ensure_rgb: 尽量保证每个ID都有vis模态
        """
        assert batch_size % num_instances == 0, f"batch_size({batch_size}) must be divisible by num_instances({num_instances})"
        
        # 处理数据集子集的情况
        if hasattr(dataset, 'dataset'):
            # 如果是Subset对象，获取原始数据集和索引映射
            self.base_dataset = dataset.dataset
            self.indices = dataset.indices
        else:
            # 如果是完整数据集，直接使用
            self.base_dataset = dataset
            self.indices = list(range(len(dataset)))
        
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.seed = seed
        self.rng = random.Random(seed)
        self.prefer_complete = prefer_complete
        self.ensure_rgb = ensure_rgb

        # 1) 建 pid -> 索引列表
        self.pid_to_indices = {}
        for subset_idx, orig_idx in enumerate(self.indices):
            # 从数据集中获取人员ID
            if hasattr(self.base_dataset, 'data_list'):
                person_id = self.base_dataset.data_list[orig_idx]['person_id']
            else:
                person_id = self.base_dataset[orig_idx]['person_id'].item() + 1
            
            if isinstance(person_id, torch.Tensor):
                person_id = int(person_id.item())
            else:
                person_id = int(person_id)
                
            self.pid_to_indices.setdefault(person_id, []).append(subset_idx)

        # 2) 建 pid -> 模态桶（vis / nonvis 以及细分）
        self.pid_to_mod = {}
        for pid, idxs in self.pid_to_indices.items():
            buckets = {'vis': [], 'nir': [], 'sk': [], 'cp': [], 'text': [], 'nonvis': []}
            for idx in idxs:
                orig_idx = self.indices[idx]
                mods = infer_modalities_of_sample(self.base_dataset, orig_idx)
                if 'vis' in mods:  
                    buckets['vis'].append(idx)
                if 'nir' in mods:  
                    buckets['nir'].append(idx)
                    buckets['nonvis'].append(idx)
                if 'sk' in mods:   
                    buckets['sk'].append(idx)
                    buckets['nonvis'].append(idx)
                if 'cp' in mods:   
                    buckets['cp'].append(idx)
                    buckets['nonvis'].append(idx)
                if 'text' in mods: 
                    buckets['text'].append(idx)
                    buckets['nonvis'].append(idx)
            self.pid_to_mod[pid] = buckets

        # 3) 可选：把"模态较齐全"的ID排在前面（提高命中率）
        self.all_pids = list(self.pid_to_indices.keys())
        if self.prefer_complete:
            # 优先级：既有vis又有nonvis > 只有vis > 只有nonvis > 其他
            def completeness_score(p):
                buckets = self.pid_to_mod[p]
                has_vis = len(buckets['vis']) > 0
                has_nonvis = len(buckets['nonvis']) > 0
                return (has_vis and has_nonvis, has_vis, has_nonvis)
            self.all_pids.sort(key=completeness_score, reverse=True)

    def __len__(self):
        # 近似：所有样本除以batch_size
        total_samples = sum(len(indices) for indices in self.pid_to_indices.values())
        return total_samples // self.batch_size

    def __iter__(self):
        pids = self.all_pids[:]  # 拷贝
        self.rng.shuffle(pids)

        batch = []
        i = 0
        while i + self.num_pids_per_batch <= len(pids):
            chosen_pids = pids[i:i+self.num_pids_per_batch]
            i += self.num_pids_per_batch

            for pid in chosen_pids:
                b = self.pid_to_mod[pid]
                chosen = []

                # A) 先拿1张 vis（如果需要且有）
                if self.ensure_rgb and len(b['vis']) > 0:
                    chosen.append(self.rng.choice(b['vis']))

                # B) 再拿1张 非vis（如果有）
                if len(b['nonvis']) > 0:
                    # 避免与已选重复
                    cand = [x for x in b['nonvis'] if x not in chosen]
                    if not cand and b['nonvis']:
                        cand = b['nonvis']
                    if cand:
                        chosen.append(self.rng.choice(cand))

                # C) 补齐到K（从该ID的全部索引里选，允许重复回绕）
                pool = self.pid_to_indices[pid][:]
                self.rng.shuffle(pool)
                j = 0
                while len(chosen) < self.num_instances and j < 10 * len(pool):
                    idx = pool[j % len(pool)]
                    chosen.append(idx)
                    j += 1

                # D) 若实在不够（极端稀疏ID），最后从该ID任取（允许重复）
                while len(chosen) < self.num_instances:
                    chosen.append(self.rng.choice(self.pid_to_indices[pid]))

                batch.extend(chosen)

            # 输出一个batch
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # 末尾不足一个batch则丢弃（常见做法）


class MultiModalBalancedSampler(Sampler):
    """
    多模态平衡采样器：确保每个ID都有RGB+非RGB模态组合
    
    专门解决SDM损失中"无正样本"问题，通过强制每个ID包含：
    - ≥1 张 RGB (vis) 模态
    - ≥1 张 非RGB模态 (nir/sk/cp/text)
    """
    def __init__(self, dataset, batch_size, num_instances=4, seed=42):
        """
        Args:
            dataset: 数据集对象
            batch_size: 批次大小
            num_instances: 每个ID的实例数量（K）
            seed: 随机种子
        """
        assert batch_size % num_instances == 0, f"batch_size({batch_size}) must be divisible by num_instances({num_instances})"
        
        # 处理数据集子集的情况
        if hasattr(dataset, 'dataset'):
            self.base_dataset = dataset.dataset
            self.indices = dataset.indices
        else:
            self.base_dataset = dataset
            self.indices = list(range(len(dataset)))
        
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.seed = seed
        self.rng = random.Random(seed)

        # 构建ID到索引的映射
        self.pid_to_indices = {}
        for subset_idx, orig_idx in enumerate(self.indices):
            if hasattr(self.base_dataset, 'data_list'):
                person_id = self.base_dataset.data_list[orig_idx]['person_id']
            else:
                person_id = self.base_dataset[orig_idx]['person_id'].item() + 1
            
            if isinstance(person_id, torch.Tensor):
                person_id = int(person_id.item())
            else:
                person_id = int(person_id)
                
            self.pid_to_indices.setdefault(person_id, []).append(subset_idx)

        # 构建每个ID的模态分布
        self.pid_to_modalities = {}
        self.valid_pids = []  # 只包含有RGB+非RGB组合的ID
        
        for pid, idxs in self.pid_to_indices.items():
            rgb_indices = []
            non_rgb_indices = []
            
            for idx in idxs:
                orig_idx = self.indices[idx]
                mods = infer_modalities_of_sample(self.base_dataset, orig_idx)
                
                if 'vis' in mods:
                    rgb_indices.append(idx)
                if any(m in mods for m in ['nir', 'sk', 'cp', 'text']):
                    non_rgb_indices.append(idx)
            
            self.pid_to_modalities[pid] = {
                'rgb': rgb_indices,
                'non_rgb': non_rgb_indices,
                'all': idxs
            }
            
            # 只保留有RGB+非RGB组合的ID
            if len(rgb_indices) > 0 and len(non_rgb_indices) > 0:
                self.valid_pids.append(pid)

    def __len__(self):
        total_samples = sum(len(indices) for indices in self.pid_to_indices.values())
        return total_samples // self.batch_size

    def __iter__(self):
        # 只使用有效的ID（有RGB+非RGB组合）
        pids = self.valid_pids[:]
        self.rng.shuffle(pids)

        batch = []
        i = 0
        
        while i + self.num_pids_per_batch <= len(pids):
            chosen_pids = pids[i:i+self.num_pids_per_batch]
            i += self.num_pids_per_batch

            for pid in chosen_pids:
                mod_info = self.pid_to_modalities[pid]
                chosen = []

                # 强制选择：1张RGB + 1张非RGB
                if mod_info['rgb']:
                    chosen.append(self.rng.choice(mod_info['rgb']))
                
                if mod_info['non_rgb']:
                    chosen.append(self.rng.choice(mod_info['non_rgb']))

                # 补齐到K张（从该ID的所有样本中选择）
                remaining_needed = self.num_instances - len(chosen)
                if remaining_needed > 0:
                    # 从所有样本中随机选择，避免重复
                    available = [idx for idx in mod_info['all'] if idx not in chosen]
                    if available:
                        # 随机选择剩余需要的样本
                        if len(available) >= remaining_needed:
                            chosen.extend(self.rng.sample(available, remaining_needed))
                        else:
                            # 如果不够，允许重复选择
                            chosen.extend(self.rng.choices(available, k=remaining_needed))
                    else:
                        # 极端情况：从所有样本中重复选择
                        chosen.extend(self.rng.choices(mod_info['all'], k=remaining_needed))

                batch.extend(chosen)

            # 输出一个batch
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # 末尾不足一个batch则丢弃


def compatible_collate_fn(batch):
    """
    兼容的批次整理函数 (Collate Function)
    
    将一个批次的独立样本整理成批处理格式的张量
    支持多种数据格式的兼容性处理，确保模型能正确接收批次数据
    
    Args:
        batch (list): 批次样本列表，每个样本是由Dataset.__getitem__返回的字典
        
    Returns:
        dict: 整理后的批次数据字典，包含以下字段：
            - person_id: 人员ID张量 (batch_size,)
            - text_description: 文本描述列表 [str, ...]
            - images: 各模态图像张量字典 {modality: (batch_size, 3, H, W)}
            - modality_mask: 模态掩码张量字典 {modality: (batch_size,)}
    """
    if not batch:
        return {}

    first_sample = batch[0]
    batch_dict = {}

    # 处理人员ID：将所有样本的person_id堆叠成张量
    if 'person_id' in first_sample:
        batch_dict['person_id'] = torch.stack([sample['person_id'] for sample in batch])

    # 处理文本描述：兼容多种文本字段格式
    text_list = []
    if 'text_description' in first_sample:
        # 标准的text_description字段
        for sample in batch:
            td = sample.get('text_description', [""])
            if isinstance(td, list) and len(td) > 0:
                text_list.append(td[0])  # 取列表的第一个元素
            elif isinstance(td, str):
                text_list.append(td)     # 直接使用字符串
            else:
                text_list.append("")     # 默认空字符串
    elif 'text_descriptions' in first_sample:
        # 兼容text_descriptions字段（复数形式）
        for sample in batch:
            td = sample.get('text_descriptions', [""])
            text_list.append(td[0] if isinstance(td, list) and len(td) > 0 else "")
    else:
        # 如果没有文本字段，使用空字符串填充
        text_list = [""] * len(batch)
    batch_dict['text_description'] = text_list

    # 处理图像数据：堆叠各模态的图像张量
    # 修复：先计算真实的模态可用性，避免零张量污染融合
    images = {}
    modalities = ['vis', 'nir', 'sk', 'cp']
    
    # 第一步：计算每个样本每个模态的真实可用性
    real_modality_mask = {}
    for m in modalities + ['text']:
        real_modality_mask[m] = []
    
    for sample_idx, sample in enumerate(batch):
        # 检查图像模态的真实可用性
        for m in modalities:
            has_valid_image = False
            if 'images' in sample and isinstance(sample['images'], dict):
                if m in sample['images'] and isinstance(sample['images'][m], torch.Tensor):
                    tensor = sample['images'][m]
                    # 关键修复：检查是否为非零张量，避免把全零张量当作有效模态
                    if tensor.numel() > 0 and tensor.abs().sum() > 1e-6:
                        has_valid_image = True
            
            # 结合原始modality_mask进行二次确认
            if 'modality_mask' in sample and isinstance(sample['modality_mask'], dict):
                original_mask = sample['modality_mask'].get(m, 0.0)
                if isinstance(original_mask, bool):
                    has_valid_image = has_valid_image and original_mask
                elif isinstance(original_mask, (float, int)):
                    has_valid_image = has_valid_image and (float(original_mask) > 0.5)
            
            real_modality_mask[m].append(1.0 if has_valid_image else 0.0)
        
        # 检查文本模态的真实可用性
        has_valid_text = False
        td = sample.get('text_description', sample.get('text_descriptions', [""]))
        if isinstance(td, list):
            has_valid_text = len(td) > 0 and isinstance(td[0], str) and len(td[0].strip()) > 0
        elif isinstance(td, str):
            has_valid_text = len(td.strip()) > 0
        real_modality_mask['text'].append(1.0 if has_valid_text else 0.0)
    
    # 第二步：构建图像batch（仍需要占位符保持batch结构，但mask会指示真实可用性）
    if 'images' in first_sample and isinstance(first_sample['images'], dict):
        for m in modalities:
            tensors = []
            for sample_idx, sample in enumerate(batch):
                if 'images' in sample and m in sample['images'] and isinstance(sample['images'][m], torch.Tensor):
                    tensors.append(sample['images'][m])
                else:
                    # 缺失模态：填充零张量作为占位符（但mask会标记为无效）
                    h = w = 224
                    tensors.append(torch.zeros(3, h, w))
            images[m] = torch.stack(tensors)
    else:
        # 兼容直接在样本根级别存储图像的格式
        for m in modalities:
            if m in first_sample and isinstance(first_sample[m], torch.Tensor):
                images[m] = torch.stack([sample[m] for sample in batch])
    batch_dict['images'] = images

    # 第三步：使用计算出的真实模态掩码
    mask_out = {}
    for m in modalities + ['text']:
        mask_out[m] = torch.tensor(real_modality_mask[m], dtype=torch.float)
    batch_dict['modality_mask'] = mask_out

    return batch_dict
