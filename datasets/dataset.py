# datasets/dataset.py
# 多模态人员重识别数据集实现
# 支持四种模态：可见光(vis)、近红外(nir)、骨架图(sk)、彩色分割图(cp)

"""
# 数据定位与对齐规则（train 集）

**总体约定**

* `text_annos.json` **仅列出 VIS（可见光）模态的文件路径**；其余模态（`nir/sk/cp`）不在 JSON 中显式给出，需要\*\*根据 VIS 路径解析出的身份 ID（PID）\*\*到对应模态子目录中检索。
* **文本模态**来自 `caption` 字段，且与 `file_path` 指向的 **同一张 VIS 图像**一一对应。

**PID 解析**

* `file_path` 形如：`vis/0941/0941_sysu_0470_cam5_0008_vis.jpg`
* 其中 `0941` 为身份 ID（PID）。一般可按正则 `^vis/(\d{4})/.*_vis\.jpg$` 提取，或按路径第二段解析。

**多模态检索**

* 给定 PID（如 `0941`），在以下目录下按 PID 子目录收集同身份的图像：

  * `nir/0941/` → 近红外（NIR）
  * `sk/0941/` → 素描（Sketch）
  * `cp/0941/` → 彩铅（Color Pencil）
* 每个子目录下通常包含该身份的**多张**对应模态图像。

**文本对齐**

* `caption` 与 `file_path` 中的 **VIS 图像**严格对齐；同一身份可能在 JSON 中出现多条记录（不同 VIS 图像与其各自的 caption）。

**字段示例（简化）**

```json
{
  "id": 562,                       // 注释编号（1–600，非连续）；同一 id 可出现多条记录
  "file_path": "vis/0941/0941_sysu_0470_cam5_0007_vis.jpg",  // 用于解析 PID=0941
  "caption": "……",                 // 与上述 VIS 图像一一对应的文本描述
  "split": "train"
}
```

**重要说明**

* `id` 为**注释编号**而非身份 ID；请以 `file_path` 解析得到的 PID 作为身份标识。
* 构建多模态样本时，建议以该 VIS 图像为锚点：

  * `images['vis']` 仅包含该条的 VIS 图像；
  * `images['nir'|'sk'|'cp']` 为同 PID 目录下检索到的全部对应模态图像；
  * 文本模态取自 `caption`。
"""

import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import random
from collections import defaultdict

# ===== Canonical modality names: keep dataset-native names =====
CANON_DS = {
    'vis': 'vis', 'rgb': 'vis', 'visible': 'vis', 'v': 'vis',
    'nir': 'nir', 'ir': 'nir', 'infrared': 'nir',
    'sk': 'sk', 'sketch': 'sk',
    'cp': 'cp', 'cpencil': 'cp', 'colorpencil': 'cp', 'coloredpencil': 'cp',
    'txt': 'text', 'text': 'text', 'caption': 'text'
}
IMG_MODALITIES = {'vis', 'nir', 'sk', 'cp'}
ALL_MODALITIES = IMG_MODALITIES | {'text'}

def canon_mod(name: str) -> str:
    """规范化模态名称到数据集原生名称"""
    if name is None:
        return ''
    return CANON_DS.get(str(name).lower().strip(), str(name).lower().strip())

def _truthy(x) -> bool:
    """判定"有内容"的通用工具：张量非空、路径非空、列表有元素、数字>0.5"""
    if x is None:
        return False
    if isinstance(x, (list, tuple, set, dict)):
        return len(x) > 0
    if isinstance(x, (int, float)):
        return float(x) > 0.5
    if isinstance(x, str):
        return len(x.strip()) > 0
    if torch.is_tensor(x):
        try:
            return int(x.nelement()) > 0 and (x.abs().sum() > 1e-6 if x.dtype.is_floating_point else True)
        except Exception:
            return False
    return True

def analyze_sampling_capability(ds, limit=None):
    """Guide19: 替换统计面板，强制走规范化"""
    from collections import Counter
    # 兼容 Subset
    if hasattr(ds, 'dataset') and hasattr(ds, 'indices'):
        base_ds = ds.dataset
        indices = list(ds.indices)
    else:
        base_ds = ds
        indices = list(range(len(ds)))

    if limit:
        indices = indices[:min(limit, len(indices))]

    c = Counter()
    pid_set = set()
    strong_ids = set()

    def get_pid(idx):
        s = None
        if hasattr(base_ds, 'samples'):
            s = base_ds.samples[idx]
        else:
            try:
                item = base_ds[idx]
                if isinstance(item, dict):
                    s = item
                elif isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[-1], dict):
                    s = item[-1]
            except Exception:
                s = None
        if isinstance(s, dict):
            return int(s.get('person_id') or s.get('pid') or s.get('label') or -1)
        return -1

    for orig_idx in indices:
        pid = get_pid(orig_idx)
        if pid >= 0:
            pid_set.add(pid)
        mods_img = infer_modalities_of_sample(base_ds, orig_idx, include_text=False)
        mods_all = infer_modalities_of_sample(base_ds, orig_idx, include_text=True)
        # 统计只计图像模态
        for m in mods_img:
            c[canon_mod(m)] += 1
        has_vis = ('vis' in mods_img)
        has_nonvis = bool(mods_img & {'nir','sk','cp'}) or ('text' in mods_all)
        if pid >= 0 and has_vis and has_nonvis:
            strong_ids.add(pid)

    print("==================================================")
    print("数据集采样能力分析")
    print("==================================================")
    print(f"[INFO] 开始分析数据集采样能力...")
    print("数据集统计:")
    print(f"  总ID数: {len(pid_set)}")
    print(f"  总样本数: {len(indices)}")
    # 只展示 vis/nir/sk/cp
    print("  各模态分布:", {k:v for k,v in c.items() if k in IMG_MODALITIES})
    print(f"  可配对ID数 (K≥2): {len(strong_ids)} ({len(strong_ids)/max(1,len(pid_set)):.1%})")
    print("==================================================")
    
    return len(strong_ids), len(pid_set)

def quick_scan(ds, n=200):
    """Guide19: 快速自检函数（跑200个样本就能看出修复成效）"""
    from collections import Counter
    if hasattr(ds, 'dataset') and hasattr(ds, 'indices'):
        base, idxs = ds.dataset, ds.indices[:min(n, len(ds.indices))]
    else:
        base, idxs = ds, list(range(min(n, len(ds))))
    c = Counter()
    pair = 0
    pids = set()
    for orig_idx in idxs:
        m_img = infer_modalities_of_sample(base, orig_idx, include_text=False)
        m_all = infer_modalities_of_sample(base, orig_idx, include_text=True)
        for m in m_img:
            c[canon_mod(m)] += 1
        vis = 'vis' in m_img
        nonvis = bool(m_img & {'nir','sk','cp'}) or ('text' in m_all)       
        pair += int(vis and nonvis)
    # pid 统计可选
    print("[仅图像] 模态计数:", {k:v for k,v in c.items() if k in IMG_MODALITIES})     
    print(f"vis+非vis 配对样本: {pair}/{len(idxs)}  比例: {pair/len(idxs):.1%}")
    # 检查是否还有旧名称泄漏
    old_names = {'rgb', 'ir', 'sketch', 'cpencil'}
    leaked_old_names = {k: v for k, v in c.items() if k in old_names}
    if leaked_old_names:
        print(f"⚠️ 检测到旧模态名称泄漏: {leaked_old_names}")
    else:
        print("✅ 未检测到旧模态名称，规范化成功")

@torch.no_grad()
def infer_modalities_of_sample(dataset, index: int, include_text: bool = True):
    """
    Guide20: 使用数据集内置方法或直接访问样本
    
    返回该样本可用模态的规范化集合（vis/nir/sk/cp/[text]）
    """
    # 优先使用数据集内置的infer_modalities_of_sample方法
    if hasattr(dataset, 'infer_modalities_of_sample') and callable(getattr(dataset, 'infer_modalities_of_sample')):
        try:
            mods = dataset.infer_modalities_of_sample(index)
            if include_text:
                return mods
            else:
                return mods & IMG_MODALITIES
        except Exception:
            pass
    
    # 兼容性后备：尝试直接访问样本数据
    sample = None
    if hasattr(dataset, 'data_list') and index < len(dataset.data_list):
        sample = dataset.data_list[index]
    elif hasattr(dataset, 'samples') and index < len(dataset.samples):
        sample = dataset.samples[index]
    else:
        try:
            s = dataset[index]
            if isinstance(s, dict):
                sample = s
            elif isinstance(s, (list, tuple)) and len(s) >= 3 and isinstance(s[-1], dict):
                sample = s[-1]
        except Exception:
            sample = None

    if sample is None:
        return set()
    
    mods = set()

    # 1) modality_mask: {'vis':1.0, 'nir':1.0, ...}
    mm = sample.get('modality_mask') or sample.get('modal_mask') or sample.get('mods')
    if isinstance(mm, dict):
        for k, v in mm.items():
            m = canon_mod(k)
            if m in IMG_MODALITIES and _truthy(v):
                mods.add(m)

    # 2) images/paths 容器：{ 'vis': [...], 'nir': [...], 'text': '...' }
    imgs = sample.get('images') or sample.get('paths') or sample.get('imgs')
    if isinstance(imgs, dict):
        for k, v in imgs.items():
            m = canon_mod(k)
            if m in IMG_MODALITIES and _truthy(v):
                mods.add(m)

    # 3) primary 字段：'modality' / 'mode' / 'mod'
    primary = sample.get('modality') or sample.get('mode') or sample.get('mod')
    if primary:
        m = canon_mod(primary)
        if m in IMG_MODALITIES:
            mods.add(m)

    # 4) 文本模态（可选）
    if include_text:
        if _truthy(sample.get('text_description')) or _truthy(sample.get('text')) or _truthy(sample.get('caption')):
            mods.add('text')
        elif isinstance(imgs, dict) and _truthy(imgs.get('text')):
            mods.add('text')

    # 最终只保留标准命名
    return {m for m in mods if m in (ALL_MODALITIES if include_text else IMG_MODALITIES)}

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
        Guide20: 构建多模态样本 - 从JSON中的VIS路径扩展到所有模态
        
        根据guide20.md的数据定位与对齐规则：
        1. JSON只列出VIS模态文件路径
        2. 根据PID到其他模态子目录检索图像
        3. 文本与特定VIS图像一一对应
        """
        with open(self.config.json_file, 'r', encoding='utf-8') as f:
            annotations_list = json.load(f)
        
        # 构建完整的多模态样本列表
        self.samples = []
        
        for entry in annotations_list:
            file_path = entry['file_path']  # e.g. "vis/0941/0941_sysu_0470_cam5_0008_vis.jpg"
            caption = entry['caption']      # 文本描述
            
            # 从file_path解析PID
            parts = file_path.split('/')
            if len(parts) >= 2:
                pid_str = parts[1]  # '0941'
                if pid_str.isdigit():
                    pid = int(pid_str)
                else:
                    continue
            else:
                continue
                
            # === 各模态路径收集 ===
            images = {'vis': [], 'nir': [], 'sk': [], 'cp': []}
            
            # VIS: 直接来自file_path
            vis_full_path = os.path.join(self.config.data_root, file_path)
            if os.path.exists(vis_full_path):
                images['vis'].append(vis_full_path)
            
            # NIR/SK/CP: 根据PID检索同身份目录下的所有图像
            for mod in ['nir', 'sk', 'cp']:
                mod_dir = os.path.join(self.config.data_root, mod, pid_str)
                if os.path.isdir(mod_dir):
                    import glob
                    imgs = glob.glob(os.path.join(mod_dir, "*.jpg"))
                    imgs.extend(glob.glob(os.path.join(mod_dir, "*.jpeg")))
                    imgs.extend(glob.glob(os.path.join(mod_dir, "*.png")))
                    images[mod].extend(imgs)
            
            # === 构建样本结构 ===
            sample = {
                'person_id': pid,
                'pid': pid,  # 采样器可能使用这个字段
                'images': {k: v for k, v in images.items() if len(v) > 0},
                'modality_mask': {k: float(len(v) > 0) for k, v in images.items()},
                'text': caption,
                'text_description': caption,  # 兼容性字段
                'file_path': file_path,  # 保留原始路径用于调试
            }
            self.samples.append(sample)
        
        print(f"[INFO] 构建了 {len(self.samples)} 个多模态样本")
    
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
        # 修复：从self.samples中获取人员ID，而不是不存在的id_to_annotations
        json_ids = set(sample['person_id'] for sample in self.samples)  # 从samples获取的所有人员ID
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
        Guide20: 过滤样本以匹配指定的person_ids
        
        现在samples已经包含完整的多模态结构，只需要过滤匹配的person_id
        """
        if hasattr(self, 'samples'):
            # 过滤出指定person_ids的样本
            self.data_list = [s for s in self.samples if s['person_id'] in self.person_ids]
        else:
            self.data_list = []
    
    def _cache_image_paths(self):
        """
        Guide20: 图像路径已经在样本构建时完成，无需额外缓存
        
        samples中的images字段已经包含了完整的路径信息
        """
        # 为了兼容性，从samples构建简化的缓存
        self.image_cache = {}
        
        for sample in self.data_list:
            person_id_str = f"{sample['person_id']:04d}"
            self.image_cache[person_id_str] = sample.get('images', {})
    
    def __len__(self):
        """返回数据集中样本的总数"""
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        Guide20: 使用新的多模态样本结构返回数据
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 多模态样本包含完整的images/modality_mask/text字段
        """
        # 获取预构建的多模态样本
        sample = self.data_list[idx].copy()
        
        # 转换person_id为标签索引
        sample['person_id'] = torch.tensor(self.pid2label[sample['person_id']], dtype=torch.long)
        
        # 处理图像数据
        images_tensor = {}
        available_modalities = []
        
        for modality in self.modality_folders:
            image_paths = sample.get('images', {}).get(modality, [])
            
            # 检查模态dropout（仅训练时）
            use_drop = self.is_training and hasattr(self.config, 'modality_dropout') and (self.config.modality_dropout > 0)
            
            if image_paths and (not use_drop or random.random() > self.config.modality_dropout):
                try:
                    # 随机选择一张图像
                    selected_path = random.choice(image_paths)
                    image = Image.open(selected_path).convert('RGB')
                    images_tensor[modality] = self.transform(image)
                    available_modalities.append(modality)
                except Exception as e:
                    print(f"Error loading {modality} image: {e}")
                    images_tensor[modality] = torch.zeros(3, self.config.image_size, self.config.image_size)
            else:
                # 无图像或被dropout，使用零张量
                images_tensor[modality] = torch.zeros(3, self.config.image_size, self.config.image_size)
        
        # 设置主模态
        if 'vis' in available_modalities:
            sample['modality'] = 'vis'
        elif available_modalities:
            sample['modality'] = available_modalities[0]
        elif sample.get('text'):
            sample['modality'] = 'text'
        else:
            sample['modality'] = 'unknown'
        
        # 更新样本结构
        sample['images'] = images_tensor
        sample['text_description'] = [sample.get('text', 'unknown person')]  # 兼容性
        
        return sample
    
    def infer_modalities_of_sample(self, item):
        """
        Guide20: 直接访问样本的模态信息 - 兼容旧版函数签名
        
        Args:
            item: 样本字典或索引
            
        Returns:
            set: 可用的模态集合
        """
        if isinstance(item, int):
            # 如果是索引，获取对应样本
            if item < len(self.data_list):
                sample = self.data_list[item]
            else:
                return set()
        elif isinstance(item, dict):
            sample = item
        else:
            return set()
        
        mods = set()
        
        # 检查modality_mask
        mm = sample.get('modality_mask', {})
        if isinstance(mm, dict):
            for k, v in mm.items():
                m = canon_mod(k)
                if m in IMG_MODALITIES and _truthy(v):
                    mods.add(m)
        
        # 检查images字段
        imgs = sample.get('images', {})
        if isinstance(imgs, dict):
            for k, v in imgs.items():
                m = canon_mod(k)
                if m in IMG_MODALITIES and _truthy(v):
                    mods.add(m)
        
        # 检查文本模态
        if _truthy(sample.get('text')) or _truthy(sample.get('text_description')):
            mods.add('text')
        
        return mods

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


class ModalAwarePKSampler_Strict(Sampler):
    """
    强配对：同一个ID必须既有 vis 也有 非vis(nir/sk/cp/text) 才算可配对。
    支持 allow_id_reuse=True：同一 epoch 内可重复使用ID，防止采样耗尽。
    """
    def __init__(self, dataset, num_ids_per_batch=4, num_instances=4,
                 allow_id_reuse=True, min_modal_coverage=0.6, include_text=True):
        # 处理数据集子集的情况
        if hasattr(dataset, 'dataset'):
            self.base_dataset = dataset.dataset
            self.indices = dataset.indices
        else:
            self.base_dataset = dataset
            self.indices = list(range(len(dataset)))
            
        self.P = int(num_ids_per_batch)
        self.K = int(num_instances)
        self.allow_id_reuse = bool(allow_id_reuse)
        self.min_modal_coverage = float(min_modal_coverage)
        self.include_text = bool(include_text)

        # 预索引：pid -> { 'vis': [idx...], 'nonvis': [idx...] }
        self.pid_to_mod_idxs = {}
        self.pids = set()

        # 获取person_id的通用函数
        def get_pid(i):
            if hasattr(self.base_dataset, 'data_list') and i < len(self.base_dataset.data_list):
                return int(self.base_dataset.data_list[i]['person_id'])
            else:
                try:
                    sample = self.base_dataset[i]
                    if isinstance(sample, dict):
                        return int(sample.get('person_id', -1))
                except Exception:
                    pass
            return -1

        # 建立索引映射
        for subset_idx, orig_idx in enumerate(self.indices):
            pid = get_pid(orig_idx)
            if pid < 0:
                continue
            self.pids.add(pid)
            
            # 使用改进的模态检测
            mods_img = infer_modalities_of_sample(self.base_dataset, orig_idx, include_text=False)
            mods_txt = infer_modalities_of_sample(self.base_dataset, orig_idx, include_text=True)

            has_vis = ('vis' in mods_img)
            has_nonvis = bool((mods_img & {'nir','sk','cp'}) or ('text' in mods_txt))

            d = self.pid_to_mod_idxs.setdefault(pid, {'vis': [], 'nonvis': []})
            if has_vis:
                d['vis'].append(subset_idx)
            if has_nonvis:
                d['nonvis'].append(subset_idx)

        # 过滤可强配对的ID
        self.strong_ids = [pid for pid, d in self.pid_to_mod_idxs.items()
                           if len(d['vis']) > 0 and len(d['nonvis']) > 0]

        # 回退ID（仅有一种侧的，用于兜底补齐K）
        self.soft_ids = [pid for pid in self.pids if pid not in self.strong_ids]

        if len(self.strong_ids) < self.P:
            print(f"[警告] 可配对ID数({len(self.strong_ids)}) < 每批需要ID数({self.P})，将使用回退ID做兜底。")

        # 估算长度（粗略）：每个强ID能支撑的实例对数
        est_pairs = sum(min(len(self.pid_to_mod_idxs[pid]['vis']),
                            len(self.pid_to_mod_idxs[pid]['nonvis'])) for pid in self.strong_ids)
        self._len_est = max(1, est_pairs // (self.P * self.K)) if self.P * self.K > 0 else 1

        import logging
        logging.info("==================================================")
        logging.info("数据集采样能力分析(基于采样器视角)")
        logging.info(f"  强配对ID数: {len(self.strong_ids)} / {len(self.pids)}")
        logging.info(f"  估算可生成batch数: ~{self._len_est}")
        logging.info("==================================================")

    def __len__(self):
        # 允许 id 复用时，len 用估算值；不允许时按可用资源计算
        return int(self._len_est) if self.allow_id_reuse else max(1, len(self.strong_ids) // self.P)

    def sample_strong_pair(self, pid, retry_count=0):
        """
        尝试为指定ID采样强配对样本
        
        Args:
            pid: 人员ID
            retry_count: 当前重试次数
            
        Returns:
            list: 采样的索引列表，如果失败返回None
        """
        b = self.pid_to_mod[pid]
        chosen = []
        
        # 尝试采样1个vis + 1个非vis
        if len(b['vis']) > 0 and len(b['nonvis']) > 0:
            # 采样1个vis
            chosen.append(self.rng.choice(b['vis']))
            
            # 采样1个非vis（避免重复）
            nonvis_candidates = [x for x in b['nonvis'] if x not in chosen]
            if nonvis_candidates:
                chosen.append(self.rng.choice(nonvis_candidates))
            else:
                chosen.append(self.rng.choice(b['nonvis']))
            
            # 补齐到K个样本
            pool = self.pid_to_indices[pid][:]
            self.rng.shuffle(pool)
            j = 0
            while len(chosen) < self.num_instances and j < 10 * len(pool):
                idx = pool[j % len(pool)]
                if idx not in chosen:
                    chosen.append(idx)
                j += 1
            
            # 若实在不够，允许重复
            while len(chosen) < self.num_instances:
                chosen.append(self.rng.choice(self.pid_to_indices[pid]))
            
            return chosen
        
        # 重试逻辑
        if retry_count < self.retry_limit:
            return self.sample_strong_pair(pid, retry_count + 1)
        
        return None

    def sample_fallback(self, pid):
        """
        软退路：普通K样本采样
        
        Args:
            pid: 人员ID
            
        Returns:
            list: 采样的索引列表
        """
        pool = self.pid_to_indices[pid][:]
        self.rng.shuffle(pool)
        
        chosen = []
        j = 0
        while len(chosen) < self.num_instances and j < 10 * len(pool):
            idx = pool[j % len(pool)]
            if idx not in chosen:
                chosen.append(idx)
            j += 1
        
        # 若实在不够，允许重复
        while len(chosen) < self.num_instances:
            chosen.append(self.rng.choice(self.pid_to_indices[pid]))
        
        return chosen

    def __iter__(self):
        """
        guide18改进的迭代器：强配对 + ID重用策略
        """
        import random
        strong_pool = list(self.strong_ids)
        soft_pool = list(self.soft_ids)

        while True:
            # 选择当前batch要用的ID
            if len(strong_pool) >= self.P:
                cur_ids = random.sample(strong_pool, self.P) if not self.allow_id_reuse else random.choices(strong_pool, k=self.P)
            else:
                # 不足 P 个强ID，用 soft 补齐
                need = self.P - len(strong_pool)
                fillers = (random.sample(soft_pool, min(need, len(soft_pool))) if not self.allow_id_reuse
                           else random.choices(soft_pool, k=need)) if soft_pool else []
                cur_ids = list(strong_pool) + fillers
                if len(cur_ids) < self.P:
                    break

            batch_indices = []
            for pid in cur_ids:
                d = self.pid_to_mod_idxs.get(pid, {'vis': [], 'nonvis': []})
                vis_pool = d['vis'] if d['vis'] else d['nonvis']  # 兜底
                nonvis_pool = d['nonvis'] if d['nonvis'] else d['vis']

                # 每个ID取 K//2 vis + K//2 nonvis（奇数时多给一个nonvis）
                k_vis = self.K // 2
                k_nonvis = self.K - k_vis
                pick = []
                if len(vis_pool) >= k_vis:
                    pick += random.sample(vis_pool, k_vis)
                else:
                    pick += random.choices(vis_pool or nonvis_pool, k=k_vis)
                if len(nonvis_pool) >= k_nonvis:
                    pick += random.sample(nonvis_pool, k_nonvis)
                else:
                    pick += random.choices(nonvis_pool or vis_pool, k=k_nonvis)
                batch_indices.extend(pick)

            if len(batch_indices) != self.P * self.K:
                # 保护：不完整批直接跳过
                continue

            yield batch_indices

            # 不允许复用则移除已用ID
            if not self.allow_id_reuse:
                for pid in set(cur_ids):
                    if pid in strong_pool:
                        strong_pool.remove(pid)
                    elif pid in soft_pool:
                        soft_pool.remove(pid)
                if len(strong_pool) < 1 and len(soft_pool) < 1:
                    break


class ModalAwarePKBatchSampler_Strict(Sampler):
    """
    Guide19修复：产出"整批索引列表"的 Batch Sampler
    - 强配对：同一ID需有 vis 和 nonvis 侧
    - 每个ID在一个batch内取 K//2 vis + K//2 nonvis（奇数时 nonvis 多1）
    - 允许 ID 复用，避免早停
    """
    def __init__(self, dataset, num_ids_per_batch=4, num_instances=4,
                 allow_id_reuse=True, include_text=True, min_modal_coverage=0.6, **_):
        self.P = int(num_ids_per_batch)  # IDs
        self.K = int(num_instances)      # instances per ID
        self.allow_id_reuse = bool(allow_id_reuse)
        self.include_text = bool(include_text)
        self.min_modal_coverage = float(min_modal_coverage)

        # 兼容 Subset
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            self.base_dataset = dataset.dataset
            self.indices = list(dataset.indices)
        else:
            self.base_dataset = dataset
            self.indices = list(range(len(dataset)))

        # 建立 pid -> {vis:[...], nonvis:[...]}
        self.pid_to_mod_idxs = {}
        self.pids = set()

        def get_pid_by_orig_idx(orig_idx):
            s = None
            if hasattr(self.base_dataset, 'samples'):
                s = self.base_dataset.samples[orig_idx]
            else:
                try:
                    item = self.base_dataset[orig_idx]
                    if isinstance(item, dict):
                        s = item
                    elif isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[-1], dict):
                        s = item[-1]
                except Exception:
                    s = None
            if isinstance(s, dict):
                return int(s.get('person_id') or s.get('pid') or s.get('label') or -1)
            return -1

        for orig_idx in self.indices:
            pid = get_pid_by_orig_idx(orig_idx)
            if pid < 0:
                continue
            self.pids.add(pid)

            mods_img = infer_modalities_of_sample(self.base_dataset, orig_idx, include_text=False)
            mods_all = infer_modalities_of_sample(self.base_dataset, orig_idx, include_text=True)

            has_vis = ('vis' in mods_img)
            has_nonvis = bool(mods_img & {'nir','sk','cp'}) or ('text' in mods_all)

            d = self.pid_to_mod_idxs.setdefault(pid, {'vis': [], 'nonvis': []})
            if has_vis: 
                d['vis'].append(orig_idx)
            if has_nonvis: 
                d['nonvis'].append(orig_idx)

        self.strong_ids = [pid for pid, d in self.pid_to_mod_idxs.items()
                           if len(d['vis']) > 0 and len(d['nonvis']) > 0]
        self.soft_ids = [pid for pid in self.pids if pid not in self.strong_ids]

        # 粗略估算可用 batch 数（仅用来 __len__）
        est_pairs = sum(min(len(self.pid_to_mod_idxs[pid]['vis']),
                            len(self.pid_to_mod_idxs[pid]['nonvis'])) for pid in self.strong_ids)
        self._len_est = max(1, est_pairs // max(1, (self.P * self.K)))

        import logging
        logging.info("==================================================")
        logging.info("数据集采样能力分析(采样器视角)")
        logging.info(f"  强配对ID数: {len(self.strong_ids)} / {len(self.pids)}")
        logging.info(f"  估算可生成batch数: ~{self._len_est}")
        logging.info("==================================================")

    def __len__(self):
        # 允许复用 → 用估算值；不允许复用可按资源上限估计
        return int(self._len_est) if self.allow_id_reuse else max(1, len(self.strong_ids) // self.P)

    def __iter__(self):
        import random
        strong_pool = list(self.strong_ids)
        soft_pool = list(self.soft_ids)

        while True:
            # 选 P 个ID（不足用 soft 或复用补齐）
            if len(strong_pool) >= self.P:
                cur_ids = (random.sample(strong_pool, self.P)
                           if not self.allow_id_reuse else random.choices(strong_pool, k=self.P))
            else:
                need = self.P - len(strong_pool)
                fillers = (random.sample(soft_pool, min(need, len(soft_pool)))
                           if not self.allow_id_reuse else random.choices(soft_pool, k=need)) if soft_pool else []
                cur_ids = list(strong_pool) + fillers
                if not cur_ids:
                    break

            batch_indices = []
            for pid in cur_ids:
                d = self.pid_to_mod_idxs.get(pid, {'vis': [], 'nonvis': []})
                vis_pool = d['vis'] if d['vis'] else d['nonvis']  # 兜底
                nonvis_pool = d['nonvis'] if d['nonvis'] else d['vis']

                k_vis = self.K // 2
                k_nonvis = self.K - k_vis  # 奇数→nonvis 多1
                if len(vis_pool) >= k_vis:
                    batch_indices += random.sample(vis_pool, k_vis)
                else:
                    batch_indices += random.choices(vis_pool or nonvis_pool, k=k_vis)
                if len(nonvis_pool) >= k_nonvis:
                    batch_indices += random.sample(nonvis_pool, k_nonvis)
                else:
                    batch_indices += random.choices(nonvis_pool or vis_pool, k=k_nonvis)

            if len(batch_indices) != self.P * self.K:
                # 防御：不完整 batch 直接跳过
                continue

            yield batch_indices

            if not self.allow_id_reuse:
                for pid in set(cur_ids):
                    if pid in strong_pool:
                        strong_pool.remove(pid)
                    elif pid in soft_pool:
                        soft_pool.remove(pid)
                if len(strong_pool) < 1 and len(soft_pool) < 1:
                    break


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

    # guide17修复: 注入规范字段 'modality'，使用数据集原生名称
    def _norm_one(s):
        if 'modality' in s: 
            v = s['modality']
        elif 'mod' in s:    
            v = s['mod']
        else: 
            # 根据modality_mask推断主要模态
            max_modal = None
            max_mask = -1
            if 'modality_mask' in s and isinstance(s['modality_mask'], dict):
                for mod_name, mask_val in s['modality_mask'].items():
                    mask_val = float(mask_val) if not isinstance(mask_val, bool) else (1.0 if mask_val else 0.0)
                    if mask_val > max_mask:
                        max_mask = mask_val
                        max_modal = mod_name
            if max_modal:
                v = max_modal
            else:
                v = s.get('meta',{}).get('modality', 'vis')  # 默认vis
        return canon_mod(str(v))  # 使用统一的规范化函数

    batch_mods = [_norm_one(s) for s in batch]
    batch_dict['modality'] = batch_mods

    return batch_dict
