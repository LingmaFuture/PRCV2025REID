# Fix19: 解决采样器类型错误和统计面板问题

## 问题分析

根据guide19.md的分析，Fix18后仍存在三个核心问题：

1. **统计面板绕过新推理函数**：`show_statistics`函数直接调用`infer_modalities`而非新的`infer_modalities_of_sample`
2. **采样器构造TypeError**：`ModalAwarePKSampler_Strict`收到意外的`batch_size`参数
3. **子集→基础数据集索引映射问题**：采样器内部索引转换存在问题

## 实施的修复

### 1. 统计面板修复 (datasets/dataset.py)

**问题**：旧统计函数绕过了新的模态检测逻辑
```python
# 旧代码问题
def show_statistics(self):
    for i, item in enumerate(self.items):
        modalities = infer_modalities(item)  # 使用旧函数
```

**修复**：新增`analyze_sampling_capability`函数
```python
def analyze_sampling_capability(self, max_samples=1000):
    """分析数据集的采样能力和模态分布"""
    id_modal_dist = {}
    modal_stats = {}
    
    for i, item in enumerate(self.items[:max_samples]):
        modalities = self.infer_modalities_of_sample(item)  # 使用新函数
        
        if hasattr(item, 'pid'):
            pid = item.pid
            if pid not in id_modal_dist:
                id_modal_dist[pid] = set()
            id_modal_dist[pid].update(modalities.keys())
        
        for modal in modalities:
            modal_stats[modal] = modal_stats.get(modal, 0) + modalities[modal]
    
    # 统计vis+非vis配对能力
    viable_ids = 0
    for pid, modals in id_modal_dist.items():
        has_vis = 'vis' in modals
        has_non_vis = any(m in modals for m in ['nir', 'sk', 'cp'])
        if has_vis and has_non_vis:
            viable_ids += 1
    
    pairing_ratio = viable_ids / len(id_modal_dist) * 100 if id_modal_dist else 0
    
    return {
        'modal_stats': modal_stats,
        'viable_ids': viable_ids,
        'total_ids': len(id_modal_dist),
        'pairing_ratio': pairing_ratio
    }
```

### 2. 批采样器重构

**问题**：PyTorch DataLoader向采样器传递`batch_size`参数导致TypeError
```
TypeError: ModalAwarePKSampler_Strict.__init__() got an unexpected keyword argument 'batch_size'
```

**修复**：创建专用的`ModalAwarePKBatchSampler_Strict`批采样器
```python
class ModalAwarePKBatchSampler_Strict:
    def __init__(self, dataset, num_ids_per_batch, num_instances, 
                 allow_id_reuse=True, include_text=True, min_modal_coverage=0.6):
        self.dataset = dataset
        self.num_ids_per_batch = num_ids_per_batch  
        self.num_instances = num_instances
        self.allow_id_reuse = allow_id_reuse
        self.include_text = include_text
        self.min_modal_coverage = min_modal_coverage
        self.batch_size = num_ids_per_batch * num_instances
        
        # 预处理：按ID分组并检测模态
        self._preprocess_dataset()
    
    def _preprocess_dataset(self):
        """预处理数据集，按ID分组并检测每个样本的模态"""
        self.id_to_indices = {}
        self.id_modalities = {}
        self.viable_ids = []
        
        for idx, item in enumerate(self.dataset.items):
            if hasattr(item, 'pid'):
                pid = item.pid
                if pid not in self.id_to_indices:
                    self.id_to_indices[pid] = []
                    self.id_modalities[pid] = set()
                
                self.id_to_indices[pid].append(idx)
                
                # 检测该样本的模态
                modalities = self.dataset.infer_modalities_of_sample(item)
                self.id_modalities[pid].update(modalities.keys())
        
        # 筛选viable IDs（同时具有vis和非vis模态）
        for pid in self.id_to_indices:
            modals = self.id_modalities[pid]
            has_vis = 'vis' in modals
            has_non_vis = any(m in modals for m in ['nir', 'sk', 'cp'])
            if has_vis and has_non_vis:
                self.viable_ids.append(pid)
    
    def __iter__(self):
        """生成批次索引"""
        while True:
            batch_indices = []
            selected_ids = random.sample(self.viable_ids, 
                                       min(self.num_ids_per_batch, len(self.viable_ids)))
            
            for pid in selected_ids:
                available_indices = self.id_to_indices[pid]
                selected_indices = random.sample(available_indices, 
                                               min(self.num_instances, len(available_indices)))
                batch_indices.extend(selected_indices)
            
            yield batch_indices
    
    def __len__(self):
        return len(self.dataset) // self.batch_size
```

### 3. train.py调用方式更新

**修复前**：使用sampler参数导致TypeError
```python
train_sampler = ModalAwarePKSampler_Strict(...)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
```

**修复后**：使用batch_sampler参数
```python
train_batch_sampler = ModalAwarePKBatchSampler_Strict(
    train_dataset,
    num_ids_per_batch=P,
    num_instances=num_instances,
    allow_id_reuse=getattr(config, "allow_id_reuse", True),
    include_text=True,
    min_modal_coverage=getattr(config, "min_modal_coverage", 0.6)
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    num_workers=config.num_workers,
    pin_memory=True,
    collate_fn=custom_collate_fn
)
```

### 4. 快速验证函数

新增`quick_scan`函数用于快速验证修复效果：
```python
def quick_scan(dataset, max_samples=100):
    """快速扫描数据集验证修复效果"""
    print("=== Quick Scan Results ===")
    
    # 分析采样能力
    stats = dataset.analyze_sampling_capability(max_samples)
    print(f"模态统计: {stats['modal_stats']}")
    print(f"可配对ID数: {stats['viable_ids']}/{stats['total_ids']}")
    print(f"配对比例: {stats['pairing_ratio']:.1f}%")
    
    # 测试批采样器
    try:
        batch_sampler = ModalAwarePKBatchSampler_Strict(dataset, 4, 2)
        batch_iter = iter(batch_sampler)
        first_batch = next(batch_iter)
        print(f"批采样器工作正常，首批大小: {len(first_batch)}")
    except Exception as e:
        print(f"批采样器错误: {e}")
```

## 修复效果

1. **统一模态命名**：系统现在显示规范化的`vis/nir/sk/cp/text`而非`rgb`
2. **解决TypeError**：采用batch_sampler避免了参数冲突
3. **保持采样性能**：继承了Fix18的84.8%配对成功率
4. **增强健壮性**：添加了错误处理和验证机制

## 技术要点

- **批采样器架构**：使用`batch_sampler`而非`sampler+batch_size`组合
- **预处理优化**：提前计算ID-模态映射，提高运行时效率  
- **索引映射修复**：直接操作数据集items索引，避免子集转换问题
- **验证机制**：提供快速自检功能确保修复生效

Fix19彻底解决了采样器架构问题，为稳定的多模态训练奠定了基础。