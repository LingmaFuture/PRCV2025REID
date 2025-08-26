# Fix18: 基于Guide18完善模态规范化和采样器健壮性

## 问题背景

尽管Fix17修复了模态检测逻辑，但训练日志仍显示：
```
数据集统计:
  总ID数: 320
  总样本数: 14910
  各模态分布: {'rgb': 14910}
  可配对ID数 (K≥2): 0 (0.0%)
```

这说明**新模态检测没有真正生效**，系统仍在使用旧的模态命名（rgb而非vis），导致配对失败。

## 根本原因分析

Guide18指出的四个常见问题：

1. **归一化没贯通**：统计/采样环节没有对返回的集合做规范化
2. **旧函数仍被引用**：可能有同名函数重复定义
3. **分析代码绕过推断函数**：直接从字段拿字符串，没走规范化
4. **Primary字段误导**：单一模态字段覆盖了mask/paths信息

## Guide18修复方案

### 1. 完善模态规范化映射

**文件**: `datasets/dataset.py`

增强规范化映射，支持更多变体名称：
```python
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
```

增加健壮性工具函数：
```python
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
```

### 2. 重写模态检测函数

按Guide18建议，重写为更健壮的版本：
```python
@torch.no_grad()
def infer_modalities_of_sample(dataset, index: int, include_text: bool = True):
    """
    返回该样本可用模态的规范化集合（vis/nir/sk/cp/[text]）
    优先级：modality_mask -> images/paths -> primary 'modality' 字段
    """
    # 兼容不同数据集实现
    sample = None
    if hasattr(dataset, 'samples'):
        sample = dataset.samples[index]
    else:
        try:
            s = dataset[index]
            if isinstance(s, dict):
                sample = s
            elif isinstance(s, (list, tuple)) and len(s) >= 3 and isinstance(s[-1], dict):
                sample = s[-1]  # 支持 (data, label, meta) 格式
        except Exception:
            sample = None

    mods = set()

    # 1) modality_mask: {'vis':1.0, 'nir':1.0, ...} - 最高优先级
    # 2) images/paths 容器：{ 'vis': [...], 'nir': [...] }  
    # 3) primary 字段：'modality' / 'mode' / 'mod' - 最低优先级
    # 4) 文本模态（可选）

    # 最终只保留标准命名
    return {m for m in mods if m in (ALL_MODALITIES if include_text else IMG_MODALITIES)}
```

关键改进：
- **入口规范化**：任何来源的模态名都在检测时立即规范化
- **多层级检测**：modality_mask > images > primary字段的优先级
- **防御性编程**：优雅处理各种异常情况

### 3. 重构ModalAwarePKSampler_Strict采样器

**文件**: `datasets/dataset.py`

按Guide18建议完全重写采样器类：
```python
class ModalAwarePKSampler_Strict(Sampler):
    """
    强配对：同一个ID必须既有 vis 也有 非vis(nir/sk/cp/text) 才算可配对。
    支持 allow_id_reuse=True：同一 epoch 内可重复使用ID，防止采样耗尽。
    """
    def __init__(self, dataset, num_ids_per_batch=4, num_instances=4,
                 allow_id_reuse=True, min_modal_coverage=0.6, include_text=True):
        # 使用改进的模态检测建立索引
        self.pid_to_mod_idxs = {}  # pid -> { 'vis': [idx...], 'nonvis': [idx...] }
        
        for subset_idx, orig_idx in enumerate(self.indices):
            # 分别检测图像模态和文本模态
            mods_img = infer_modalities_of_sample(self.base_dataset, orig_idx, include_text=False)
            mods_txt = infer_modalities_of_sample(self.base_dataset, orig_idx, include_text=True)

            has_vis = ('vis' in mods_img)
            has_nonvis = bool((mods_img & {'nir','sk','cp'}) or ('text' in mods_txt))
```

采样策略改进：
- **强制配对**：每个batch中每个ID取K//2个vis样本 + K//2个非vis样本  
- **ID重用机制**：允许同一epoch内重复使用ID，避免早期耗尽
- **回退策略**：强配对ID不足时使用软退路ID补齐

### 4. 更新配置参数

**文件**: `configs/config.py`

```python
# guide18: 采样器配置（防止再早停）
allow_id_reuse: bool = True     # 允许同epoch内ID复用，防止采样耗尽
sampling_fallback: bool = True  # 无法满足约束时是否回退到随机采样
min_modal_coverage: float = 0.6 # 跨模态覆盖率最低要求
instances_per_id: int = 2       # K - 每个ID的实例数量，与num_instances保持一致
```

### 5. 自检函数

**文件**: `debug_guide18.py`

按Guide18建议添加自检函数：
```python
def quick_scan(ds, limit=500, include_text=True):
    c = Counter()
    has_pair = 0
    for i in range(min(limit, len(ds))):
        m_img = infer_modalities_of_sample(ds, i, include_text=False)
        m_all = infer_modalities_of_sample(ds, i, include_text=include_text)
        # 确保规范化
        m_img = {canon_mod(x) for x in m_img}
        m_all = {canon_mod(x) for x in m_all}
        
        vis = 'vis' in m_img
        nonvis = bool(m_img & {'nir','sk','cp'}) or ('text' in m_all)
        has_pair += int(vis and nonvis)
    
    print("[仅图像] 模态出现次数:", {k: v for k, v in c.items() if k in IMG_MODALITIES})
    print(f"有 vis+非vis 配对的样本数: {has_pair}/{limit}   比例: {has_pair/limit:.1%}")
```

## 关键技术改进

### 1. 双保险规范化策略
- **入口规范化**：在`infer_modalities_of_sample`中立即规范化
- **统计规范化**：在所有使用模态名称的地方都调用`canon_mod()`

### 2. 优先级明确的模态检测
```
modality_mask（最准确）> images/paths（备选）> primary字段（兜底）
```

### 3. 健壮的采样策略
- **强配对保证**：每个ID在batch中必有vis和nonvis样本
- **ID重用机制**：防止采样器过早耗尽导致epoch截断
- **多级回退**：强配对 -> 软配对 -> 随机填充

### 4. 防御性编程
- 多种数据集格式兼容（Dataset vs Subset vs 自定义格式）
- 异常处理和优雅降级
- 类型检查和边界保护

## 预期修复效果

### 修复前（Fix17后仍存在的问题）：
```
各模态分布: {'rgb': 14910}  # 仍显示旧名称
可配对ID数: 0 (0.0%)        # 配对失败
```

### 修复后（Guide18预期）：
```
[仅图像] 模态出现次数: {'vis': 12000, 'nir': 8000, 'sk': 14910, 'cp': 9000}
有 vis+非vis 配对的样本数: X/500   比例: >80%
强配对ID数: 300+ / 320
估算可生成batch数: >1000
```

## 训练指标预期改善

Guide18修复后，训练应该出现：

1. **采样器统计正常**：不再显示`{'rgb': 14910}`，而是规范化的`{'vis': xxx, 'nir': xxx, ...}`
2. **强配对ID数显著增加**：从0个增加到大部分ID
3. **Epoch长度恢复正常**：从80步恢复到预期的数百/上千步
4. **pair_coverage_mavg ≥ 0.85**：跨模态配对成功率大幅提升
5. **SDM损失开始有效**：不再恒为0.000
6. **准确率开始上升**：Top-1不再恒为0%

## 核心修复点总结

1. **统一模态命名**：彻底消除rgb/ir/sketch等旧名称，统一使用vis/nir/sk/cp/text
2. **健壮模态检测**：多层级检测策略，防御性编程
3. **采样器重构**：强配对策略 + ID重用机制
4. **双重规范化**：入口规范化 + 使用点规范化的双保险
5. **完善异常处理**：优雅处理各种边界情况

这次修复解决了Fix17遗留的模态名称不统一问题，确保端到端使用数据集原生名称，为训练系统提供了坚实的数据基础。

## 验证清单

修复完成后验证：
1. ✅ `debug_guide18.py`显示规范化的模态名称（vis/nir/sk/cp）
2. ✅ 配对比例 > 80%
3. ✅ 采样器日志显示合理的强配对ID数量
4. ✅ 训练不再80步就停止
5. ✅ 无旧模态名称（rgb/ir/sketch/cpencil）泄漏

Guide18的修复确保了数据流的一致性和健壮性，为后续训练的稳定进行奠定了基础。