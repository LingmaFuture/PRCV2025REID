看到了：现在卡在 `KeyError: 'modality'`。也就是**你在 `train_epoch` 里取 `batch['modality']`，但 collate 出来的 batch 根本没有这个键**（有可能叫 `modalities`/`mod`/`modality_id` 等）。

给你两条最快修复路径，任选其一（建议 A+断言一起做，最稳）。

---

## 方案 A（推荐）：在 `train_epoch` 里做**健壮取模态**的小工具

把下面两段粘进去：一段是**模态名归一化**，一段是**从 batch 中鲁棒提取模态**。之后统一用它来算 `pair_coverage`、`sampler-dbg` 等。

```python
# === 放在 train_epoch 顶部或一个utils里 ===
MOD_MAP = {
    'vis':'rgb','rgb':'rgb',
    'nir':'ir','ir':'ir',
    'sk':'sketch','sketch':'sketch',
    'cp':'cp','cpencil':'cp',
    'txt':'text','text':'text'
}
ID2MOD = {0:'rgb', 1:'ir', 2:'cp', 3:'sketch', 4:'text'}

def _extract_modalities_from_batch(batch):
    """
    返回标准化后的模态名列表（长度等于batch大小），元素 ∈ {'rgb','ir','cp','sketch','text'}
    兼容多种字段：'modality' | 'modalities' | 'mod' | 'modality_id' 等
    """
    if isinstance(batch, dict):
        if 'modality' in batch:
            raw = batch['modality']
        elif 'modalities' in batch:
            raw = batch['modalities']
        elif 'mod' in batch:
            raw = batch['mod']
        elif 'modality_id' in batch:  # tensor/list of ints
            ids = batch['modality_id']
            if hasattr(ids, 'tolist'): ids = ids.tolist()
            raw = [ID2MOD.get(int(i), str(i)) for i in ids]
        else:
            # 最后兜底：如果每个样本在 batch['meta'] 里
            if 'meta' in batch and isinstance(batch['meta'], list) and len(batch['meta'])>0:
                raw = [m.get('modality') or m.get('mod') for m in batch['meta']]
            else:
                raise KeyError("Batch has no modality-like key: expected one of "
                               "['modality','modalities','mod','modality_id','meta[*].modality']")
    else:
        raise TypeError("Batch must be a dict-like object with modality info")

    # 统一成 list[str]
    if not isinstance(raw, list):
        if hasattr(raw, 'tolist'):  # torch tensor
            raw = raw.tolist()
        else:
            raw = list(raw)

    # 归一化到标准模态名
    mods = [MOD_MAP.get(str(x).lower(), str(x).lower()) for x in raw]
    return mods
```

然后把你原来写的：

```python
mod = batch['modality']
```

替换为：

```python
mod = _extract_modalities_from_batch(batch)
```

再加一条**轻断言**，避免后面又栽坑：

```python
assert len(mod) == (labels.shape[0] if hasattr(labels, 'shape') else len(batch.get('person_id', []))), \
    f"mod length {len(mod)} != batch size"
```

> 这样不需要立刻改你的 `collate_fn`，先保证训练能继续跑；后面再把字段彻底统一。

---

## 方案 B：一次性修 `compatible_collate_fn`，**永远产出标准键**

在 `compatible_collate_fn` 里，收集每个样本的模态并**注入一个规范字段**：

```python
def compatible_collate_fn(samples):
    # ... 你已有的拼接逻辑 ...
    # 假设每个 sample 是 dict，含 'modality'/'mod'/'modality_id'
    MOD_MAP = {'vis':'rgb','rgb':'rgb','nir':'ir','ir':'ir',
               'sk':'sketch','cp':'cp','cpencil':'cp','txt':'text','text':'text'}
    ID2MOD = {0:'rgb',1:'ir',2:'cp',3:'sketch',4:'text'}

    def _norm_one(s):
        if 'modality' in s: v = s['modality']
        elif 'mod' in s:    v = s['mod']
        elif 'modality_id' in s: v = ID2MOD.get(int(s['modality_id']), s['modality_id'])
        else: v = s.get('meta',{}).get('modality', None)
        return MOD_MAP.get(str(v).lower(), str(v).lower())

    batch_mods = [_norm_one(s) for s in samples]
    batch['modality'] = batch_mods
    return batch
```

> 做了 B 之后，你的 `train_epoch` 就可以放心继续使用 `batch['modality']`，不会再 KeyError。

---

## 再加两处小自检（强烈推荐）

1. **打印一次 batch keys**（只在 step==0 打）：

```python
if step == 0:
    print(f"[dbg] batch keys: {list(batch.keys())[:12]}")
```

2. **采样器自检输出（每 50 step）**：
   配合我之前给你的 `sampler-dbg` 和 `pair_coverage_mavg` 代码，确认：

* `Kmin ≥ 2`
* `paired_ids > 0`
* `pair_coverage_mavg` 很快从 0.000 升起来（目标 ≥0.85）

---

## 常见关联坑（顺手检查）

* **`person_id` 键名**：你的 `sampler-dbg` 用到了 `batch['person_id']`，确保 collate 后就是这个名字（不是 `pid`/`label`）。
* **张量/列表维度**：如果 `labels` 是 tensor、`mod` 是 list，要注意长度一致；别处若做了 `repeat_interleave`，也要同步更新模态列表。
* **文本样本**：若 batch 里存在纯文本 query，确保 `modality == 'text'` 也能走到你的配对/覆盖率统计（别被过滤掉）。

---

## 一句话结论

这次报错仅仅是**字段名不统一**。按方案 A 先在 `train_epoch` 做鲁棒提取，**立刻可跑**；随后再把 `collate_fn` 统一标准键，彻底消灭同类问题。跑起来后，把 `sampler-dbg / pair_coverage_mavg` 的三项数值贴我，我再帮你确认采样器是否真正“强配对”生效。
