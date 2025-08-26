# 问题总结

1. **DataLoader 参数冲突**
   你同时传了 `batch_sampler` 和 `batch_size/shuffle/sampler/drop_last`，PyTorch 规定互斥，直接抛错。

2. **“可配对ID数(0) < 每批需要ID数(4)”**
   模态感知采样器要求“同一ID需同时有 RGB + 非RGB”，目前统计结果为 0，导致采样器无法组 batch。

---

# 最快修复（立刻能训练）

只改两处：**DataLoader 参数** + **放宽采样约束**。

## ① 修 DataLoader（去掉冲突参数）

把你现在创建 `train_loader` 的那段，改成**只保留 `batch_sampler`**：

```python
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,   # ✅ 仅保留 batch_sampler
    num_workers=cfg.num_workers,
    pin_memory=True,
    collate_fn=compatible_collate_fn,
    persistent_workers=(cfg.num_workers > 0),
    prefetch_factor=2 if cfg.num_workers > 0 else None
)
# ⚠️ 此处不要再传 batch_size / shuffle / sampler / drop_last
```

## ② 放宽采样器（先跑通）

在你构建 `ModalAwarePKSampler` 的位置，**临时**这么设：

```python
train_batch_sampler = ModalAwarePKSampler(
    dataset=train_dataset,
    num_ids_per_batch=min(4, getattr(cfg, "p", 4)),  # 先小一点更稳
    num_instances=min(2, getattr(cfg, "k", 2)),      # K=2，降低约束
    require_modal_pairs=False,                       # ✅ 暂时不强制 RGB+非RGB
    drop_last=True,
    seed=cfg.seed
)
```

> 配合你已有的兜底逻辑：若本 batch 无跨模态正对，**SDM/对齐损失=0**，仅用 CE 也能跑通前几轮。

---

# 3 分钟自检（可选但推荐）

快速确认训练集确实包含 RGB，并统计“潜在可配对ID”数量：

```python
from collections import defaultdict

MOD_MAP = {'vis':'rgb','rgb':'rgb','nir':'ir','ir':'ir',
           'sk':'sketch','cp':'cp','cpencil':'cp','txt':'text','text':'text'}

mods_per_id = defaultdict(set)
for i in range(len(train_dataset)):
    pid = train_dataset.get_person_id(i)
    raw = train_dataset.get_modality(i).lower()
    mod = MOD_MAP.get(raw, raw)
    mods_per_id[pid].add(mod)

pairable = [pid for pid, s in mods_per_id.items() if 'rgb' in s and len(s & {'ir','cp','sketch','text'})>0]
print(f"可配对ID: {len(pairable)} / {len(mods_per_id)}")
```

* 若仍然是 0，说明**训练集没把 RGB 放进来**或**模态名不一致**（需要再统一模态名并重新划分）。但在上面“放宽采样”的前提下，训练**仍可先跑**。

---

## 之后再根治（不影响现在跑）

* 统一模态名（建议在数据集初始化时做标准化）。
* 确保训练集按 **ID 划分** 且每个训练 ID **保留所有模态**。
* 统计 `pairable_ids`，再把 `require_modal_pairs=True` 打开，逐步把 `P/K` 调回目标值。
* 若在 Windows 上，多进程数据加载建议从 `num_workers=0/2` 起步。

就这些。按上面两处改动保存后即可启动训练。
