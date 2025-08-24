结论先说：**不是模型坏了，也不是数据集坏了**（划分/ID范围都正常）。这些“⚠️ 无正样本”基本来自**训练流程里的“批内正样本构造失败”**——也就是**采样器/批次组织**没保证“同一 ID 在同一批里既有 RGB（当作 gallery），又有非 RGB（当作 query）”。你前一步为了跑通把 `require_modal_pairs=False` 放宽了，所以现在 SDM 仍在尝试对齐，但经常找不到正样本行，日志就刷屏了。

下面给你**最快三步修复**，按顺序做就行：

---

## ✅ 最快止血（立刻能稳跑）

1. **前 1–3 个 epoch 关闭 SDM**（或按行过滤后再算）

   * 配置里临时设：`sdm_weight = 0.0`（或在计算 SDM 前对 “无正样本行” 过滤，没有有效行就 return 0，不打印警告）。
   * 先用 **CE 预热**，等分类器有基本判别力再开 SDM，训练会稳定很多。

2. **把警告降噪**

   * 日志里把“无正样本行”的逐行打印改成**每 batch 汇总**（如 `missing_rows_cnt`），或仅在 `epoch % 5 == 0` 打印一次示例。这不影响训练，但能避免干扰判断。

3. **降低批内约束，减小 P/K**

   * 例如先用 `P=4, K=2`（或更小），批次更容易凑齐正对，吞吐也会上去。

---

## 🛠 根治方案（开回 SDM 且有效）

1. **恢复“模态配对采样”**

   * 打开：`require_modal_pairs=True`。
   * 但要先确保采样器只从**可配对 ID**里取（该 ID 同时拥有 `rgb` 和任一非 `rgb` 模态）。
   * 统一模态名映射（放在 dataset 最早处）：

     ```python
     MOD_MAP = {'vis':'rgb','rgb':'rgb','nir':'ir','ir':'ir',
                'sk':'sketch','cp':'cp','cpencil':'cp','txt':'text','text':'text'}
     ```

   * 统计并缓存：

     ```python
     pairable_ids = [pid for pid, s in mods_per_id.items()
                     if ('rgb' in s) and (len(s & {'ir','cp','sketch','text'})>0)]
     ```

   * 采样器从 `pairable_ids` 里选 P 个 ID，对每个 ID 强制采到**至少1张 RGB + 1张非RGB**（若不足就重采该 ID 或换 ID）。

2. **批内构造检查（30秒自测）**
   在 `collate_fn` 之后加一个极简断言/统计：

   * 统计该批里每个 `person_id` 的模态集合；
   * 若某个 `pid` 出现在非 RGB，但没有 RGB，则记为“未配对”；
   * 未配对率 > 10% 就减小 `P/K` 或提升 `pairable_ids` 覆盖。

3. **没有配对时的 SDM 处理**

   * 你现在已经能跳过/置 0；保持即可。
   * 真要提升 SDM 覆盖度，可加一个**跨批记忆库（memory bank）**：缓存近 N 个 step 的 **RGB(特征, label)**，当前 batch 的非 RGB 可以去 bank 里找正样本，这样就不强依赖“同批必配对”。

---

## 🎯 为何不是 dataset / model 的锅？

* **dataset**：划分打印正常（320/80、无重叠），`labels 范围: 14-399` 也符合随机抽样；
* **model**：初始 `CE≈5.991` 正常（=log(400)），仅 `bn_neck.classifier` 可训练也合理做法；
* **症状直指采样/批组织**：SDM 是“跨模态同 ID 对齐”的损失，批内若没有“同 ID 的 RGB 与非 RGB”就会出现“无正样本行”。

---

## 🔧 两段关键小改（可直接落地）

**A. SDM 前过滤（没有有效行就返回 0，不刷屏）**

```python
pos_mask = (qry_labels.view(-1,1) == gal_labels.view(1,-1))
valid_rows = pos_mask.any(dim=1)
if not valid_rows.any():
    return torch.zeros([], device=qry.device, dtype=qry.dtype)  # 无声返回 0
# 只对 valid_rows 计算 SDM
```

**B. 采样器硬性配对（伪代码要点）**

```python
# 每个 batch:
ids = random.sample(pairable_ids, P)
batch_indices = []
for pid in ids:
    rgb_idx = sample_indices(pid, modality='rgb', k=1)
    nonrgb_idx = sample_indices(pid, modality_in={'ir','cp','sketch','text'}, k=K-1)
    if len(rgb_idx)==0 or len(nonrgb_idx)==0:
        # 换一个 id 或者重试本 id，直到满足
        continue
    batch_indices += rgb_idx + nonrgb_idx
yield batch_indices  # size ~= P*K
```

---

## 额外小建议（数值稳定）

* `Feat(BN)=22.5` 偏大：考虑启用 **特征范数约束** 或在 BNNeck 后做 L2 normalize，目标范数控制在 \~8–10；
* 把 `tau/temperature` 设 0.15–0.2，避免相似度过“尖”；
* Windows 上 `num_workers` 先用 0/2，`persistent_workers=True` 仅在 `num_workers>0` 时开启。

---

**一句话总结**：现在的告警主要是**批内缺少跨模态同 ID 的正样本**；先用 **CE 预热 + 过滤无效行** 快速稳定训练，然后按上面的“**可配对 ID + 强制每 ID 至少 1 RGB + 1 非 RGB**”把采样器修好，再重开 SDM，效果就上来了。
