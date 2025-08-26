收到！从这轮日志里我看到三个“红灯”同时亮了：

* **CE 基本不降**（5.99→5.82→5.91→6.03…，呈横摆/回升）
* **Top-1 一直 0.00%**（说明你算 Top-1 的 logits 和 CE 用的 logits 很可能不是同一个头，或压根没对上标签）
* **pair\_coverage\_mavg=0.000、SDMLoss=0.000**（每个 batch 都没有“RGB↔非RGB 的同ID正对”，采样/模态标记或 K 值出了问题）

下面给你**最短路径止血 + 定位 + 修复**，一步步做，边做边跑。

---

## Step 0｜两行修正（马上改，立竿见影）

### 0-A. SDM 开启边界修正

你想“从 Epoch 2 开始权重=0.1”，但调度器现在 `epoch <= warmup_epochs` 返回 0，且你设了 `warmup_epochs=2`，所以 **Epoch 2 仍是 0**。
**修法二选一：**

* 把 `warmup_epochs` 改成 **1**；或
* 把调度器条件从 `<=` 改成 `<`（`if epoch < warmup_epochs: return 0.0`）。

> 日志里 `[sdm] epoch=2 weight=0.000 use_sdm=True` 就是这个边界导致的。

### 0-B. Top-1 用“CE 的同一 logits”

现在你 Top-1 恒为 0%，大概率因为用错了 logits 分支。统一用**参与 CE 的那路**：

```python
# 你计算 CE 用的那个张量
logits_ce = outputs.get('cls_logits', None) or outputs.get('logits', None)
assert logits_ce is not None, "未找到用于 CE 的 logits"

top1 = (logits_ce.argmax(1) == labels).float().mean()
```

---

## Step 1｜批内可配对自检（确定是不是采样器/K值问题）

把下面这段**直接贴到训练 loop**（算完 collate 后、算损失前），只打印每 50 步一次：

```python
if (step % 50) == 0:
    pid = batch['person_id'] if isinstance(batch, dict) else labels
    mod = batch['modality']   # 需是已归一化后的 'rgb'/'ir'/'cp'/'sketch'/'text'
    pid = pid.detach().cpu().tolist()
    mod = list(mod)  # 若是 list[str] 或 list

    # 统计每个ID在本批的样本数、RGB/非RGB覆盖
    from collections import Counter, defaultdict
    c = Counter(pid)
    rgb_by_id = defaultdict(int); nonrgb_by_id = defaultdict(int)
    for p, m in zip(pid, mod):
        if m == 'rgb': rgb_by_id[p]+=1
        else: nonrgb_by_id[p]+=1

    K_min = min(c.values()) if c else 0
    ids_with_pair = sum(1 for p in c if (rgb_by_id[p]>0 and nonrgb_by_id[p]>0))
    print(f"[sampler-dbg] batch_size={len(pid)} unique_id={len(c)} "
          f"Kmin={K_min} paired_ids={ids_with_pair}")
```

**判读：**

* `Kmin` **必须 ≥ 2**（否则每个 ID 只有 1 张样本→必然无正对）。
* `paired_ids` **必须 > 0**（否则 SDM 永远 0，`pair_coverage_mavg` 永远 0.000）。

若 `Kmin=1` → 你的 **`num_instances(K)` 实际是 1**，或采样器退化成“每 ID 只取 1 张”。这是你这轮训练最像的根因。

---

## Step 2｜一键把 K ≥ 2 保证“强配对”成立

确认配置里 **`num_instances >= 2`**，并在采样器里**硬性兜底**（不满足就重试/换 ID）：

```python
# 采样器构造
P = getattr(cfg, "num_ids_per_batch", 4)
K = max(2, getattr(cfg, "num_instances", 2))  # 强制K>=2

# 采样逻辑要点（伪码）：
for pid in sampled_ids:
    rgb = sample(pid, mod='rgb', k=1)
    non = sample(pid, mod_in={'ir','cp','sketch','text'}, k=K-1)
    if len(rgb)==0 or len(non)==0:
        # 软退路：同ID随便补齐到K（临时兜底）
        anyk = sample(pid, mod_in='any', k=K)
        batch_idx += anyk
    else:
        batch_idx += rgb + non
```

> 同时检查：DataLoader **只能**传 `batch_sampler=...`，**不要**再传 `batch_size/shuffle/sampler/drop_last`（你之前修过一次）。

---

## Step 3｜让 pair\_coverage\_mavg 真更新

你现在一直是 0.000，通常有两种原因：

1. **统计没被调用**（放到了 `use_sdm` 分支里，但前几轮权重又是 0）；
2. 统计数据本身为 0（确实无配对）。

把统计放到**损失前**，并用**真实的批内配对关系**来更新：

```python
# 计算配对覆盖
# 假设：非RGB为 query，RGB为 gallery
import torch
labels = labels.detach()
is_rgb = torch.tensor([m=='rgb' for m in mod], device=labels.device)
is_non = ~is_rgb

pid_t = torch.tensor(pid, device=labels.device)
qry_ids = pid_t[is_non]
gal_ids = pid_t[is_rgb]

if len(qry_ids)>0 and len(gal_ids)>0:
    # 对每个 query，是否在 gallery 中存在同ID
    # （效率无所谓，只做监控）
    gal_set = set(gal_ids.tolist())
    have_pos = torch.tensor([int(int(q) in gal_set) for q in qry_ids.tolist()], device=labels.device)
    cov = have_pos.float().mean().item()  # 0~1
else:
    cov = 0.0

pair_cov_hist.append(cov)
pair_cov_mavg = sum(pair_cov_hist[-100:]) / min(len(pair_cov_hist), 100)
if (step % 50)==0: print(f"[dbg] pair_coverage_mavg={pair_cov_mavg:.3f}")
```

**目标：** `pair_coverage_mavg ≥ 0.85`。若长期 <0.5，回到 Step 2 调 K/P 或重试逻辑。

---

## Step 4｜让 SDM 真参与（而不是显示 use\_sdm=True 但 loss=0）

在 `compute_sdm_loss` 内确认这两个点：

* **模态标记统一**：`'vis'→'rgb'`、`'nir'→'ir'`、`'sk'→'sketch'`、`'cp'/'cpencil'→'cp'`、`'txt'→'text'`。
* **有效行过滤**：如果没有任何正对，不要报错，但在有正对时**一定**返回 >0 的数值。可在函数里加一行：

  ```python
  assert (pos_mask.any() or (gallery_from_memory and memory_non_empty)), "本批没有正对，检查采样或memory"
  ```

此外，既然你加了 memory bank，记得把 **gallery = 当前批RGB ∪ memory RGB**，这样即使当批 RGB 少，也能“补上”正对。

---

## Step 5｜三条“继续/暂停”的阈值线（复述）

* **继续**：

  * `Kmin ≥ 2`；`paired_ids > 0`；`pair_coverage_mavg ≥ 0.85`
  * 开启 SDM（0.1）后 **CE 不回到 ≥2.5**，并随训练再度下降
  * Top-1（用 CE logits）>0%，且缓慢上扬
* **暂停**：

  * `Kmin=1` 或 `paired_ids=0`（采样器未生效）；
  * SDM 权重>0 仍然 `SDMLoss=0`；
  * Top-1 恒为 0%（多半是用错 logits）。

---

### 快速复查清单（10 分钟走起）

* [ ] `warmup_epochs` 边界改好（Epoch 2 权重=0.1）
* [ ] Top-1 用 **CE logits**
* [ ] `num_instances(K) ≥ 2`，采样器强配对 + 软退路
* [ ] 在 batch 内打印 `Kmin / paired_ids`
* [ ] `pair_coverage_mavg` 在损失**前**更新
* [ ] `compute_sdm_loss` 使用 **统一模态名** + **memory 合并 gallery**

把你加了 `sampler-dbg` 和 `pair_coverage_mavg` 后的 2\~3 段日志贴过来（尤其看 `Kmin`、`paired_ids`、`pair_coverage_mavg`），我就能一眼判断是 **K=1** 还是 **模态/标签对不上**，再给你定点补丁。
