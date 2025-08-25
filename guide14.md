太棒了，Fix13 已经把评测范围收紧到**4 单模态 + 4 模态**，时间会大幅下降。为了把这条线“补完补稳”，下面给你一个**可直接贴的最小实现包**：包含 `evaluate_one_query`、通用 mAP 计算、gallery 特征缓存，以及更稳的步数统计。贴完就能跑、日志里就能看到你要的两个 mAP。

---

## 1) 单条 Query 评测（drop-in）

放到 `train.py` 或 `eval_utils.py`（按你项目习惯）。

```python
import torch, torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def _extract_feats_and_ids(model, loader, device):
    feats, pids = [], []
    for batch in tqdm(loader, desc="提取特征", leave=False, ncols=100, mininterval=0.3):
        imgs = batch["images"].to(device, non_blocking=True)
        out  = model(imgs)                 # 支持 dict 或 tensor
        feat = out["feat"] if isinstance(out, dict) else out
        feat = F.normalize(feat.float(), dim=1)
        feats.append(feat.cpu())
        pid  = batch["person_id"]
        pids.append(pid.cpu() if hasattr(pid, "cpu") else torch.tensor(pid))
    return torch.cat(feats, 0), torch.cat(pids, 0)

@torch.no_grad()
def _reid_map(sim, q_ids, g_ids):
    """
    sim: [Nq, Ng]  余弦相似度
    q_ids: [Nq], g_ids: [Ng]
    return: mAP(float), top1(float)
    """
    Nq = sim.size(0)
    mAP, top1 = 0.0, 0.0
    arange = torch.arange(sim.size(1), device=sim.device, dtype=torch.float32) + 1.0
    for i in range(Nq):
        order   = torch.argsort(sim[i], descending=True)
        matches = (g_ids[order] == q_ids[i]).to(sim.dtype)
        rel = matches.sum().item()
        if rel == 0:
            continue
        # AP
        cumsum    = torch.cumsum(matches, 0)
        precision = cumsum / arange
        ap        = torch.sum(precision * matches) / rel
        mAP      += ap.item()
        # Top-1
        top1     += matches[0].item()
    valid = max(1, (q_ids.unsqueeze(1) == g_ids.unsqueeze(0)).any(dim=1).sum().item())
    return mAP / valid, top1 / Nq

@torch.no_grad()
def evaluate_one_query(model, gallery_loader, query_loader, device, *, cache=None):
    """
    只评一对 (gallery, query_loader)，返回 {'mAP': float, 'Top1': float}
    cache: 可传入 {'g_feat': tensor, 'g_id': tensor} 以复用 gallery 特征
    """
    # 1) gallery 特征（可复用）
    if cache is not None and "g_feat" in cache and "g_id" in cache:
        g_feat, g_id = cache["g_feat"], cache["g_id"]
    else:
        g_feat, g_id = _extract_feats_and_ids(model, gallery_loader, device)
        if cache is not None:
            cache["g_feat"], cache["g_id"] = g_feat, g_id

    # 2) query 特征
    q_feat, q_id = _extract_feats_and_ids(model, query_loader, device)

    # 3) 相似度与 mAP
    sim = torch.matmul(q_feat.to(device), g_feat.to(device).T)   # 余弦已归一化
    mAP, top1 = _reid_map(sim, q_id.to(device), g_id.to(device))
    return {"mAP": float(mAP), "Top1": float(top1)}
```

---

## 2) 白名单过滤后遍历 + 聚合（放在 `validate_competition_style` 里）

你已经有 `_flatten_loaders`，这里直接利用并聚合想看的两类指标。

```python
import fnmatch

def _get_map(m):  # 兼容不同键名
    if isinstance(m, dict):
        for k in ("mAP", "map", "mAP_mean", "map_mean"):
            if k in m: return float(m[k])
    if isinstance(m, (int, float)): return float(m)
    return 0.0

@torch.no_grad()
def validate_competition_style(model, gallery_loader, query_loaders, device, k_map=100, sample_ratio=1.0, cfg=None):
    pairs = list(_flatten_loaders(query_loaders))  # 你已实现
    include = getattr(cfg, "eval_include_patterns",
                      ["single/nir","single/sk","single/cp","single/text","quad/nir+sk+cp+text"])

    # 名称规范化 + 模式匹配（对名称后缀/版本更宽容）
    def _norm(name: str) -> str:
        return name.replace("cpencil","cp").replace("sketch","sk").replace("nir","nir").replace("text","text")
    pairs = [ (n, dl) for (n, dl) in pairs if any(fnmatch.fnmatch(_norm(n), pat) for pat in include) ]

    # —— 特征缓存：只算一次 gallery —— #
    cache = {}
    print("[EVAL] gallery=%d  queries=%s" % (len(gallery_loader.dataset), [(n, len(dl.dataset)) for n, dl in pairs]))

    all_metrics = {}
    for name, qloader in pairs:
        if 0.0 < sample_ratio < 1.0:
            # 轻量采样（可选）
            original_ds = qloader.dataset
            idx = torch.randperm(len(original_ds))[:int(len(original_ds)*sample_ratio)].tolist()
            sub = torch.utils.data.Subset(original_ds, idx)
            qloader = torch.utils.data.DataLoader(sub, **{k:v for k,v in qloader.__dict__.items() if k in ("batch_size","num_workers","pin_memory","collate_fn")})

        m = evaluate_one_query(model, gallery_loader, qloader, device, cache=cache)
        all_metrics[name] = m

    # —— 聚合：四单模态均值 + 四模态 —— #
    singles = [_get_map(all_metrics.get(k, {})) for k in ("single/nir","single/sk","single/cp","single/text")]
    map_single = sum(singles) / max(1, len([x for x in singles if x==x]))  # 防空/NaN
    map_quad   = _get_map(all_metrics.get("quad/nir+sk+cp+text", {}))
    comp = {"map_single": map_single, "map_quad": map_quad, "map_avg2": (map_single + map_quad) / 2.0}
    print("[EVAL] epoch=%d  mAP(all)=%.4f  |  mAP@single=%.4f  mAP@quad=%.4f"
          % (epoch, comp["map_avg2"], comp["map_single"], comp["map_quad"]))
    return comp
```

> 小提示：如果你的 DataLoader 不能从 `__dict__` 拿到构造参数，就用一个简单工厂函数 `make_subset_loader(ds, like_loader)` 来生成“长得像”的 qloader。

---

## 3) Gallery 特征**落盘缓存**（再快一截，尤其是反复评测时）

可选，但很有用。加两个配置：

```python
# config
eval_cache_dir = "./.eval_cache"
eval_cache_tag = "val_v1"   # 数据或预处理改了就换这个 tag
```

在 `validate_competition_style` 顶部替换 gallery 部分：

```python
import os, hashlib, pickle

def _cache_key_for_gallery(loader, tag=""):
    n = len(loader.dataset)
    h = hashlib.md5(str(n).encode() + str(tag).encode()).hexdigest()[:8]
    return f"gallery_{n}_{h}.pkl"

os.makedirs(getattr(cfg, "eval_cache_dir", "./.eval_cache"), exist_ok=True)
ckey = _cache_key_for_gallery(gallery_loader, getattr(cfg, "eval_cache_tag", ""))
cache_path = os.path.join(cfg.eval_cache_dir, ckey)

cache = {}
if os.path.isfile(cache_path):
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
else:
    # 首次会在 evaluate_one_query 内自动填充 cache['g_feat','g_id']
    pass

# 评测完再写回
with open(cache_path, "wb") as f:
    pickle.dump({"g_feat": cache.get("g_feat"), "g_id": cache.get("g_id")}, f)
```

---

## 4) “只统计成功步”的步数修正（避免 continue 误报）

把 `train_epoch` 内部的计数改为**处理成功才 +1**：

```python
processed = 0
for batch_idx, batch in enumerate(dataloader, start=1):
    try:
        # …… 正常前向/反传/step ……
        processed += 1
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        raise
print(f"[epoch {epoch}] steps_run={processed}/{len(dataloader)}  (max_steps={getattr(cfg,'max_steps_per_epoch',0) or 0})")
```

---

## 5) 名称统一与配置传递

* 全项目统一使用 **`cfg`**；所有 `validate_*` / `build_eval_*` 函数都**显式接收 cfg**。
* 名称规范化 `_norm()` 已给出，能兼容 `cpencil/sketch` 等变体，`fnmatch` 提升容错。

---

## 6) 你应该看到的日志（示例）

```
[EVAL] gallery=3510  queries=[('single/nir', 3510), ('single/sk', 3510), ('single/cp', 3510), ('single/text', 3510), ('quad/nir+sk+cp+text', 3510)]
[EVAL] epoch=1  mAP(all)=0.2xxx  |  mAP@single=0.1xxx  mAP@quad=0.3xxx
[epoch 1] steps_run=1863/1863  (max_steps=0)
```

---

## 小结（接下来建议）

* 先把上面 1–3 贴上，评测就会**只跑 5 种**并且复用 gallery 特征，速度明显提升。
* 如果评测仍在 epoch 中途触发：确保**训练循环里没有“每 N 步评测”的残留**，只在 epoch 结束后调用 `validate_competition_style`。
* 看到 mAP 之后，再回到采样器把 `pair_coverage_mavg` 从 \~0.7 拉到 ≥0.85（提升 K 或提高强配对成功率），SDM 收敛会更顺。
