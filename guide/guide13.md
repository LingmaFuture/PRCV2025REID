没问题！我们把评测**只保留 4 个单模态 + 1 个四模态**，其他（双/三模态）一律跳过。下面是**最小改动清单**，贴上就能生效、评测时间立减。

---

## 1) 配置里加白名单（建议）

`configs/config.py` 增加：

```python
# 仅评测四单模态 + 四模态
eval_include_patterns = ["single/nir", "single/sk", "single/cp", "single/text", "quad/nir+sk+cp+text"]

# 只在每个epoch结束评测；若你以前在“每N步评测”，务必设为0或删除那段逻辑
eval_every_n_epoch = 1
eval_every_n_steps = 0
```

---

## 2) 在 validate\_competition\_style 里过滤需要的 query（核心补丁）

如果你已经按 Fix12 加了 `_flatten_loaders`，直接在它后面加过滤即可：

```python
import fnmatch

pairs = list(_flatten_loaders(query_loaders))  # [("single/nir", dl), ("double/nir+sk", dl), ...]
include = getattr(cfg, "eval_include_patterns", ["single/nir", "single/sk", "single/cp", "single/text", "quad/nir+sk+cp+text"])

# 只保留白名单
pairs = [(name, dl) for (name, dl) in pairs if any(fnmatch.fnmatch(name, pat) for pat in include)]

print("[EVAL] gallery=%d  queries=%s" % (len(gallery_loader.dataset), [(n, len(dl.dataset)) for n, dl in pairs]))
```

随后把原来遍历所有 query 的评测循环，改成只遍历 `pairs`：

```python
all_metrics = {}
for name, qloader in pairs:
    # 保留你现有的“单条query评测”逻辑（提特征→相似度→mAP）
    m = evaluate_one_query(model, gallery_loader, qloader, device, k_map=k_map, sample_ratio=sample_ratio)
    all_metrics[name] = m
```

> 如果你没有 `evaluate_one_query` 这个函数，就把**你原来对单个 qloader 的评测代码块**搬进来当作这个函数的内容即可。

---

## 3) 只聚合“单模态均值 + 四模态”并打印

在 `validate_competition_style` 末尾聚合并返回（键名与你当前主循环一致）：

```python
def _get_map(m):
    # 兼容不同命名的mAP字段
    for k in ("mAP", "map", "mAP_mean"):
        if isinstance(m, dict) and k in m:
            return float(m[k])
    return 0.0

# 单模态均值
single_maps = [_get_map(all_metrics.get(k, {})) for k in ("single/nir","single/sk","single/cp","single/text")]
map_single = sum(single_maps) / max(1, len([x for x in single_maps if x>0 or x==0]))  # 防除零

# 四模态
map_quad = _get_map(all_metrics.get("quad/nir+sk+cp+text", {}))

comp_metrics = {
    "map_single": map_single,
    "map_quad": map_quad,
    "map_avg2": (map_single + map_quad) / 2.0
}

print("[EVAL] epoch=%d  mAP(all)=%.4f  |  mAP@single=%.4f  mAP@quad=%.4f"
      % (epoch, comp_metrics["map_avg2"], comp_metrics["map_single"], comp_metrics["map_quad"]))

return comp_metrics
```

---

## 4) 再补一刀：确保“只在 epoch 结束评测”

如果你**仍然**看到训练只跑几十步就评测，说明还有“每 N 步评测”的残留。把 `train_epoch` 内部类似下面的触发删掉或关掉：

```python
# ❌ 删掉/注释
# if eval_every_n_steps > 0 and (batch_idx + 1) % eval_every_n_steps == 0:
#     validate_competition_style(...)
#     return metrics  # 更要避免这种提前 return
```

评测应该只在外层 `for epoch in ...` 的**每轮结束**触发。

---

## 预期日志（示例）

```
[EVAL] gallery=3510  queries=[('single/nir', 3510), ('single/sk', 3510), ('single/cp', 3510), ('single/text', 3510), ('quad/nir+sk+cp+text', 3510)]
[EVAL] epoch=1  mAP(all)=0.2314  |  mAP@single=0.2103  mAP@quad=0.2525
```

这样就只跑 5 类评测，速度会比全组合（单/双/三/四）快很多。

---

## 可选再提速（不改评测定义）

* **sample\_ratio**：评测时对 gallery 或 query 采样一部分（你的函数已有该参数就利用起来，比如 `sample_ratio=0.5` 先看趋势）。
* **缓存特征**：对 gallery 的特征在首次评测后缓存到磁盘/内存，下次评测直接加载，减少重复前向。

---

需要的话我可以把 `evaluate_one_query` 的最小模板按你当前代码风格改好（提特征→相似度→mAP），直接可用。
