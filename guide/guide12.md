两件事一起修：**评测崩溃（dict没dataset）** + **训练只跑到step=80就结束**。下面给你最小可落地补丁。

---

## ① 修复评测：`query_loaders` 里既有 DataLoader 也有 dict

你的 `validate_competition_style` 里这样写：

```python
[(k, len(v.dataset)) for k, v in query_loaders.items()]
```

但 `v` 有时是 **dict**（组合检索/多模态规则），不是 DataLoader，于是报：
`AttributeError: 'dict' object has no attribute 'dataset'`。

### ✅ 补丁：通用“扁平化”查询加载器

把这段工具函数加到 `validate_competition_style` 顶部（或 utils）：

```python
def _flatten_loaders(obj, prefix=""):
    """
    把 {key: DataLoader | dict | list} 递归展开成 [(name, dataloader), ...]
    name 形如 'single/nir' 或 'quad/0' 等，便于打印/统计
    """
    # DataLoader-like
    if hasattr(obj, "dataset") and hasattr(obj, "__iter__"):
        yield (prefix.rstrip("/") or "root", obj)
        return

    # dict of loaders or nested dict
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten_loaders(v, f"{prefix}{k}/")
        return

    # list/tuple of loaders
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten_loaders(v, f"{prefix}{i}/")
        return

    raise TypeError(f"Unsupported query_loaders node type: {type(obj)} at {prefix!r}")
```

然后把你原来的打印/评测遍历替换为：

```python
pairs = list(_flatten_loaders(query_loaders))
print(
    "[EVAL] gallery=%d  queries=%s"
    % (len(gallery_loader.dataset), [(k, len(dl.dataset)) for k, dl in pairs])
)

# 真正评测时也遍历 pairs
all_metrics = {}
for name, qloader in pairs:
    # 逐类/逐规则评测
    m = evaluate_one(model, gallery_loader, qloader, device, k_map=k_map, sample_ratio=sample_ratio)
    all_metrics[name] = m

# 聚合并打印你项目里需要的指标键（以下示例按你日志里的键名）
comp_metrics = {
    "map_single":  aggregate_subset(all_metrics, key_contains="single"),
    "map_quad":    aggregate_subset(all_metrics, key_contains="quad"),
}
comp_metrics["map_avg2"] = (comp_metrics["map_single"] + comp_metrics["map_quad"]) / 2.0
print("[EVAL] epoch=%d  mAP(all)=%.4f  |  mAP@single=%.4f  mAP@quad=%.4f"
      % (epoch, comp_metrics["map_avg2"], comp_metrics["map_single"], comp_metrics["map_quad"]))
```

> 这样无论 `query_loaders` 是 `{single: {nir: DL, sk: DL, ...}, quad: DL}` 还是更深的嵌套，都能跑通和打印。

---

## ② 修复“每个 epoch 只跑到 step=80 就结束”

你在 epoch 尾部打印了：

```
[epoch 1] steps_run=80/1863
```

这说明 **`train_epoch` 在第 80 步提前 return/ break**。常见源头有：

* `max_steps_per_epoch / smoke_steps / debug_steps` 等限制仍然生效；
* 把 **评测触发** 写在训练循环里并 `return`；
* 基于健康指标（例如 `pair_coverage_mavg`）的“早停”还在。

### ✅ 补丁：在 `train_epoch` 明确禁用截断

把下面这段放到 `train_epoch` loop 外面，并在 loop 里唯一允许的截断就是“显式配置的 max\_steps>0”。

```python
max_steps = int(getattr(cfg, "max_steps_per_epoch", 0) or 0)
steps_run = 0
for batch_idx, batch in enumerate(dataloader, start=1):
    # ... 正常训练 ...
    steps_run = batch_idx
    if max_steps > 0 and steps_run >= max_steps:
        break  # 只有这一处允许截断

# 统一打印步数（便于确认是否完整跑完）
print(f"[epoch {epoch}] steps_run={steps_run}/{len(dataloader)}  (max_steps={max_steps})")
```

同时**搜一遍**把任何会在训练循环内 `break/return` 的逻辑注释掉（尤其是下面几类）：

* `if (batch_idx + 1) % eval_freq == 0: return ...`
* `if pair_coverage_mavg < target: return ...`
* `if smoke_steps and batch_idx >= smoke_steps: break`

> 评测请放到 **`train_epoch` 返回之后** 再在 `train_multimodal_reid` 里执行，不要混在训练循环内。

---

## ③ 评测触发与频率（确保你“看得到 mAP”）

在 `train_multimodal_reid` 中使用下面的触发规则（你已经接近了，只是再确保不进训练循环里）：

```python
eval_start_epoch   = int(getattr(cfg, "eval_start_epoch", 1))
eval_every_n_epoch = int(getattr(cfg, "eval_every_n_epoch", 1))

for epoch in range(1, cfg.max_epoch + 1):
    train_epoch(...)
    if epoch >= eval_start_epoch and ((epoch - eval_start_epoch) % eval_every_n_epoch == 0):
        gallery_loader, query_loaders = build_eval_loaders_by_rule(cfg)
        comp_metrics = validate_competition_style(model, gallery_loader, query_loaders, device, k_map=100, sample_ratio=cfg.get("sample_ratio", 1.0))
        print("[EVAL] epoch=%d  mAP(all)=%.4f  |  mAP@single=%.4f  mAP@quad=%.4f"
              % (epoch, comp_metrics["map_avg2"], comp_metrics["map_single"], comp_metrics["map_quad"]))
```

---

## ④ 快速验证清单

* 训练一个 epoch 后，应该看到：

  ```
  [epoch 1] steps_run=1863/1863  (max_steps=0)
  ```

* 紧接着看到评测打印：

  ```
  [EVAL] gallery=XXXXX  queries=[('single/nir', N1), ('single/sketch', N2), ...]
  [EVAL] epoch=1  mAP(all)=0.1234  |  mAP@single=0.1100  mAP@quad=0.1368
  ```

---

## ⑤ 小提示（与你的当前日志相关）

* 你现在 `pair_coverage_mavg ≈ 0.7`，低于我们建议的 0.85。等“完整 epoch + 正常评测”跑通后，再把 **K 提到 ≥3（若显存允许）** 或增大“强配对成功率”，让覆盖率上来，SDM 才更有效。
* `[sdm] epoch=1 weight=0.000 use_sdm=True`：warmup=1 时 epoch2 应显示 0.1；确认调度器边界已改成 `<` 或把 warmup 设为 1。

---

需要的话，把 `validate_competition_style` 里你现在的函数头和 10 行上下文贴一下，我帮你把“扁平化+聚合”的代码直接对齐到你的变量名。
