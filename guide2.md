下面是**超短总结 + 最快修复**（立刻能跑通）：

# 问题总结

* 报错：`prefetch_factor option could only be specified in multiprocessing...`
* 根因：在 **评测加载器**（`build_eval_loaders_by_rule`）里，`num_workers=0` 却仍然给了 `prefetch_factor`（以及可能给了 `persistent_workers=True`）。`prefetch_factor` 只允许在 `num_workers>0` 的多进程模式下使用。

---

# 最快修复（两种任选其一）

## 方案 A（不启多进程，最稳）

把 `prefetch_factor` 和 `persistent_workers` **按需启用**：当 `num_workers==0` 时不传它们。

```python
# 在 build_eval_loaders_by_rule 里，统一写法：
nw = getattr(cfg, "num_workers_eval", None) or getattr(cfg, "num_workers", 0)

gallery_loader = DataLoader(
    gallery_dataset,
    batch_size=cfg.eval_batch_size,
    shuffle=False,
    num_workers=nw,
    pin_memory=True,
    collate_fn=compatible_collate_fn,
    persistent_workers=(nw > 0),                 # 只有 >0 才开启
    prefetch_factor=(2 if nw > 0 else None)      # 只有 >0 才设置，否则为 None 或干脆删掉该参数
)

# 对每个 query_loader 同样处理（复制这两行判断逻辑）
```

> 要点：当 `nw==0` 时，**不要**传 `prefetch_factor`；`persistent_workers` 也要跟随 `nw>0` 再设 True。

## 方案 B（想要预取，开多进程）

把评测的 `num_workers` 设为 2（Windows 常用值），保留 `prefetch_factor=2` 与 `persistent_workers=True`。

```python
nw = 2  # 或 cfg.num_workers_eval = 2
gallery_loader = DataLoader(
    gallery_dataset,
    batch_size=cfg.eval_batch_size,
    shuffle=False,
    num_workers=nw,
    pin_memory=True,
    collate_fn=compatible_collate_fn,
    persistent_workers=True,
    prefetch_factor=2
)
```

> Windows 提示：若改为多进程，确保 `if __name__ == "__main__":` 保护已存在（你的脚本入口已具备即可）。

---

# 推荐小封装（一次改好、到处复用）

```python
def dl_kwargs(nw, collate_fn):
    kw = dict(num_workers=nw, pin_memory=True, collate_fn=collate_fn)
    if nw > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return kw

# 用法：
nw = getattr(cfg, "num_workers_eval", None) or getattr(cfg, "num_workers", 0)
gallery_loader = DataLoader(gallery_dataset, batch_size=cfg.eval_batch_size, shuffle=False, **dl_kwargs(nw, compatible_collate_fn))
# 每个 query_loader 同样用 **dl_kwargs(...)
```

---

## 一句话结论

把 `prefetch_factor`（以及 `persistent_workers`）**只在 `num_workers>0` 时传**；否则去掉或设为 `None`。这是一处小改动，保存后即可继续训练/评测。
