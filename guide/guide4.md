* **steps\_per\_epoch = len(train\_loader)**。你日志里就是 **1863**。
* tqdm 左边的 **394/1863**，分子 **394** 就是**当前 epoch 的第 394 个 step**。
* **全局步** = `(epoch_idx-1) * 1863 + 394`。
  所以你现在正处在我说的 **300–500 step 冒烟窗口**里。

你给出的日志里 **CE≈5.99=log(400)** 几乎不降 → 按“Go/No-Go”规则：**No-Go（先停）**。

---

## 立刻做这三步（5 分钟内见效）

1. **先只训 CE，关 SDM 预热 1 个 epoch**
   `sdm_weight = 0.0`。CE 起码要能从 5.99 往 5.6/5.5 掉，否则是梯度/优化器问题。

2. **把分类头的学习率拉高并确认真的在更新**
   用 **AdamW lr=1e-2**（只给 `bn_neck.classifier`），其余保持原来或更小：

```python
head = [p for n,p in model.named_parameters() if n.startswith("bn_neck.classifier")]
backbone = [p for n,p in model.named_parameters() if not n.startswith("bn_neck.classifier")]

optimizer = torch.optim.AdamW([
    {"params": head, "lr": 1e-2, "weight_decay": 1e-4},
    {"params": backbone, "lr": 1e-5, "weight_decay": 1e-4}
])

# 每 100 step 打一次头部权重范数与更新幅度，确认在动
if (step % 100) == 0:
    w = model.bn_neck.classifier.weight
    print(f"[dbg] head |w|={w.norm():.4f}")
```

3. **确认真的在做 backward/step（很多人卡在这）**

```python
# 训练循环里确保：
scaler.scale(loss).backward()           # 有用混精的话
# 梯度累积到位再：
scaler.step(optimizer); scaler.update()
optimizer.zero_grad(set_to_none=True)

# 打印梯度范数（每100步一次）
if (step % 100) == 0:
    g = 0.0
    for p in model.bn_neck.classifier.parameters():
        if p.grad is not None:
            g += (p.grad.detach().float().norm().item())
    print(f"[dbg] head grad-norm ≈ {g:.4f}")
```

如果 `|w|` 和 `grad-norm` 长时间不变，说明**没在更新**（要么没进 `optimizer`，要么没 `backward/step`，要么被 `grad_scaler`/accumulation 逻辑跳过）。

---

## 两个快速自测（十分钟内判因）

* **标签合法性**（CrossEntropy 要求 0…C-1）：

```python
assert labels.min().item() >= 0 and labels.max().item() < model.num_classes
```

> 你之前显示范围 14–399、`num_classes=400` 是合法的，但再次用断言确保没有越界/噪声标签。

* **小数据过拟合**：抽 256 条固定样本，单独训 CE，10 个 epoch。
  若 CE 仍≈5.99，不是“数据难”，就是**优化器/梯度没生效**。

---

## 什么时候再继续长跑？

* 做完上面三步，跑到 **step≈200–300**：

  * CE **≤5.6** 且下降趋势明确 → 继续本 epoch；
  * 仍在 **≈5.99** → 停，进一步排查：

    * 检查 `classifier` 是否确实接在你用于 CE 的特征上（BNNeck后的那支）；
    * 取消所有 `no_grad()` 上下文；
    * 暂关梯度裁剪/跳步逻辑；
    * 将 `accumulation_steps=1` 临时化简流程。

> 你的 `Feat(BN)=8.00` 已经 OK，说明范数约束在工作；症结更可能在 **学习率/优化器/梯度流**，而不是模型结构或数据集本身。
