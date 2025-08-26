# Fix4.md - Guide4.py 实施总结报告

## 问题诊断
基于guide4.py的分析，训练在步数 300-500 的"冒烟窗口"内出现 **CE loss ≈ 5.99 = log(400) 几乎不降** 的问题，按照"Go/No-Go"规则判定为 **No-Go（先停）**，需要立即诊断和修复。

## 核心问题识别
1. **CE损失停滞**: CE ≈ 5.99 几乎不变，说明分类器没有学习到有效的判别信息
2. **潜在梯度流问题**: 可能存在优化器/梯度没生效的情况
3. **学习率配置**: 分类头学习率可能过低，无法快速收敛

## 实施的修复方案

### 1. ✅ CE-only严格预热
```python
# configs/config.py
sdm_weight_warmup_epochs: int = 1  # guide4.py: 先只用1个epoch CE调试
```
**修复目标**: 确保第1个epoch完全禁用SDM，专注CE损失是否能从5.99降到5.6以下

### 2. ✅ 分类头高学习率策略
```python
# models/model.py - get_learnable_params()
if name.startswith("bn_neck.classifier"):
    classifier_params.append(param)

# 分类头使用1e-2高学习率
param_groups.append({
    'params': classifier_params,
    'lr': 1e-2,  # guide4.py: 分类头高学习率确保CE能降
    'name': 'classification_head'
})
```
**修复目标**: 通过高学习率(1e-2)确保分类头能快速更新，而backbone保持1e-5的保守学习率

### 3. ✅ 梯度和权重监控
```python
# train.py - 每100步监控
if (batch_idx + 1) % 100 == 0:
    w = model.bn_neck.classifier.weight
    print(f"[guide4-dbg] step={batch_idx+1} head |w|={w.norm():.4f}")
    
    g = 0.0
    for p in model.bn_neck.classifier.parameters():
        if p.grad is not None:
            g += (p.grad.detach().float().norm().item())
    print(f"[guide4-dbg] step={batch_idx+1} head grad-norm ≈ {g:.4f}")
```
**修复目标**: 实时监控分类头权重范数和梯度范数，确认参数确实在更新

### 4. ✅ 标签合法性断言
```python
# train.py - 每个batch检查
assert labels.min().item() >= 0 and labels.max().item() < model.num_classes, \
    f"guide4.py: 标签越界! labels范围[{labels.min().item()}, {labels.max().item()}], 要求[0, {model.num_classes-1}]"
```
**修复目标**: 确保CrossEntropy输入标签在合法范围[0, C-1]内，避免越界导致的训练失效

### 5. ✅ 简化训练流程
```python
# configs/config.py
gradient_accumulation_steps: int = 1  # guide4.py: 临时化简为1，确保梯度流正常
```
**修复目标**: 临时移除梯度累积复杂性，每个batch直接backward和step，简化调试流程

## 期望效果与判定标准

### 成功标准 (200-300步内)
- ✅ **CE loss ≤ 5.6** 且呈现明确下降趋势
- ✅ **权重范数变化**: `|w|` 数值逐步增长，表明参数在更新
- ✅ **梯度范数正常**: `grad-norm > 0` 且有合理数值，不为0或极小值

### 失败症状及后续排查
如果仍然 **CE ≈ 5.99 不变**，需要进一步检查：
1. **分类器连接性**: 确认`classifier`确实接在BNNeck后的特征上
2. **梯度流中断**: 检查是否有`no_grad()`上下文意外阻断
3. **优化器配置**: 确认分类头参数确实加入了优化器
4. **数值稳定性**: 暂时关闭梯度裁剪/跳步逻辑

## 调试工具和监控点

### 实时监控输出格式
```
[guide4-dbg] step=100 head |w|=0.0312
[guide4-dbg] step=100 head grad-norm ≈ 0.0847
[guide4-dbg] step=200 head |w|=0.0324  # 权重在增长 ✅
[guide4-dbg] step=200 head grad-norm ≈ 0.0761
```

### 标签检查输出
```
AssertionError: guide4.py: 标签越界! labels范围[0, 425], 要求[0, 399]
```
如出现此错误，说明数据划分或ID映射存在问题。

## 意见和建议

### 🟡 不够清晰的地方
1. **guide4.py第65行"小数据过拟合"测试**: 建议提供具体的256样本抽取代码实现
2. **监控频率**: 每100步可能过于频繁，建议可配置化（如每50/100/200步）
3. **权重范数基准**: 缺少"正常"权重范数的参考值范围

### 🔴 不够合理的地方
1. **学习率1e-2过于激进**: 对于预训练CLIP模型，1e-2可能导致灾难性遗忘，建议先尝试5e-3或1e-3
2. **完全禁用梯度累积**: 在GPU内存允许的情况下，建议保持小的accumulation（如2-4），完全禁用可能影响收敛稳定性
3. **缺少early stopping**: 当CE确实开始下降后，应该有清晰的"何时恢复SDM"的条件

### 🟢 改进建议
1. **渐进式学习率**: 分类头从5e-3开始，观察100步后再决定是否提高到1e-2
2. **监控dashboard**: 建议将监控信息写入tensorboard或wandb，便于可视化分析
3. **自动回退机制**: 如果权重范数增长过快(>10倍)，自动降低学习率

## 总结

guide4.py的诊断思路清晰，通过"止血-诊断-修复"三步法系统性解决CE损失停滞问题。所有建议的修复方案已完整实施，现在需要通过实际训练验证这些修复措施的效果。

**下一步行动**: 运行 `python quick_start.py`，观察前200-300步的CE损失和监控输出，根据结果决定是否需要进一步调整参数。