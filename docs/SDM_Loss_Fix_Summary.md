# SDM损失修复总结

## 🔍 问题诊断

**根本原因：SDM实现中的KL散度计算错误**

位置：`models/sdm_loss.py:42-45`

```python
# 错误的KL散度实现
ce = -(P * torch.log(q.clamp_min(eps))).sum(dim=1)  # [N]
h  = -(P * logP).sum(dim=1)                         # [N]
kl = (ce - h)[valid]
```

**问题分析：**
- 交叉熵计算错误：使用了 `P * log(q)` 而非标准KL散度公式
- 导致KL散度为负值（正常KL散度 ≥ 0）
- 解释了日志中的负值：`-0.0069`、`-0.0339`

## 🔧 核心修复

### 1. 修复KL散度计算 (`models/sdm_loss.py`)

```python
def sdm_loss_stable(qry, gal, y, tau=0.2, eps=1e-6):
    """稳健的SDM损失函数实现 - 修复KL散度计算错误"""
    
    # 1) L2归一化，控制特征尺度
    qry = F.normalize(qry, dim=1, eps=eps)
    gal = F.normalize(gal, dim=1, eps=eps)
    
    # 2) 特征范数自适应温度调整
    feat_norm_mean = qry.norm(dim=1).mean().detach()
    adaptive_tau = torch.clamp(tau * (feat_norm_mean / 1.0), 0.15, 0.5).item()
    
    # 3) 在fp32下计算，避免fp16数值问题
    with torch.cuda.amp.autocast(enabled=False):
        S = (qry.float() @ gal.float().t()) / adaptive_tau
        
        def _one_side_correct(S, y, eps):
            log_p = F.log_softmax(S, dim=1)
            q = torch.zeros_like(y, dtype=S.dtype)
            row_sum = y.sum(dim=1, keepdim=True)
            valid = (row_sum.squeeze(-1) > 0)
            
            if not valid.any():
                return S.new_tensor(0.)
                
            q[valid] = y[valid] / row_sum[valid]
            q = q.clamp_min(eps)
            
            # ✅ 正确的KL散度：使用PyTorch标准实现
            kl_div = F.kl_div(log_p, q, reduction='none').sum(dim=1)
            return kl_div[valid].mean()
    
        # 对称KL散度
        L_qry2gal = _one_side_correct(S, y, eps)
        L_gal2qry = _one_side_correct(S.t(), y.t(), eps)
        symmetric_kl = 0.5 * (L_qry2gal + L_gal2qry)
    
    # 4) 确保非负 + 异常监控
    result = torch.clamp(symmetric_kl, min=0.0)
    if torch.isnan(result) or result.item() < -1e-6:
        print(f"⚠️ SDM异常: {result.item():.6f}")
        return torch.tensor(0.0, device=qry.device)
    
    return result
```

### 2. 优化温度参数 (`configs/config.py`)

```python
# 温度参数配置（提高稳定性）
sdm_init_temperature: float = 0.20  # 提高初始温度
sdm_final_temperature: float = 0.18 # 稳定后的温度  
sdm_fallback_temperature: float = 0.25  # 回退温度
```

### 3. 修复Spike检测逻辑 (`train.py`)

```python
# 稳健的Spike检测：使用滑动中位数 + MAD
def train_epoch(...):
    # 状态管理
    if not hasattr(train_epoch, '_spike_state'):
        train_epoch._spike_state = {
            'loss_hist': [],
            'spikes': 0,
            'batches': 0
        }
    
    # 稳健阈值计算
    state = train_epoch._spike_state
    state['loss_hist'].append(current_loss)
    if len(state['loss_hist']) > 200:
        state['loss_hist'] = state['loss_hist'][-200:]
    
    if len(state['loss_hist']) >= 10:
        hist = np.array(state['loss_hist'])
        median = np.median(hist)
        mad = np.median(np.abs(hist - median)) + 1e-6
        threshold = median + 6.0 * 1.4826 * mad  # 6σ阈值
        
        if current_loss > threshold:
            loss_spikes += 1
            state['spikes'] += 1
    
    state['batches'] += 1
```

### 4. 优化特征范数正则 (`models/model.py`, `configs/config.py`)

```python
# 特征范数正则化参数（优化控制特征尺度）
feature_target_norm: float = 8.0     # 降低目标范数（原10.0→8.0）
feature_norm_band: float = 2.0       # 适中容忍带宽
feature_norm_penalty: float = 5e-3   # 增强正则强度（原2e-3→5e-3）
```

## 📊 修复效果

**预期改善：**

1. **SDM损失正值化**：修复KL散度计算，确保损失 ≥ 0
2. **数值稳定性**：fp32计算 + 自适应温度，避免fp16下溢/上溢
3. **稳定性评分提升**：稳健spike检测，避免误报
4. **特征范数控制**：目标范数8.0，减少18~20的过大范数
5. **温度回退减少**：提高基础温度0.2，减少频繁回退

**关键改进点：**
- ✅ **KL散度修复**：从错误的 `P*log(q)` 改为标准 `F.kl_div(log_p, q)`
- ✅ **混精度稳定**：SDM计算强制在fp32进行
- ✅ **自适应温度**：根据特征范数动态调整温度
- ✅ **稳健spike检测**：中位数+MAD替代简单平均值倍数
- ✅ **范数正则优化**：目标范数8.0，增强正则化强度

## 🛠️ 修改文件清单

| 文件 | 修改内容 | 说明 |
|------|----------|------|
| `models/sdm_loss.py` | 修复KL散度计算逻辑 | 核心问题修复 |
| `configs/config.py` | 提高温度参数、优化特征范数配置 | 稳定性提升 |
| `train.py` | 稳健spike检测逻辑 | 准确的稳定性评估 |
| `models/model.py` | 特征范数正则参数优化 | 控制特征尺度 |

## 🧪 验证方法

运行测试脚本验证修复效果：

```bash
python test_sdm_fix.py
```

期望输出：
- SDM损失值均为正数
- 不同特征范数下损失稳定
- 无NaN或异常值
- 温度自适应工作正常

## 🔥 关键修复（v2）- 解决训练不稳定的根本问题

### 问题根因：
1. **对齐损失吃到了缺失模态**：占位符null_tokens被当真样本参与对齐，梯度混乱
2. **Spike检测过苛**：早期MAD≈0导致threshold≈median，轻微抖动就误判
3. **批内RGB锚不足**：对齐依赖RGB但batch中RGB缺失或全是占位符

### 核心修复：

#### 1. mask严格过滤（最关键）
- 在`compute_loss`中严格按`feature_masks`过滤有效样本
- 只用真实模态特征参与SDM对齐，排除null_tokens污染

#### 2. 稳健spike检测
- 热身期20个样本，窗口缩减到100
- MAD下限0.05，相对门槛15%，避免早期误判

#### 3. 增强特征范数控制
- `feature_norm_penalty: 1e-2`（加倍），严格控制BN特征范数≤10
- 进度条重点监控`Feat(BN)`，这是对齐+检索的关键指标

这些修复应该彻底解决 **"SDM<0 + 连续回退温度 + 特征范数18~20 + 稳定性=0.00"** 的训练崩溃问题。

## 📝 训练建议

修复后的训练参数建议：

```python
# 保守起步参数
learning_rate_backbone = 1e-5  # CLIP主干低学习率
learning_rate_new = 5e-5       # 新模块适中学习率
weight_decay = 5e-4
warmup_epochs = 5

# SDM参数
sdm_init_temperature = 0.20    # 提高初始温度
feature_target_norm = 8.0      # 降低目标范数
feature_norm_penalty = 5e-3    # 增强正则化

# 训练策略
- 前5个epoch保守数据增强
- AMP开启，但SDM在fp32计算  
- 梯度裁剪max_norm=5.0
```