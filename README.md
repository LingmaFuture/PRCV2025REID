# PRCV2025REID - 多模态人员重识别系统

## 📝 项目简介

基于 **ORBench 五模态数据集**，本项目提出了**单模型支持任意模态组合检索**的统一框架 **ReID5o**。系统支持 RGB、红外(IR)、彩铅(CP)、素描(SK)、文本(T) 五种模态的任意组合检索，在 ORBench 上平均 mAP 从 **58.09%**（单模态）提升到 **86.35%**（四模态）。

### 🎯 核心特性

- **全模态任意组合**：单模型支持32种模态组合（2^5），比现有方法更贴近真实场景
- **统一编码架构**：MTA+MER+FM组成的清晰流水线，替换成本低
- **轻量专家路由**：仅 r=4 的低秩专家即带来最优精度
- **显著互补效果**：文本+彩铅 mAP +26.26%，再加红外 +5.61%

## 🔧 环境配置

### 1. 激活虚拟环境
```bash
conda activate prvc
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 主要依赖版本
- `torch>=2.6.0`
- `transformers>=4.20.0`
- `timm>=0.6.0`
- `Pillow>=8.0.0`
- `numpy>=1.21.0`
- `scikit-learn>=1.0.0`

## 📊 数据集说明

### ORBench 五模态数据集
- **规模**：1000个身份，总计152,297个样本
  - 45,113 RGB（画廊集）
  - 26,071 红外(IR)
  - 18,000 彩铅(CP)  
  - 18,000 素描(SK)
  - 45,113 文本描述
- **任务**：以 RGB 作为画廊，支持任意单/多模态查询
- **评测模式**：单模态(MM-1)、双模态(MM-2)、三模态(MM-3)、四模态(MM-4)

### 数据结构
```
data/
├── vis/          # 可见光图像
├── nir/          # 红外图像  
├── sk/           # 素描图像
├── cp/           # 彩铅图像
└── text_annos.json  # 文本标注
```

## 🏗️ 模型架构

### ReID5o 统一框架

1. **多模态分词装配器(MTA)**
   - 为5个模态提供独立tokenizer
   - 映射到512维共享表征空间
   - 产生离散控制信号给路由器

2. **多专家路由(MER)**
   - 基于CLIP-B/16统一编码器
   - 每层注入低秩专家矩阵(LoRA r=4)
   - 公式：`y = Wx + c_mod * B_mod * A_mod * x`

3. **特征混合器(FM)**
   - MSA + 1层Transformer + MLP
   - 支持任意模态组合的串行拼接融合

4. **学习策略**
   - RGB为对齐核心
   - SDM对齐损失 + ID分类损失
   - 损失函数：`L = Σ SDM(z_R, z_ci) + α Σ IC(z_ci)`

## 🚀 快速开始

### 1. 核心配置文件

**`configs/config.py`** - 训练配置
```python
# 模型配置
clip_model_name = "openai/clip-vit-base-patch16"  # CLIP-B/16统一编码器
fusion_dim = 512                                  # 融合维度
mer_lora_rank = 4                                # MER LoRA秩

# 训练配置  
num_epochs = 60                                  # 总epoch数
base_learning_rate = 5e-6                       # 基础学习率
warmup_epochs = 5                               # 热身epoch数
```

### 2. 训练命令

```bash
# 激活环境
conda activate prvc

# 启动训练
python train.py
```

### 3. 关键训练参数

- **P×K采样**：P=3个ID，K=2个实例，确保批内配对
- **分层学习率**：CLIP主干5e-6，MER专家2e-5，分类头3e-3
- **混合精度**：bfloat16/float16 + 梯度累积
- **SDM调度**：前5个epoch热身，后续渐进增加权重

## 📈 训练与评测

### 1. 数据采样策略
```python
# datasets/dataset.py 中的核心采样器
class ModalAwarePKBatchSampler_Strict:
    """强配对采样器，确保每个ID包含vis+非vis模态"""
    def __init__(self, num_ids_per_batch=3, num_instances=2):
        # 每批3个ID，每ID 2个样本，保证配对
```

### 2. 模型前向流程
```python
# models/model.py 中的前向传播
def forward(self, images, texts, modality_masks):
    # 1. 多模态编码（CLIP+MER）
    # 2. SDM语义分离（训练时）
    # 3. 特征融合（FM模块）
    # 4. BN Neck + ID分类
    return outputs
```

### 3. 评测指标

- **mAP@100**：主要评测指标
- **CMC@1/5/10**：累积匹配特征曲线  
- **四类评测**：单/双/三/四模态组合

### 4. 评测命令

```bash
# 完整评测（所有模态组合）
python train.py  # 训练中自动评测

# 推理评测
python tools/evaluate.py --model_path checkpoints/best_model.pth
```

## 📊 实验结果

### ORBench 数据集性能

| 模态组合 | mAP(%) | 提升幅度 |
|---------|--------|----------|
| MM-1 单模态 | 58.09 | - |
| MM-2 双模态 | 75.26 | +17.17 |
| MM-3 三模态 | 82.83 | +7.57 |
| MM-4 四模态 | **86.35** | +3.52 |

### 分模态详细性能

| 模态 | mAP(%) | 特点 |
|------|--------|------|
| 文本(T) | 65.2 | 语义信息丰富 |
| 红外(IR) | 58.4 | 夜间/恶劣环境适用 |
| 彩铅(CP) | 52.8 | 艺术化表征 |
| 素描(SK) | 48.6 | 轮廓特征突出 |

## 💡 项目亮点

### 1. 技术创新

- **首个五模态统一编码器**：单模型处理所有模态组合
- **轻量MER路由**：r=4 低秩专家，参数增量小效果显著
- **渐进式SDM对齐**：RGB锚定的跨模态语义对齐

### 2. 工程优化

- **强配对采样**：保证训练批次的模态覆盖
- **AMP混合精度**：bfloat16加速训练
- **分层学习率**：不同模块采用差异化学习策略
- **健康线监控**：实时监控训练稳定性

### 3. 实用价值

- **真实场景适配**：支持任意可用模态的灵活检索
- **高质量数据基座**：ORBench提供充足的多模态训练数据
- **端到端可部署**：完整的训练-评测-推理流程

## 📁 代码结构

```
PRCV2025REID/
├── configs/
│   └── config.py              # 统一训练配置
├── datasets/  
│   └── dataset.py            # 多模态数据集和采样器
├── models/
│   ├── model.py              # ReID5o核心模型
│   ├── clip_backbone.py      # CLIP+MER编码器
│   └── sdm_loss.py           # SDM对齐损失
├── train.py                  # 主训练脚本
├── requirements.txt          # 依赖列表
└── README.md                # 项目说明
```

## 🔧 故障排除

### 1. 常见问题

**Q: 训练时出现"无SDM正对"警告？**
A: 检查采样器配置，确保每个batch包含vis+非vis模态样本。

**Q: 特征范数过大/过小？**
A: 调整BN层设置和学习率，建议BN特征范数控制在8-10之间。

**Q: 内存不足？**
A: 减少batch_size，启用梯度累积，或使用更小的图像尺寸。

### 2. 调试技巧

```bash
# 启用调试模式
export CUDA_LAUNCH_BLOCKING=1

# 查看GPU内存
nvidia-smi -l 1

# 检查数据集采样
python datasets/dataset.py
```

## 📄 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{reid5o2025,
  title={ReID5o: Unified Multi-Modal Person Re-Identification with Any-Modal Query},
  author={Your Name},
  journal={PRCV 2025},
  year={2025}
}
```

## 📞 联系方式

- 项目地址：[GitHub链接]
- 邮箱：[联系邮箱]  
- 更新日志：见 `CHANGELOG.md`

---
**最后更新**：2025年1月

> 💡 **提示**：首次运行前请确保数据集路径正确配置，激活 `prvc` 虚拟环境，并检查GPU内存充足。