# 多模态人员重识别评估协议使用指南

## 📋 概述

本文档详细介绍了基于您提供的评估协议细则实现的多模态ReID评估系统。该系统完全适配您的CLIP+MER架构，支持MM-1/2/3/4模态组合评估，并提供完整的mAP和CMC指标计算。

## 🎯 评估协议要点

### 核心设计
- **Gallery**: 全部RGB图像
- **Query**: MM-1/2/3/4（1/2/3/4种模态组合）
- **特征提取**: CLIP+MER统一编码器
- **融合**: FeatureFusion或简单加权平均
- **评估指标**: mAP、R@1/5/10

### 模态映射
```
数据集模态 → 模型模态
vis       → rgb
nir       → ir  
sk        → sketch
cp        → cpencil
text      → text
```

## 📁 文件结构

```
tools/
├── eval_mm_protocol.py        # 主评估脚本
├── test_eval_protocol.py      # 测试脚本
├── integrate_eval_to_train.py # 训练集成脚本
└── export_submission.py       # Kaggle提交导出

docs/
└── evaluation_protocol_usage.md  # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate prvc

# 确保所有依赖已安装
pip install torch torchvision transformers tqdm pandas numpy Pillow
```

### 2. 基础测试

首先运行测试脚本，确保所有组件正常工作：

```bash
python tools/test_eval_protocol.py
```

预期输出：
```
🧪 开始测试多模态评估协议
=== 测试数据加载和索引构建 ===
✅ 数据索引构建成功，共 XXX 个身份
模态覆盖情况:
  rgb: XXX 个身份
  ir: XXX 个身份
  text: XXX 个身份
  ...

=== 测试模型加载 ===
使用设备: cuda
✅ 模型创建成功
✅ 前向传播成功

...

🎉 所有测试都通过！评估协议可以正常使用
```

### 3. 运行完整评估

```bash
python tools/eval_mm_protocol.py \
    --dataset_root ./data/train \
    --model_path ./checkpoints/best_model.pth \
    --cache_dir ./eval_cache \
    --device cuda
```

## 📊 评估输出解读

### 终端输出示例
```json
{
  "MM-1": {
    "mAP": 0.3456,
    "R@1": 0.2890,
    "R@5": 0.4512,
    "R@10": 0.5234,
    "num_queries": 1245
  },
  "MM-2": {
    "mAP": 0.3789,
    "R@1": 0.3123,
    "R@5": 0.4856,
    "R@10": 0.5567,
    "num_queries": 2341
  },
  "MM-3": {
    "mAP": 0.4012,
    "R@1": 0.3345,
    "R@5": 0.5123,
    "R@10": 0.5890,
    "num_queries": 1876
  },
  "MM-4": {
    "mAP": 0.4234,
    "R@1": 0.3567,
    "R@5": 0.5345,
    "R@10": 0.6123,
    "num_queries": 987
  },
  "AVG(1-4)": {
    "mAP": 0.3873,
    "R@1": 0.3231,
    "R@5": 0.4959,
    "R@10": 0.5704
  }
}
```

### 指标含义
- **MM-1**: 单模态查询（ir/cpencil/sketch/text各自单独）
- **MM-2**: 双模态组合查询（如ir+text、sketch+cpencil等）
- **MM-3**: 三模态组合查询
- **MM-4**: 四模态组合查询
- **AVG(1-4)**: 所有模态组合的平均性能

## 🔧 高级配置

### 1. 模态权重调优

编辑`tools/eval_mm_protocol.py`中的默认权重配置：

```python
# 默认配置
weight_cfg = {
    "ir": 1.0,
    "cpencil": 1.0, 
    "sketch": 1.0,
    "text": 1.2  # 给文本稍高权重
}

# 如果发现某模态表现特别好，可以适当提高权重
weight_cfg = {
    "ir": 1.0,
    "cpencil": 1.1,  # 彩铅表现好，提高权重
    "sketch": 0.9,   # 素描表现差，降低权重
    "text": 1.3      # 文本很重要，进一步提高
}
```

### 2. 缓存管理

Gallery特征会自动缓存到`eval_cache/`目录：
- `rgb_feats.npy`: Gallery特征数组
- `rgb_meta.json`: Gallery元数据

如果模型或数据发生变化，需要清除缓存：
```bash
rm -rf ./eval_cache
```

### 3. 随机种子控制

为了确保结果可复现，所有随机操作都使用固定种子：
```python
# 在eval_mm_protocol.py中
seed = 42  # 固定种子
rng = random.Random(seed)  # 查询构建
```

## 🔗 集成到训练流程

### 方法1：手动集成

在`train.py`的最后添加：

```python
# 训练完成后运行评估
if __name__ == "__main__":
    train_multimodal_reid()
    
    # 自动评估
    try:
        from tools.integrate_eval_to_train import run_post_training_evaluation
        
        eval_results = run_post_training_evaluation(
            dataset_root="./data/train",
            model_path="./checkpoints/best_model.pth"
        )
        
        if eval_results:
            print(f"✅ 最终评估完成！平均mAP: {eval_results['AVG(1-4)']['mAP']:.4f}")
        
    except Exception as e:
        print(f"自动评估失败: {e}")
        print("请手动运行: python tools/eval_mm_protocol.py")
```

### 方法2：使用集成脚本

```bash
python tools/integrate_eval_to_train.py \
    --model_path ./checkpoints/best_model.pth
```

## 📤 Kaggle提交文件生成

### 生成提交CSV

```bash
python tools/export_submission.py \
    --dataset_root ./data/train \
    --model_path ./checkpoints/best_model.pth \
    --output_csv ./submission.csv \
    --top_k 100 \
    --validate
```

### 提交文件格式

生成的CSV包含两列：
- `query_key`: 查询标识符，格式为`{pid}|{mode}|{modalities}|{sample_ids}`
- `ranked_gallery_ids`: 排序后的Gallery ID列表，空格分隔

示例：
```csv
query_key,ranked_gallery_ids
1|MM-1|text|1_text,img_001 img_045 img_123 ...
2|MM-2|ir+text|2_ir+2_text,img_067 img_089 img_201 ...
```

## 🛠️ 故障排除

### 常见问题

1. **数据路径错误**
   ```
   错误：数据集目录不存在 ./data/train
   解决：检查config.py中的data_root配置
   ```

2. **模型加载失败**
   ```
   错误：模型文件不存在 ./checkpoints/best_model.pth
   解决：确保已完成训练并保存了最佳模型
   ```

3. **CUDA内存不足**
   ```
   错误：CUDA out of memory
   解决：使用 --device cpu 或减少batch_size
   ```

4. **图像文件缺失**
   ```
   警告：图像文件不存在 /path/to/image.jpg
   解决：检查数据集完整性，确保所有引用的图像文件存在
   ```

### 调试模式

启用详细输出进行调试：

```python
# 在eval_mm_protocol.py开头添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 性能优化

1. **使用缓存**: Gallery特征只需提取一次
2. **GPU加速**: 确保CUDA可用
3. **批处理**: 可以修改代码支持批量特征提取

## 📈 性能基准

### 预期性能范围

基于CLIP+MER架构的典型性能：
- **MM-1单模态**: mAP 0.25-0.40
- **MM-2双模态**: mAP 0.30-0.45  
- **MM-3三模态**: mAP 0.35-0.50
- **MM-4四模态**: mAP 0.40-0.55

### 性能分析

1. **模态重要性**: 通常 text > rgb > ir > cpencil > sketch
2. **组合效应**: 多模态组合通常优于单模态
3. **文本影响**: 文本描述质量对性能影响很大

## 📝 注意事项

### 1. 数据一致性
- 确保训练和评估使用相同的数据集
- 检查person_id的一致性
- 验证模态文件的完整性

### 2. 模型兼容性
- 评估协议假设使用CLIPBasedMultiModalReIDModel
- 确保模型的fusion_dim与配置一致
- 检查分类器的num_classes设置

### 3. 随机性控制
- 使用固定种子确保结果可复现
- 查询构建的随机性是协议要求的
- 主模态选择策略影响结果

### 4. 内存管理
- Gallery特征会占用较多内存
- 大数据集建议使用缓存机制
- 必要时可以分批处理

## 🎯 最佳实践

1. **训练完成后立即评估**: 确保模型状态一致
2. **保存评估结果**: 便于后续分析和比较
3. **监控各模态性能**: 找出性能瓶颈
4. **调优权重配置**: 根据模态表现调整权重
5. **验证提交格式**: 确保Kaggle提交无误

---

## 🔚 总结

本评估协议完全按照您提供的操作细则实现，与您的CLIP+MER架构无缝集成。通过固定随机种子和规范化的评估流程，确保了结果的可复现性和公平性。

如果在使用过程中遇到任何问题，请参考故障排除部分或查看相关代码注释。

*最后更新: 2024年*
