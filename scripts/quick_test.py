# 快速测试模态检测
from configs.config import TrainingConfig
from datasets.dataset import MultiModalDataset, infer_modalities_of_sample

config = TrainingConfig()
full_dataset = MultiModalDataset(config, split='train')
print(f"数据集大小: {len(full_dataset)}")

# 测试一个样本
sample = full_dataset[0]
print(f"样本0模态检测结果: {infer_modalities_of_sample(full_dataset, 0)}")
print(f"样本0的modality_mask: {sample['modality_mask']}")

# 计数前100个样本的可配对情况
pairable_count = 0
for i in range(min(100, len(full_dataset))):
    mods = infer_modalities_of_sample(full_dataset, i)
    has_vis = 'vis' in mods
    has_nonvis = any(m in mods for m in ['nir', 'sk', 'cp', 'text'])
    if has_vis and has_nonvis:
        pairable_count += 1

print(f"前100个样本中可配对的数量: {pairable_count}")