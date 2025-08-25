# 测试修复后的采样器
import torch
from configs.config import TrainingConfig
from datasets.dataset import MultiModalDataset, ModalAwarePKSampler_Strict, compatible_collate_fn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def test_sampler():
    print("=== 测试修复后的采样器 ===")
    
    config = TrainingConfig()
    
    # 创建数据集
    full_dataset = MultiModalDataset(config, split='train')
    print(f"完整数据集大小: {len(full_dataset)}")
    
    # 划分训练集
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=config.val_ratio, random_state=config.seed)
    
    # 创建训练子集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    print(f"训练集大小: {len(train_dataset)}")
    
    # 创建改进的采样器
    sampler = ModalAwarePKSampler_Strict(
        train_dataset,
        batch_size=config.batch_size,
        num_instances=config.num_instances,
        seed=config.seed
    )
    
    print(f"采样器长度: {len(sampler)}")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,  # 设为0避免多进程问题
        collate_fn=compatible_collate_fn,
        pin_memory=False
    )
    
    print(f"DataLoader长度: {len(train_loader)}")
    
    # 测试前几个batch
    batch_count = 0
    for batch_idx, batch in enumerate(train_loader):
        batch_count += 1
        print(f"Batch {batch_idx + 1}: person_ids shape = {batch['person_id'].shape}")
        
        if batch_idx >= 4:  # 只测试前5个batch
            break
    
    print(f"成功生成 {batch_count} 个batch")
    print("测试完成！")

if __name__ == "__main__":
    test_sampler()