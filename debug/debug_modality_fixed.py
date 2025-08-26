# 测试guide17修复后的模态检测
from configs.config import TrainingConfig
from datasets.dataset import MultiModalDataset, infer_modalities_of_sample

def test_fixed_modality_detection():
    """测试修复后的模态检测"""
    print("=== guide17修复后的模态检测测试 ===")
    
    config = TrainingConfig()
    full_dataset = MultiModalDataset(config, split='train')
    print(f"完整数据集大小: {len(full_dataset)}")
    
    print("\n=== 前10个样本的模态检测结果 ===")
    for i in range(min(10, len(full_dataset))):
        # 分别测试仅图像模态和包含文本的模态
        img_mods = infer_modalities_of_sample(full_dataset, i, include_text=False)
        all_mods = infer_modalities_of_sample(full_dataset, i, include_text=True)
        
        print(f"样本 {i}:")
        print(f"  仅图像模态: {img_mods}")
        print(f"  包含文本模态: {all_mods}")
        
        # 检查vis+非vis配对
        has_vis = 'vis' in img_mods
        has_nonvis = any(m in img_mods for m in ['nir', 'sk', 'cp'])
        print(f"  vis+非vis配对: {has_vis and has_nonvis}")
        print()
    
    # 统计整个数据集的模态分布
    print("=== 整个数据集模态统计 ===")
    img_modality_stats = {}
    all_modality_stats = {}
    vis_nonvis_pairs = 0
    
    # 只统计前1000个样本，避免过长时间
    sample_count = min(1000, len(full_dataset))
    print(f"统计前{sample_count}个样本...")
    
    for i in range(sample_count):
        # 仅图像模态统计
        img_mods = infer_modalities_of_sample(full_dataset, i, include_text=False)
        for mod in img_mods:
            img_modality_stats[mod] = img_modality_stats.get(mod, 0) + 1
        
        # 包含文本模态统计
        all_mods = infer_modalities_of_sample(full_dataset, i, include_text=True)
        for mod in all_mods:
            all_modality_stats[mod] = all_modality_stats.get(mod, 0) + 1
        
        # 检查vis+非vis配对（基于图像模态）
        has_vis = 'vis' in img_mods
        has_nonvis = any(m in img_mods for m in ['nir', 'sk', 'cp'])
        if has_vis and has_nonvis:
            vis_nonvis_pairs += 1
    
    print("[仅图像] 模态出现次数:")
    for mod, count in img_modality_stats.items():
        print(f"  {mod}: {count}")
    
    print("[包含文本] 模态出现次数:")
    for mod, count in all_modality_stats.items():
        print(f"  {mod}: {count}")
    
    print(f"\n有vis+非vis配对的样本数: {vis_nonvis_pairs}")
    print(f"配对比例: {vis_nonvis_pairs/sample_count*100:.1f}%")

if __name__ == "__main__":
    test_fixed_modality_detection()