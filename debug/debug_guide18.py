# Guide18自检函数：验证模态规范化修复效果
from collections import Counter
from configs.config import TrainingConfig
from datasets.dataset import MultiModalDataset, infer_modalities_of_sample, canon_mod, IMG_MODALITIES

def quick_scan(ds, limit=500, include_text=True):
    """Guide18推荐的自检函数"""
    c = Counter()
    has_pair = 0
    total = min(limit, len(ds))
    print(f"正在扫描前{total}个样本...")
    
    for i in range(total):
        m_img = infer_modalities_of_sample(ds, i, include_text=False)
        m_all = infer_modalities_of_sample(ds, i, include_text=include_text)
        
        # 确保规范化
        m_img = {canon_mod(x) for x in m_img}
        m_all = {canon_mod(x) for x in m_all}
        
        # 统计图像模态
        for m in m_img:
            if m in IMG_MODALITIES:
                c[m] += 1
        
        # 检查配对能力
        vis = 'vis' in m_img
        nonvis = bool(m_img & {'nir','sk','cp'}) or ('text' in m_all)
        has_pair += int(vis and nonvis)
    
    print(f"\n=== Guide18修复后的模态统计 ===")
    print("[仅图像] 模态出现次数:", {k: v for k, v in c.items() if k in IMG_MODALITIES})
    if include_text:
        text_count = sum(1 for i in range(total) if 'text' in infer_modalities_of_sample(ds, i, include_text=True))
        print(f"[包含文本] text出现次数: {text_count}")
    print(f"有 vis+非vis 配对的样本数: {has_pair}/{total}")
    print(f"配对比例: {has_pair/total:.1%}")
    
    # 验证是否还有旧名称泄漏
    old_names = {'rgb', 'ir', 'sketch', 'cpencil'}
    leaked_old_names = {k: v for k, v in c.items() if k in old_names}
    if leaked_old_names:
        print(f"⚠️ 检测到旧模态名称泄漏: {leaked_old_names}")
    else:
        print("✅ 未检测到旧模态名称，规范化成功")

def test_sampler_compatibility():
    """测试采样器兼容性"""
    print(f"\n=== 测试采样器兼容性 ===")
    config = TrainingConfig()
    
    # 创建数据集
    full_dataset = MultiModalDataset(config, split='train')
    
    # 测试新采样器
    from datasets.dataset import ModalAwarePKSampler_Strict
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    import torch
    
    # 创建训练子集
    indices = list(range(len(full_dataset)))
    train_indices, _ = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    
    # 使用Guide18改进的采样器
    try:
        sampler = ModalAwarePKSampler_Strict(
            train_dataset,
            num_ids_per_batch=config.num_ids_per_batch,
            num_instances=config.num_instances,
            allow_id_reuse=config.allow_id_reuse,
            min_modal_coverage=config.min_modal_coverage,
            include_text=True
        )
        print(f"✅ 采样器创建成功")
        print(f"   采样器长度: {len(sampler)}")
        
        # 测试生成batch
        batch_count = 0
        for batch_indices in sampler:
            batch_count += 1
            if batch_count >= 5:  # 只测试前5个batch
                break
        print(f"✅ 成功生成{batch_count}个batch")
        
    except Exception as e:
        print(f"❌ 采样器测试失败: {e}")

if __name__ == "__main__":
    print("=== Guide18修复效果验证 ===")
    
    config = TrainingConfig()
    full_dataset = MultiModalDataset(config, split='train')
    
    # 执行自检
    quick_scan(full_dataset, limit=1000, include_text=True)
    
    # 测试采样器
    test_sampler_compatibility()
    
    print("\n=== 验证完成 ===")