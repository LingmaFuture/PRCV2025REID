# tools/test_fixes.py
"""
测试修复效果的脚本
验证数据集划分和SDM损失修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
from configs.config import TrainingConfig
from datasets.dataset import MultiModalDataset
from tools.split import split_ids, create_split_datasets, verify_split_integrity
from models.sdm_loss import sdm_loss_stable, SDMLoss

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def test_dataset_split():
    """测试数据集划分修复"""
    print("=" * 50)
    print("测试数据集划分修复")
    print("=" * 50)
    
    # 加载配置
    config = TrainingConfig()
    
    # 加载完整数据集
    print("加载完整数据集...")
    full_dataset = MultiModalDataset(config, split='train')
    
    # 获取所有人员ID
    all_person_ids = [full_dataset.data_list[i]['person_id'] for i in range(len(full_dataset))]
    all_person_ids = sorted(list(set(all_person_ids)))
    print(f"总人员ID数: {len(all_person_ids)}")
    print(f"总样本数: {len(full_dataset.data_list)}")
    
    # 测试划分
    print("\n测试数据集划分...")
    train_ids, val_ids = split_ids(all_person_ids, val_ratio=0.2, seed=42)
    
    print(f"训练集ID数: {len(train_ids)}")
    print(f"验证集ID数: {len(val_ids)}")
    print(f"验证集比例: {len(val_ids)/len(all_person_ids):.3f}")
    
    # 验证互斥性
    common_ids = train_ids & val_ids
    if len(common_ids) == 0:
        print("✅ 训练集和验证集ID互斥")
    else:
        print(f"❌ 训练集和验证集存在共同ID: {common_ids}")
        return False
    
    # 创建划分后的数据集
    print("\n创建划分后的数据集...")
    train_dataset, val_dataset = create_split_datasets(full_dataset, train_ids, val_ids, config)
    
    # 验证完整性
    print("\n验证数据集完整性...")
    verify_split_integrity(train_dataset, val_dataset)
    
    # 检查样本数
    expected_train_samples = sum(1 for item in full_dataset.data_list if item['person_id'] in train_ids)
    expected_val_samples = sum(1 for item in full_dataset.data_list if item['person_id'] in val_ids)
    
    print(f"\n样本数验证:")
    print(f"  训练集: {len(train_dataset.data_list)} (期望: {expected_train_samples})")
    print(f"  验证集: {len(val_dataset.data_list)} (期望: {expected_val_samples})")
    print(f"  总计: {len(train_dataset.data_list) + len(val_dataset.data_list)} (期望: {len(full_dataset.data_list)})")
    
    if (len(train_dataset.data_list) == expected_train_samples and 
        len(val_dataset.data_list) == expected_val_samples):
        print("✅ 数据集划分样本数正确")
        return True
    else:
        print("❌ 数据集划分样本数不正确")
        return False


def test_sdm_loss():
    """测试SDM损失修复"""
    print("\n" + "=" * 50)
    print("测试SDM损失修复")
    print("=" * 50)
    
    # 测试数据
    batch_size = 16
    feature_dim = 512
    num_classes = 8
    
    print(f"测试参数: batch_size={batch_size}, feature_dim={feature_dim}, num_classes={num_classes}")
    
    # 生成测试数据
    qry_features = torch.randn(batch_size, feature_dim)
    gal_features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 构造同身份指示矩阵
    labels_qry = labels.view(-1, 1)
    labels_gal = labels.view(1, -1)
    y = (labels_qry == labels_gal).float()
    
    print(f"正样本数量: {y.sum().item()}")
    
    # 测试函数版本
    print("\n测试函数版本SDM损失...")
    loss_func = sdm_loss_stable(qry_features, gal_features, y, tau=0.1)
    print(f"函数版本损失: {loss_func.item():.6f}")
    
    # 测试模块版本
    print("\n测试模块版本SDM损失...")
    sdm_module = SDMLoss(temperature=0.1)
    loss_module, details = sdm_module(qry_features, gal_features, labels, return_details=True)
    print(f"模块版本损失: {loss_module.item():.6f}")
    print(f"详细信息: {details}")
    
    # 验证损失为正
    if loss_func.item() >= 0 and loss_module.item() >= 0:
        print("✅ SDM损失计算正确（非负值）")
        return True
    else:
        print("❌ SDM损失计算错误（出现负值）")
        return False


def test_extreme_cases():
    """测试极端情况"""
    print("\n" + "=" * 50)
    print("测试极端情况")
    print("=" * 50)
    
    # 测试全零特征
    print("测试全零特征...")
    batch_size = 8
    feature_dim = 512
    qry_features = torch.zeros(batch_size, feature_dim)
    gal_features = torch.zeros(batch_size, feature_dim)
    labels = torch.randint(0, 4, (batch_size,))
    
    labels_qry = labels.view(-1, 1)
    labels_gal = labels.view(1, -1)
    y = (labels_qry == labels_gal).float()
    
    try:
        loss = sdm_loss_stable(qry_features, gal_features, y, tau=0.1)
        print(f"全零特征损失: {loss.item():.6f}")
        print("✅ 全零特征处理正常")
    except Exception as e:
        print(f"❌ 全零特征处理异常: {e}")
        return False
    
    # 测试单样本
    print("\n测试单样本...")
    qry_features = torch.randn(1, feature_dim)
    gal_features = torch.randn(1, feature_dim)
    labels = torch.tensor([0])
    
    labels_qry = labels.view(-1, 1)
    labels_gal = labels.view(1, -1)
    y = (labels_qry == labels_gal).float()
    
    try:
        loss = sdm_loss_stable(qry_features, gal_features, y, tau=0.1)
        print(f"单样本损失: {loss.item():.6f}")
        print("✅ 单样本处理正常")
    except Exception as e:
        print(f"❌ 单样本处理异常: {e}")
        return False
    
    return True


def main():
    """主测试函数"""
    print("开始测试修复效果...")
    
    # 测试数据集划分
    split_ok = test_dataset_split()
    
    # 测试SDM损失
    sdm_ok = test_sdm_loss()
    
    # 测试极端情况
    extreme_ok = test_extreme_cases()
    
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    print(f"数据集划分: {'✅ 通过' if split_ok else '❌ 失败'}")
    print(f"SDM损失修复: {'✅ 通过' if sdm_ok else '❌ 失败'}")
    print(f"极端情况处理: {'✅ 通过' if extreme_ok else '❌ 失败'}")
    
    if split_ok and sdm_ok and extreme_ok:
        print("\n🎉 所有测试通过！修复成功！")
        return True
    else:
        print("\n⚠️ 部分测试失败，需要进一步检查")
        return False


if __name__ == "__main__":
    main()
