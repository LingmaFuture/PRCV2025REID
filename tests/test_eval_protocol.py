# tools/test_eval_protocol.py
"""
测试多模态评估协议
验证各个组件是否正常工作
"""

import os
import sys
import torch
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import TrainingConfig
from datasets.dataset import MultiModalDataset
from models.model import CLIPBasedMultiModalReIDModel
from tools.eval_mm_protocol import (
    build_index, build_gallery, build_queries, 
    FeatureExtractor, extract_gallery_feats,
    extract_query_feat, rank_and_metrics
)

def test_data_loading():
    """测试数据加载和索引构建"""
    print("=== 测试数据加载和索引构建 ===")
    
    try:
        # 创建配置
        config = TrainingConfig()
        
        # 检查数据路径
        if not os.path.exists(config.data_root):
            print(f"⚠️ 数据路径不存在: {config.data_root}")
            print("请确保数据集已正确放置")
            return False
        
        if not os.path.exists(config.json_file):
            print(f"⚠️ 标注文件不存在: {config.json_file}")
            print("请确保JSON标注文件已正确放置")
            return False
        
        # 加载数据集（使用较少的数据进行测试）
        dataset = MultiModalDataset(config, split='val')
        
        # 构建数据索引
        index = build_index(dataset)
        
        print(f"✅ 数据索引构建成功，共 {len(index)} 个身份")
        
        # 检查模态覆盖
        modality_coverage = {}
        for pid, mods_map in index.items():
            for mod, items in mods_map.items():
                if items:
                    if mod not in modality_coverage:
                        modality_coverage[mod] = 0
                    modality_coverage[mod] += 1
        
        print("模态覆盖情况:")
        for mod, count in modality_coverage.items():
            print(f"  {mod}: {count} 个身份")
        
        return index, dataset
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n=== 测试模型加载 ===")
    
    try:
        # 创建配置
        config = TrainingConfig()
        
        # 创建模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        model = CLIPBasedMultiModalReIDModel(config).to(device)
        model.set_num_classes(100)  # 假设100个类别
        
        print("✅ 模型创建成功")
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            # 测试图像输入
            test_images = {
                'rgb': torch.randn(2, 3, 224, 224).to(device),
                'ir': torch.randn(2, 1, 224, 224).to(device)
            }
            test_texts = ["测试文本1", "测试文本2"]
            
            outputs = model(images=test_images, texts=test_texts)
            
            print(f"✅ 前向传播成功")
            print(f"  特征维度: {outputs['features'].shape}")
            print(f"  分类输出维度: {outputs['logits'].shape}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_feature_extraction(model, device, index):
    """测试特征提取"""
    print("\n=== 测试特征提取 ===")
    
    try:
        # 创建特征提取器
        extractor = FeatureExtractor(model, device)
        
        # 找一个有RGB图像的样本进行测试
        test_sample = None
        for pid, mods_map in index.items():
            if 'rgb' in mods_map and len(mods_map['rgb']) > 0:
                test_sample = mods_map['rgb'][0]
                break
        
        if test_sample is None:
            print("⚠️ 找不到RGB测试样本")
            return False
        
        print(f"测试样本: {test_sample['img_path']}")
        
        # 检查文件是否存在
        if not os.path.exists(test_sample['img_path']):
            print(f"⚠️ 测试图像文件不存在: {test_sample['img_path']}")
            return False
        
        # 提取特征
        feature = extractor.encode_rgb(test_sample['img_path'])
        
        print(f"✅ RGB特征提取成功，维度: {feature.shape}")
        
        # 测试文本特征提取
        text_feature = extractor.encode_text("测试文本")
        print(f"✅ 文本特征提取成功，维度: {text_feature.shape}")
        
        return extractor
        
    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_query_building(index):
    """测试查询构建"""
    print("\n=== 测试查询构建 ===")
    
    try:
        import random
        rng = random.Random(42)
        
        # 测试各种模态组合
        for k in [1, 2, 3, 4]:
            queries = build_queries(index, mode_k=k, rng=rng)
            print(f"MM-{k}: {len(queries)} 个查询")
            
            if queries:
                # 显示第一个查询的详情
                first_query = queries[0]
                print(f"  示例查询: PID={first_query['pid']}, "
                      f"模态={first_query['modalities']}")
        
        print("✅ 查询构建成功")
        return True
        
    except Exception as e:
        print(f"❌ 查询构建失败: {e}")
        return False

def test_gallery_building(index):
    """测试Gallery构建"""
    print("\n=== 测试Gallery构建 ===")
    
    try:
        gallery = build_gallery(index)
        print(f"✅ Gallery构建成功，共 {len(gallery)} 张RGB图像")
        
        if gallery:
            print(f"  示例Gallery项: {gallery[0]}")
        
        return gallery
        
    except Exception as e:
        print(f"❌ Gallery构建失败: {e}")
        return False

def test_small_evaluation(index, extractor, gallery):
    """测试小规模评估"""
    print("\n=== 测试小规模评估 ===")
    
    try:
        # 创建临时缓存目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 只取前5个Gallery样本进行测试
            small_gallery = gallery[:5] if len(gallery) > 5 else gallery
            
            if not small_gallery:
                print("⚠️ 没有可用的Gallery样本")
                return False
            
            print(f"测试Gallery大小: {len(small_gallery)}")
            
            # 提取Gallery特征（小规模）
            print("提取Gallery特征...")
            
            feats, meta = [], []
            for item in small_gallery:
                try:
                    if os.path.exists(item["img_path"]):
                        f = extractor.encode_rgb(item["img_path"])
                        f = torch.nn.functional.normalize(f.float().view(1, -1)).squeeze(0)
                        feats.append(f.cpu())
                        meta.append({
                            "img_id": item.get("img_id", None),
                            "pid": int(item["pid"]),
                            "camid": item.get("camid", None)
                        })
                    else:
                        print(f"⚠️ 图像文件不存在: {item['img_path']}")
                except Exception as e:
                    print(f"⚠️ 提取特征失败 {item['img_path']}: {e}")
            
            if not feats:
                print("⚠️ 没有成功提取的Gallery特征")
                return False
            
            gallery_feats = torch.stack(feats, 0)
            print(f"✅ Gallery特征提取成功，形状: {gallery_feats.shape}")
            
            # 构建查询（只测试MM-1）
            import random
            rng = random.Random(42)
            queries = build_queries(index, mode_k=1, rng=rng)
            
            # 只取前2个查询进行测试
            test_queries = queries[:2] if len(queries) > 2 else queries
            
            if not test_queries:
                print("⚠️ 没有可用的查询样本")
                return False
            
            print(f"测试查询数量: {len(test_queries)}")
            
            # 简单的评估测试
            weight_cfg = {"ir": 1.0, "cpencil": 1.0, "sketch": 1.0, "text": 1.2}
            
            for i, q in enumerate(test_queries):
                try:
                    print(f"测试查询 {i+1}: PID={q['pid']}, 模态={q['modalities']}")
                    
                    # 提取查询特征
                    q_feat = extract_query_feat(q, extractor, weight_cfg)
                    print(f"  查询特征维度: {q_feat.shape}")
                    
                    # 计算相似度
                    q_feat_norm = torch.nn.functional.normalize(q_feat.view(1, -1), dim=-1)
                    gallery_feats_norm = torch.nn.functional.normalize(gallery_feats, dim=-1)
                    sims = q_feat_norm @ gallery_feats_norm.T
                    
                    print(f"  相似度范围: {sims.min().item():.3f} ~ {sims.max().item():.3f}")
                    
                except Exception as e:
                    print(f"  ❌ 查询 {i+1} 失败: {e}")
            
            print("✅ 小规模评估测试完成")
            return True
            
    except Exception as e:
        print(f"❌ 小规模评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 开始测试多模态评估协议")
    print("=" * 50)
    
    # 1. 测试数据加载
    data_result = test_data_loading()
    if not data_result:
        print("❌ 数据加载测试失败，请检查数据集路径和格式")
        return
    
    index, dataset = data_result
    
    # 2. 测试模型加载
    model_result = test_model_loading()
    if not model_result:
        print("❌ 模型加载测试失败")
        return
    
    model, device = model_result
    
    # 3. 测试特征提取
    extractor = test_feature_extraction(model, device, index)
    if not extractor:
        print("❌ 特征提取测试失败")
        return
    
    # 4. 测试查询构建
    if not test_query_building(index):
        print("❌ 查询构建测试失败")
        return
    
    # 5. 测试Gallery构建
    gallery = test_gallery_building(index)
    if not gallery:
        print("❌ Gallery构建测试失败")
        return
    
    # 6. 测试小规模评估
    if not test_small_evaluation(index, extractor, gallery):
        print("❌ 小规模评估测试失败")
        return
    
    print("\n" + "=" * 50)
    print("🎉 所有测试都通过！评估协议可以正常使用")
    print("=" * 50)
    
    print("\n📝 使用说明:")
    print("1. 确保已训练好模型并保存权重")
    print("2. 运行完整评估:")
    print("   python tools/eval_mm_protocol.py --dataset_root ./data/train --model_path ./checkpoints/best_model.pth")
    print("3. 如需要，可以修改权重配置和其他参数")

if __name__ == "__main__":
    main()
