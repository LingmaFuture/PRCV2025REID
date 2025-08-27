#!/usr/bin/env python3
"""
PreNorm=0 快速排查调试脚本
用于诊断embedding塌缩问题，不修改训练代码
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_config
from datasets.dataset import build_dataloader
from models.model import build_model
from train import setup_logging

class PreNormDebugger:
    def __init__(self, config_path=None):
        """初始化调试器"""
        self.config = get_config(config_path) if config_path else get_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置日志
        self.logger = self._setup_debug_logger()
        
        # 初始化模型和数据
        self.model = None
        self.train_loader = None
        self.val_loader = None
        
    def _setup_debug_logger(self):
        """设置调试日志"""
        logger = logging.getLogger('PreNormDebugger')
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建文件处理器
        debug_dir = Path('debug_logs')
        debug_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            debug_dir / f'prenorm_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_model_and_data(self):
        """加载模型和数据"""
        self.logger.info("正在加载模型和数据...")
        
        # 构建数据加载器
        self.train_loader, self.val_loader = build_dataloader(self.config)
        self.logger.info(f"训练集大小: {len(self.train_loader.dataset)}")
        self.logger.info(f"验证集大小: {len(self.val_loader.dataset)}")
        
        # 构建模型
        self.model = build_model(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("模型和数据加载完成")
        
    def check_batch_data(self, batch, batch_idx=0):
        """检查批次数据"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"批次 {batch_idx} 数据检查")
        self.logger.info(f"{'='*50}")
        
        # 1. 检查基本形状
        if 'images' in batch:
            self.logger.info(f"图像形状: {batch['images'].shape}")
            self.logger.info(f"图像数据类型: {batch['images'].dtype}")
            self.logger.info(f"图像值范围: [{batch['images'].min():.4f}, {batch['images'].max():.4f}]")
            self.logger.info(f"图像均值: {batch['images'].mean():.4f}")
            self.logger.info(f"图像标准差: {batch['images'].std():.4f}")
            
            # 检查是否有全黑图像
            zero_images = (batch['images'].sum(dim=[1,2,3]) == 0).sum().item()
            self.logger.info(f"全黑图像数量: {zero_images}")
            
        # 2. 检查模态掩码
        if 'modality_mask' in batch:
            self.logger.info(f"模态掩码形状: {batch['modality_mask'].shape}")
            self.logger.info(f"模态掩码: {batch['modality_mask']}")
            
            # 检查各模态是否存在
            modality_names = ['vis', 'nir', 'sk', 'text']
            for i, name in enumerate(modality_names):
                if i < batch['modality_mask'].shape[1]:
                    count = batch['modality_mask'][:, i].sum().item()
                    self.logger.info(f"{name} 模态样本数: {count}")
                    
        # 3. 检查人员ID
        if 'person_id' in batch:
            self.logger.info(f"人员ID形状: {batch['person_id'].shape}")
            self.logger.info(f"人员ID: {batch['person_id']}")
            self.logger.info(f"唯一人员ID数量: {len(torch.unique(batch['person_id']))}")
            
        # 4. 检查文本数据
        if 'text' in batch:
            self.logger.info(f"文本数据形状: {batch['text'].shape}")
            self.logger.info(f"文本示例: {batch['text'][:3]}")
            
    def check_model_structure(self):
        """检查模型结构"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info("模型结构检查")
        self.logger.info(f"{'='*50}")
        
        if self.model is None:
            self.logger.error("模型未加载")
            return
            
        # 1. 检查模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"总参数数量: {total_params:,}")
        self.logger.info(f"可训练参数数量: {trainable_params:,}")
        
        # 2. 检查关键层
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d)):
                self.logger.info(f"发现归一化层: {name}")
                if hasattr(module, 'weight'):
                    self.logger.info(f"  - weight均值: {module.weight.mean().item():.4f}")
                    self.logger.info(f"  - weight标准差: {module.weight.std().item():.4f}")
                if hasattr(module, 'bias') and module.bias is not None:
                    self.logger.info(f"  - bias均值: {module.bias.mean().item():.4f}")
                    
            elif isinstance(module, torch.nn.Dropout):
                self.logger.info(f"发现Dropout层: {name}, p={module.p}")
                
    def forward_with_debug(self, batch):
        """带调试信息的前向传播"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info("前向传播调试")
        self.logger.info(f"{'='*50}")
        
        # 将数据移到设备
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # 注册钩子来捕获中间输出
        activations = {}
        
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach()
                elif isinstance(output, dict):
                    activations[name] = {k: v.detach() if isinstance(v, torch.Tensor) else v 
                                       for k, v in output.items()}
            return hook
        
        # 注册钩子到关键层
        hooks = []
        for name, module in self.model.named_modules():
            if any(keyword in name for keyword in ['backbone', 'fusion', 'bnneck', 'classifier']):
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)
        
        try:
            # 前向传播
            with torch.no_grad():
                outputs = self.model(batch)
            
            # 分析输出
            self.logger.info("模型输出分析:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    self.logger.info(f"  {key}: shape={value.shape}, "
                                   f"norm={value.norm().item():.4f}, "
                                   f"mean={value.mean().item():.4f}, "
                                   f"std={value.std().item():.4f}")
                    
                    # 检查是否接近零
                    if value.norm().item() < 1e-6:
                        self.logger.warning(f"  ⚠️ {key} 范数接近零!")
                        
            # 分析中间激活
            self.logger.info("\n中间激活分析:")
            for name, activation in activations.items():
                if isinstance(activation, torch.Tensor):
                    self.logger.info(f"  {name}: norm={activation.norm().item():.4f}, "
                                   f"mean={activation.mean().item():.4f}")
                    
                    if activation.norm().item() < 1e-6:
                        self.logger.warning(f"  ⚠️ {name} 激活接近零!")
                        
        finally:
            # 移除钩子
            for hook in hooks:
                hook.remove()
                
        return outputs, activations
    
    def check_loss_computation(self, batch, outputs):
        """检查损失计算"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info("损失计算检查")
        self.logger.info(f"{'='*50}")
        
        # 这里需要根据实际的损失函数实现来检查
        # 由于不修改训练代码，我们只做基本的检查
        
        if 'reid_features' in outputs:
            features = outputs['reid_features']
            self.logger.info(f"ReID特征形状: {features.shape}")
            self.logger.info(f"ReID特征范数: {features.norm().item():.4f}")
            
            # 检查特征是否归一化
            if features.shape[1] > 1:
                feature_norms = torch.norm(features, dim=1)
                self.logger.info(f"各样本特征范数: 均值={feature_norms.mean().item():.4f}, "
                               f"标准差={feature_norms.std().item():.4f}")
                
    def run_comprehensive_check(self, num_batches=3):
        """运行综合检查"""
        self.logger.info("开始PreNorm=0综合排查...")
        
        # 1. 检查模型结构
        self.check_model_structure()
        
        # 2. 检查训练数据
        self.logger.info(f"\n检查训练数据 (前{num_batches}个批次)...")
        for i, batch in enumerate(self.train_loader):
            if i >= num_batches:
                break
            self.check_batch_data(batch, i)
            
            # 3. 前向传播调试
            outputs, activations = self.forward_with_debug(batch)
            
            # 4. 损失检查
            self.check_loss_computation(batch, outputs)
            
            self.logger.info(f"\n批次 {i} 检查完成")
            
        # 5. 检查验证数据
        self.logger.info(f"\n检查验证数据 (前{num_batches}个批次)...")
        for i, batch in enumerate(self.val_loader):
            if i >= num_batches:
                break
            self.check_batch_data(batch, i)
            outputs, activations = self.forward_with_debug(batch)
            self.check_loss_computation(batch, outputs)
            
        self.logger.info("\n综合检查完成!")
        
    def check_specific_batch(self, batch_idx=0, is_train=True):
        """检查特定批次"""
        loader = self.train_loader if is_train else self.val_loader
        loader_name = "训练" if is_train else "验证"
        
        self.logger.info(f"检查{loader_name}批次 {batch_idx}...")
        
        for i, batch in enumerate(loader):
            if i == batch_idx:
                self.check_batch_data(batch, i)
                outputs, activations = self.forward_with_debug(batch)
                self.check_loss_computation(batch, outputs)
                break
                
    def generate_report(self):
        """生成调试报告"""
        report_path = Path('debug_logs') / f'prenorm_debug_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PreNorm=0 调试报告\n")
            f.write("="*50 + "\n")
            f.write(f"生成时间: {datetime.now()}\n")
            f.write(f"设备: {self.device}\n\n")
            
            # 这里可以添加更多报告内容
            
        self.logger.info(f"调试报告已生成: {report_path}")

def main():
    """主函数"""
    print("PreNorm=0 快速排查调试脚本")
    print("="*50)
    
    # 创建调试器
    debugger = PreNormDebugger()
    
    try:
        # 加载模型和数据
        debugger.load_model_and_data()
        
        # 运行综合检查
        debugger.run_comprehensive_check(num_batches=2)
        
        # 生成报告
        debugger.generate_report()
        
    except Exception as e:
        debugger.logger.error(f"调试过程中出现错误: {e}")
        import traceback
        debugger.logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
