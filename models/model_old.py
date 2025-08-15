import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer
import math

class MultiModalReIDModel(nn.Module):
    """多模态ReID模型 - 专注核心功能"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 视觉编码器 - 使用ResNet50
        self.visual_encoder = models.resnet50(pretrained=True)
        self.visual_encoder.fc = nn.Identity()  # 移除分类层
        
        # 模态适配层 - 使用LayerNorm替代BatchNorm避免batch size问题
        self.modality_adapters = nn.ModuleDict({
            'vis': nn.Sequential(
                nn.Linear(2048, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            'nir': nn.Sequential(
                nn.Linear(2048, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            'sk': nn.Sequential(
                nn.Linear(2048, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            'cp': nn.Sequential(
                nn.Linear(2048, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        })
        
        # 文本编码器 - 简化版本
        self.text_projection = nn.Sequential(
            nn.Linear(384, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 融合层 - 简单的注意力机制
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 最终融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 分类器
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        
        # 文本编码器初始化
        self._init_text_encoder()
        
    def _init_text_encoder(self):
        """初始化文本编码器"""
        try:
            self.text_encoder = AutoModel.from_pretrained(
                'sentence-transformers/all-MiniLM-L6-v2',
                local_files_only=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                'sentence-transformers/all-MiniLM-L6-v2',
                local_files_only=False
            )
        except Exception as e:
            print(f"警告: 无法加载文本编码器 {e}")
            # 使用简单的嵌入层作为备选
            self.text_encoder = None
            self.text_embedding = nn.Embedding(10000, 384)
    
    def encode_image(self, image, modality):
        """编码图像"""
        # ResNet特征提取
        features = self.visual_encoder(image)  # (batch, 2048)
        
        # 模态特异性适配
        if modality in self.modality_adapters:
            features = self.modality_adapters[modality](features)
        else:
            # 通用适配器
            features = self.modality_adapters['vis'](features)
        
        return features
    
    def encode_text(self, text_list):
        """编码文本"""
        if self.text_encoder is None:
            # 备选方案：简单词汇编码
            batch_size = len(text_list)
            device = next(self.parameters()).device
            return torch.randn(batch_size, 384, device=device)
        
        try:
            # 处理文本
            processed_texts = []
            for text in text_list:
                if isinstance(text, str) and len(text.strip()) > 0:
                    processed_texts.append(text.strip())
                else:
                    processed_texts.append("unknown person")
            
            # Tokenize
            encoded = self.tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            device = next(self.parameters()).device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # 编码
            with torch.no_grad():
                text_output = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # 池化
            text_features = text_output.last_hidden_state.mean(dim=1)  # (batch, 384)
            return text_features
            
        except Exception as e:
            print(f"文本编码失败: {e}")
            batch_size = len(text_list)
            device = next(self.parameters()).device
            return torch.randn(batch_size, 384, device=device)
    
    def forward(self, batch, return_features=False):
        """前向传播"""
        batch_size = batch['person_id'].size(0)
        device = batch['person_id'].device
        
        # 收集所有模态特征
        modal_features = []
        modality_mask = batch.get('modality_mask', {})
        
        # 处理图像模态
        for modality in ['vis', 'nir', 'sk', 'cp']:
            if modality in batch:
                # 检查模态掩码
                mask_value = modality_mask.get(modality, 1.0)
                
                if isinstance(mask_value, torch.Tensor):
                    # 批次级掩码
                    valid_samples = mask_value > 0.5
                    if valid_samples.sum() > 0:
                        valid_images = batch[modality][valid_samples]
                        valid_features = self.encode_image(valid_images, modality)
                        
                        # 补全特征到完整批次
                        full_features = torch.zeros(batch_size, self.config.hidden_dim, device=device)
                        full_features[valid_samples] = valid_features
                        modal_features.append(full_features)
                elif mask_value > 0.5:
                    # 标量掩码
                    features = self.encode_image(batch[modality], modality)
                    # 确保特征batch size正确
                    if features.size(0) != batch_size:
                        print(f"警告: {modality}模态特征batch size不匹配: {features.size(0)} vs {batch_size}")
                        if features.size(0) < batch_size:
                            # 复制特征以匹配batch size
                            features = features.repeat(batch_size // features.size(0) + 1, 1)[:batch_size]
                        else:
                            # 截断特征以匹配batch size
                            features = features[:batch_size]
                    modal_features.append(features)
        
        # 处理文本模态
        if 'text_descriptions' in batch:
            text_features = self.encode_text(batch['text_descriptions'])
            text_features = self.text_projection(text_features)
            # 确保文本特征batch size正确
            if text_features.size(0) != batch_size:
                print(f"警告: 文本特征batch size不匹配: {text_features.size(0)} vs {batch_size}")
                if text_features.size(0) < batch_size:
                    # 复制特征以匹配batch size
                    text_features = text_features.repeat(batch_size // text_features.size(0) + 1, 1)[:batch_size]
                else:
                    # 截断特征以匹配batch size
                    text_features = text_features[:batch_size]
            modal_features.append(text_features)
        
        # 模态融合
        if len(modal_features) == 0:
            # 如果没有有效模态，返回零特征
            fused_features = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        elif len(modal_features) == 1:
            # 只有一个模态
            fused_features = modal_features[0]
        else:
            # 多模态融合 - 确保所有特征batch size一致
            # 检查并修复batch size不匹配
            target_batch_size = modal_features[0].size(0)
            for i, features in enumerate(modal_features):
                if features.size(0) != target_batch_size:
                    print(f"警告: 模态{i}特征batch size不匹配: {features.size(0)} vs {target_batch_size}")
                    if features.size(0) < target_batch_size:
                        # 复制特征以匹配batch size
                        modal_features[i] = features.repeat(target_batch_size // features.size(0) + 1, 1)[:target_batch_size]
                    else:
                        # 截断特征以匹配batch size
                        modal_features[i] = features[:target_batch_size]
            
            # 堆叠特征用于注意力机制
            stacked_features = torch.stack(modal_features, dim=1)  # (batch, num_modalities, hidden_dim)
            
            # 自注意力融合
            attended_features, _ = self.fusion_attention(
                stacked_features, stacked_features, stacked_features
            )
            
            # 平均池化
            fused_features = attended_features.mean(dim=1)  # (batch, hidden_dim)
        
        # 最终融合层
        fused_features = self.fusion_layer(fused_features)
        
        if return_features:
            return F.normalize(fused_features, p=2, dim=1)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'features': fused_features
        }

class SimplifiedContrastiveLoss(nn.Module):
    """简化的对比损失"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        # 标准化特征
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # 创建正样本掩码
        labels = labels.unsqueeze(1)
        pos_mask = torch.eq(labels, labels.T).float()
        
        # 移除对角线
        eye_mask = torch.eye(labels.size(0), device=labels.device)
        pos_mask = pos_mask - eye_mask
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(similarity)
        
        # 分母：所有样本（除自身）
        denominator = torch.sum(exp_sim * (1 - eye_mask), dim=1, keepdim=True)
        
        # 分子：正样本
        numerator = torch.sum(exp_sim * pos_mask, dim=1, keepdim=True)
        
        # 只计算有正样本的损失
        valid_samples = numerator.squeeze() > 0
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        loss = -torch.log(numerator[valid_samples] / (denominator[valid_samples] + 1e-8))
        return loss.mean()
