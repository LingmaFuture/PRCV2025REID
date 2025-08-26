# models/patch_embeds.py
"""
多模态非共享PatchEmbeds实现
各模态独立的patch embedding层，从CLIP初始化但训练时不共享参数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class PatchEmbed(nn.Module):
    """单模态Patch Embedding层"""
    
    def __init__(
        self, 
        in_chans: int = 3, 
        embed_dim: int = 768, 
        patch_size: int = 16,
        img_size: int = 224,
        bias: bool = True
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 核心投影层：卷积实现patch embedding
        self.proj = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
            bias=bias
        )
        
        # 单通道到三通道的适配器（用于nir和sk）
        if in_chans == 1:
            self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            # 初始化为平均值，让单通道均匀分布到三通道
            nn.init.constant_(self.channel_adapter.weight, 1.0/3.0)
        else:
            self.channel_adapter = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 输入图像tensor
        Returns:
            [B, num_patches, embed_dim] patch embeddings
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # 检查通道数是否匹配
        expected_chans = self.proj.weight.shape[1]  # 卷积层期望的输入通道数
        
        if C != expected_chans:
            if self.channel_adapter is not None and C == 1 and expected_chans == 3:
                # 单通道适配：1通道 -> 3通道
                x = self.channel_adapter(x)  # [B,1,H,W] -> [B,3,H,W]
            elif C == 3 and expected_chans == 1:
                # 3通道适配：取灰度图（RGB -> 灰度）
                x = torch.mean(x, dim=1, keepdim=True)  # [B,3,H,W] -> [B,1,H,W]
            else:
                raise ValueError(
                    f"Channel mismatch: input has {C} channels, "
                    f"but PatchEmbed expects {expected_chans} channels"
                )
        
        # Patch embedding: [B,C,H,W] -> [B,embed_dim,H/P,W/P] -> [B,embed_dim,N] -> [B,N,embed_dim]
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        return x
    
    def load_clip_weights(self, clip_patch_embed_weight: torch.Tensor, clip_bias: Optional[torch.Tensor] = None):
        """从CLIP的patch embedding权重初始化"""
        with torch.no_grad():
            # 处理通道数不匹配的情况
            clip_in_chans = clip_patch_embed_weight.shape[1]  # CLIP通常是3通道
            self_in_chans = self.proj.weight.shape[1]         # 当前层的输入通道数
            
            if clip_in_chans == self_in_chans:
                # 通道数匹配，直接复制
                self.proj.weight.copy_(clip_patch_embed_weight)
            elif self_in_chans == 1 and clip_in_chans == 3:
                # 单通道模态：取vis三通道的平均值作为单通道权重
                averaged_weight = clip_patch_embed_weight.mean(dim=1, keepdim=True)  # [768, 1, 16, 16]
                self.proj.weight.copy_(averaged_weight)
            elif self_in_chans == 3 and clip_in_chans == 3:
                # 三通道模态：直接复制
                self.proj.weight.copy_(clip_patch_embed_weight)
            else:
                # 其他情况：使用CLIP权重的第一个通道进行初始化，然后复制到所有通道
                first_channel = clip_patch_embed_weight[:, 0:1, :, :]  # [768, 1, 16, 16]
                if self_in_chans == 1:
                    self.proj.weight.copy_(first_channel)
                else:
                    # 复制到多个通道
                    self.proj.weight.copy_(first_channel.repeat(1, self_in_chans, 1, 1))
            
            # 复制bias
            if clip_bias is not None and self.proj.bias is not None:
                self.proj.bias.copy_(clip_bias)


class MultiModalPatchEmbeds(nn.Module):
    """多模态非共享Patch Embeddings"""
    
    def __init__(
        self, 
        embed_dim: int = 768, 
        patch_size: int = 16,
        img_size: int = 224
    ):
        super().__init__()
        
        # 各模态独立的patch embedding（结构相同但参数不共享）
        # 使用统一的模态命名，与config.modalities保持一致
        self.vis = PatchEmbed(in_chans=3, embed_dim=embed_dim, patch_size=patch_size, img_size=img_size)
        self.nir = PatchEmbed(in_chans=1, embed_dim=embed_dim, patch_size=patch_size, img_size=img_size)        
        self.cp = PatchEmbed(in_chans=3, embed_dim=embed_dim, patch_size=patch_size, img_size=img_size)
        self.sk = PatchEmbed(in_chans=1, embed_dim=embed_dim, patch_size=patch_size, img_size=img_size)
        
        self.modality_names = ['vis', 'nir', 'cp', 'sk']
        
    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        根据模态选择对应的patch embedding
        Args:
            x: [B, C, H, W] 输入图像
            modality: 模态名称，'vis'|'nir'|'cp'|'sk'
        Returns:
            [B, num_patches, embed_dim] patch embeddings
        """
        if modality not in self.modality_names:
            raise ValueError(f"Unknown modality: {modality}. Supported: {self.modality_names}")
        
        patch_embed = getattr(self, modality)
        return patch_embed(x)
    
    def get_patch_embed(self, modality: str) -> PatchEmbed:
        """获取指定模态的patch embedding模块"""
        if modality not in self.modality_names:
            raise ValueError(f"Unknown modality: {modality}. Supported: {self.modality_names}")
        return getattr(self, modality)
    
    def load_clip_weights(self, clip_patch_embed_weight: torch.Tensor, clip_bias: Optional[torch.Tensor] = None):
        """
        从CLIP权重初始化所有模态的patch embedding
        每个模态从相同CLIP权重初始化，但后续训练独立更新
        """
        for modality in self.modality_names:
            patch_embed = self.get_patch_embed(modality)
            patch_embed.load_clip_weights(clip_patch_embed_weight, clip_bias)
            
            # 为不同模态添加小的随机扰动，避免完全相同
            if modality != 'vis':  # vis保持原始权重，其他模态添加噪声
                with torch.no_grad():
                    noise = torch.randn_like(patch_embed.proj.weight) * 0.02
                    patch_embed.proj.weight.add_(noise)
                    
                    if patch_embed.proj.bias is not None:
                        bias_noise = torch.randn_like(patch_embed.proj.bias) * 0.01
                        patch_embed.proj.bias.add_(bias_noise)
    
    def get_num_patches(self) -> int:
        """返回patch数量"""
        return self.vis.num_patches


if __name__ == "__main__":
    # 简单测试
    import torch
    
    # 创建多模态patch embeddings
    patch_embeds = MultiModalPatchEmbeds(embed_dim=768, patch_size=16, img_size=224)
    
    # 测试各模态
    vis_img = torch.randn(2, 3, 224, 224)  # 可见光图像
    nir_img = torch.randn(2, 1, 224, 224)   # 红外图像
    
    vis_patches = patch_embeds(vis_img, 'vis')
    nir_patches = patch_embeds(nir_img, 'nir')
    
    print(f"vis patches shape: {vis_patches.shape}")  # [2, 196, 768]
    print(f"nir patches shape: {nir_patches.shape}")    # [2, 196, 768]
    
    print("✅ 多模态PatchEmbeds测试通过！")
