# tools/check_clip_tokens.py
"""检查CLIP的token形状"""
import torch
from transformers import CLIPModel

def check_clip_tokens():
    print("🔍 检查CLIP token形状...")
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    
    # 检查vision model的embeddings
    vision_embeddings = clip_model.vision_model.embeddings
    
    print(f"📊 Vision Embeddings:")
    for name, param in vision_embeddings.named_parameters():
        print(f"  - {name}: {param.shape}")
    
    # 检查class embedding
    if hasattr(vision_embeddings, 'class_embedding'):
        cls_embedding = vision_embeddings.class_embedding
        print(f"  - class_embedding shape: {cls_embedding.shape}")
        print(f"  - class_embedding ndim: {cls_embedding.ndim}")
    
    # 检查position embedding
    if hasattr(vision_embeddings, 'position_embedding'):
        pos_embedding = vision_embeddings.position_embedding
        print(f"  - position_embedding shape: {pos_embedding.weight.shape}")

if __name__ == "__main__":
    check_clip_tokens()
