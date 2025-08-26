# tools/check_clip_projections.py
"""检查CLIP模型的投影层"""
import torch
from transformers import CLIPModel

def check_clip_projections():
    print("🔍 检查CLIP投影层...")
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    
    print("\n📊 模型顶层属性:")
    for attr in dir(clip_model):
        if not attr.startswith('_') and 'proj' in attr.lower():
            print(f"  - {attr}: {type(getattr(clip_model, attr))}")
    
    print("\n📊 检查具体投影层:")
    attrs_to_check = [
        'visual_projection', 'text_projection', 
        'vision_projection', 'text_proj',
        'vision_proj', 'visual_proj'
    ]
    
    for attr in attrs_to_check:
        if hasattr(clip_model, attr):
            proj = getattr(clip_model, attr)
            print(f"  ✅ {attr}: {type(proj)} - shape: {proj.weight.shape if hasattr(proj, 'weight') else 'N/A'}")
        else:
            print(f"  ❌ {attr}: 不存在")
    
    # 检查vision model和text model是否有投影层
    print("\n📊 Vision Model 投影层:")
    vision_model = clip_model.vision_model
    for attr in ['projection', 'proj', 'final_proj']:
        if hasattr(vision_model, attr):
            proj = getattr(vision_model, attr)
            print(f"  ✅ vision_model.{attr}: {type(proj)}")
    
    print("\n📊 Text Model 投影层:")
    text_model = clip_model.text_model
    for attr in ['projection', 'proj', 'final_proj']:
        if hasattr(text_model, attr):
            proj = getattr(text_model, attr)
            print(f"  ✅ text_model.{attr}: {type(proj)}")

if __name__ == "__main__":
    check_clip_projections()
