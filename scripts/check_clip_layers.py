# tools/check_clip_layers.py
"""详细检查CLIP transformer层结构"""
import torch
from transformers import CLIPModel

def check_clip_layers():
    print("🔍 详细检查CLIP transformer层结构...")
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    
    print("\n📊 Vision Encoder第一层详细结构:")
    first_vision_layer = clip_model.vision_model.encoder.layers[0]
    for name, module in first_vision_layer.named_children():
        print(f"  - {name}: {type(module).__name__}")
        if hasattr(module, 'named_children'):
            for sub_name, sub_module in module.named_children():
                print(f"    - {sub_name}: {type(sub_module).__name__}")
    
    print("\n📊 Text Encoder第一层详细结构:")
    first_text_layer = clip_model.text_model.encoder.layers[0]
    for name, module in first_text_layer.named_children():
        print(f"  - {name}: {type(module).__name__}")
        if hasattr(module, 'named_children'):
            for sub_name, sub_module in module.named_children():
                print(f"    - {sub_name}: {type(sub_module).__name__}")
    
    print("\n📊 Vision Model预处理层:")
    vision_model = clip_model.vision_model
    attrs = ['pre_layrnorm', 'post_layernorm', 'layernorm']
    for attr in attrs:
        if hasattr(vision_model, attr):
            print(f"  ✅ {attr}: {type(getattr(vision_model, attr))}")
    
    print("\n📊 检查所有可能的layer norm属性:")
    all_attrs = dir(first_vision_layer)
    norm_attrs = [attr for attr in all_attrs if 'norm' in attr.lower() and not attr.startswith('_')]
    for attr in norm_attrs:
        print(f"  - {attr}: {type(getattr(first_vision_layer, attr))}")

if __name__ == "__main__":
    check_clip_layers()
