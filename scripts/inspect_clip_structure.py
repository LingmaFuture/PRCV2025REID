# tools/inspect_clip_structure.py
"""检查CLIP模型的实际结构"""
import torch
from transformers import CLIPModel

def inspect_clip_structure():
    print("🔍 检查CLIP模型结构...")
    
    # 加载CLIP模型
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    
    print("\n📊 Vision Model 结构:")
    vision_model = clip_model.vision_model
    for name, module in vision_model.named_children():
        print(f"  - {name}: {type(module).__name__}")
        if hasattr(module, 'named_children'):
            for sub_name, sub_module in module.named_children():
                print(f"    - {sub_name}: {type(sub_module).__name__}")
    
    print("\n📊 Vision Model 主要属性:")
    for attr in ['embeddings', 'encoder', 'layernorm', 'post_layernorm']:
        if hasattr(vision_model, attr):
            print(f"  ✅ {attr}: {type(getattr(vision_model, attr)).__name__}")
        else:
            print(f"  ❌ {attr}: 不存在")
    
    print("\n📊 Embeddings 结构:")
    embeddings = vision_model.embeddings
    for name, module in embeddings.named_children():
        print(f"  - {name}: {type(module).__name__}")
    
    print("\n📊 Text Model 结构:")
    text_model = clip_model.text_model
    for name, module in text_model.named_children():
        print(f"  - {name}: {type(module).__name__}")
    
    print("\n📊 Text Model 主要属性:")
    for attr in ['embeddings', 'encoder', 'final_layer_norm']:
        if hasattr(text_model, attr):
            print(f"  ✅ {attr}: {type(getattr(text_model, attr)).__name__}")
        else:
            print(f"  ❌ {attr}: 不存在")
    
    print("\n📊 Encoder 层结构:")
    if hasattr(vision_model, 'encoder'):
        encoder = vision_model.encoder
        for name, module in encoder.named_children():
            print(f"  - {name}: {type(module).__name__}")
            if name == 'layers' and hasattr(module, '__len__'):
                print(f"    - 层数: {len(module)}")
                if len(module) > 0:
                    first_layer = module[0]
                    for layer_attr, layer_module in first_layer.named_children():
                        print(f"      - {layer_attr}: {type(layer_module).__name__}")

if __name__ == "__main__":
    inspect_clip_structure()
