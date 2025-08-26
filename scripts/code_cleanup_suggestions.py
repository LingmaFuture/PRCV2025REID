#!/usr/bin/env python3
"""
代码清理建议脚本
用于识别和清理train.py中的冗余代码
"""

CLEANUP_SUGGESTIONS = [
    {
        "category": "重复函数定义",
        "issue": "move_batch_to_device函数定义了两次",
        "location": "第73-88行 和 第182-194行",
        "action": "删除第182-194行的重复定义",
        "priority": "高"
    },
    {
        "category": "重复导入",
        "issue": "math模块被导入两次", 
        "location": "第4行 和 第13行",
        "action": "删除第13行的重复导入",
        "priority": "中"
    },
    {
        "category": "重复TF32设置",
        "issue": "TF32配置在三个地方重复设置",
        "location": "第24-26行, 第58-60行, 第1386-1388行",
        "action": "只保留第24-26行的全局设置，删除其他重复",
        "priority": "中"
    },
    {
        "category": "未使用的函数",
        "issue": "split_train_dataset函数定义但未使用",
        "location": "第147-161行",
        "action": "删除整个函数定义",
        "priority": "中"
    },
    {
        "category": "重复映射",
        "issue": "MODALITY_MAPPING和MOD_MAP功能重复",
        "location": "第168-180行 和 第763-770行",
        "action": "统一使用一个映射，删除另一个",
        "priority": "中"
    },
    {
        "category": "batch_size计算混乱",
        "issue": "actual_batch_size在多处重复计算",
        "location": "第1450-1502行",
        "action": "统一在一处计算，消除重复逻辑",
        "priority": "高"
    },
    {
        "category": "过时注释",
        "issue": "大量guide*.md引用的注释",
        "location": "整个文件",
        "action": "清理过时的guide引用，保留核心说明",
        "priority": "低"
    },
    {
        "category": "调试代码",
        "issue": "大量调试print和logging",
        "location": "train_epoch_fixed函数内",
        "action": "保留关键监控，删除调试输出",
        "priority": "低"
    }
]

def generate_cleanup_plan():
    """生成清理计划"""
    print("🧹 代码清理建议")
    print("=" * 50)
    
    high_priority = [item for item in CLEANUP_SUGGESTIONS if item["priority"] == "高"]
    medium_priority = [item for item in CLEANUP_SUGGESTIONS if item["priority"] == "中"]
    low_priority = [item for item in CLEANUP_SUGGESTIONS if item["priority"] == "低"]
    
    for priority_level, items in [("高优先级", high_priority), 
                                  ("中优先级", medium_priority), 
                                  ("低优先级", low_priority)]:
        if items:
            print(f"\n📌 {priority_level} 清理项:")
            for i, item in enumerate(items, 1):
                print(f"{i}. {item['category']}")
                print(f"   问题: {item['issue']}")
                print(f"   位置: {item['location']}")
                print(f"   建议: {item['action']}")
                print()

def estimate_cleanup_impact():
    """估算清理影响"""
    total_items = len(CLEANUP_SUGGESTIONS)
    high_items = len([item for item in CLEANUP_SUGGESTIONS if item["priority"] == "高"])
    
    estimated_lines_removed = {
        "重复函数定义": 22,
        "重复导入": 1,
        "重复TF32设置": 6,
        "未使用的函数": 15,
        "重复映射": 12,
        "batch_size计算混乱": 30,
        "过时注释": 50,
        "调试代码": 100
    }
    
    total_lines_to_remove = sum(estimated_lines_removed.values())
    
    print("📊 清理影响评估")
    print("=" * 50)
    print(f"总清理项: {total_items}")
    print(f"高优先级项: {high_items}")
    print(f"预估删除代码行数: {total_lines_to_remove}")
    print(f"预估文件大小减少: {total_lines_to_remove/1884*100:.1f}%")
    print("\n✅ 清理后的预期效果:")
    print("- 消除重复定义，避免潜在bug")
    print("- 减少代码复杂度，提高可维护性") 
    print("- 清理过时逻辑，简化流程")
    print("- 保留核心功能，不影响训练性能")

if __name__ == "__main__":
    generate_cleanup_plan()
    print("\n" + "=" * 50)
    estimate_cleanup_impact()
