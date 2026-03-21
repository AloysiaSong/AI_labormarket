#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试esco-skill-extractor对中文职位描述的支持
"""

from esco_skill_extractor import SkillExtractor
import time

def test_chinese_support():
    """测试esco-skill-extractor对中文的支持程度"""
    
    print("="*60)
    print("测试 esco-skill-extractor 对中文职位描述的支持")
    print("="*60)
    
    # 测试用例：典型的中文职位描述片段
    test_cases = [
        "负责软件开发，精通Python和Java编程，熟悉数据库操作",
        "需要优秀的沟通能力和团队协作精神，能独立完成项目",
        "熟练使用Office办公软件，具有数据分析能力，Excel精通",
        "要求有管理经验，带领团队完成销售任务",
        "设计岗位，精通PS、AI等设计软件，有创意思维"
    ]
    
    print("\n初始化 SkillExtractor（首次运行会下载模型，需要时间）...")
    start_time = time.time()
    
    try:
        extractor = SkillExtractor()
        init_time = time.time() - start_time
        print(f"✓ 初始化完成，耗时: {init_time:.2f}秒\n")
        
        print("-"*60)
        print("开始测试中文识别能力...\n")
        
        total_skills = 0
        for i, desc in enumerate(test_cases, 1):
            print(f"测试 {i}:")
            print(f"职位描述: {desc}")
            
            try:
                start = time.time()
                skills = extractor.get_skills([desc])
                elapsed = time.time() - start
                
                skill_count = len(skills[0]) if skills else 0
                total_skills += skill_count
                
                print(f"识别的技能数: {skill_count}")
                print(f"处理时间: {elapsed:.2f}秒")
                
                if skill_count > 0:
                    print(f"技能URI示例: {skills[0][:3]}")  # 显示前3个
                else:
                    print("⚠️ 未识别到任何技能")
                
                print("-"*60)
                
            except Exception as e:
                print(f"❌ 处理出错: {str(e)}")
                print("-"*60)
        
        # 评估结果
        avg_skills = total_skills / len(test_cases)
        
        print("\n" + "="*60)
        print("测试结果总结")
        print("="*60)
        print(f"总测试案例数: {len(test_cases)}")
        print(f"识别的技能总数: {total_skills}")
        print(f"平均每条识别技能数: {avg_skills:.2f}")
        
        if avg_skills >= 3:
            result = "excellent"
            print("\n✅ 结论: 中文支持优秀，可以直接使用方案A")
            print("   - 识别率高，能准确提取中文职位描述中的技能")
            print("   - 推荐直接使用该工具进行批量处理")
        elif avg_skills >= 1.5:
            result = "good"
            print("\n⚠️ 结论: 中文支持尚可，建议使用方案A并结合关键词验证")
            print("   - 有一定识别能力，但可能遗漏部分技能")
            print("   - 建议在使用后进行抽样验证")
        elif avg_skills >= 0.5:
            result = "poor"
            print("\n⚠️ 结论: 中文支持较弱，建议使用方案B（翻译增强）")
            print("   - 识别率较低，需要翻译关键词后再匹配")
            print("   - 或直接使用方案C（纯中文关键词）")
        else:
            result = "none"
            print("\n❌ 结论: 不支持中文，必须使用方案B或方案C")
            print("   - 完全无法识别中文职位描述")
            print("   - 推荐使用方案C（纯中文关键词匹配）")
        
        print("="*60)
        
        return result, avg_skills
        
    except Exception as e:
        print(f"\n❌ 初始化失败: {str(e)}")
        print("可能原因：")
        print("  1. 网络问题导致模型下载失败")
        print("  2. 依赖包版本冲突")
        print("  3. 系统资源不足")
        print("\n建议：切换到方案C（纯中文关键词方案）")
        return "error", 0

if __name__ == "__main__":
    result, avg = test_chinese_support()
    
    print(f"\n返回值: {result} (平均识别技能数: {avg:.2f})")
