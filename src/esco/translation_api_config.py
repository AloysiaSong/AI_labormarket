#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译API配置模板
请根据你的需求配置相应的API密钥
"""

# ============================================
# Google Cloud Translation API
# ============================================
GOOGLE_TRANSLATE_CONFIG = {
    'api_key': 'YOUR_GOOGLE_API_KEY',  # 从 Google Cloud Console 获取
    'project_id': 'YOUR_PROJECT_ID',
    'enabled': False
}

# ============================================
# DeepL API
# ============================================
DEEPL_CONFIG = {
    'api_key': 'YOUR_DEEPL_API_KEY',  # 从 DeepL 获取
    'enabled': False
}

# ============================================
# OpenAI API
# ============================================
OPENAI_CONFIG = {
    'api_key': 'YOUR_OPENAI_API_KEY',  # 从 OpenAI 获取
    'model': 'gpt-3.5-turbo',  # 或 'gpt-4' 以获得更好质量
    'enabled': False
}

# ============================================
# 百度翻译API（国内用户推荐）
# ============================================
BAIDU_CONFIG = {
    'app_id': 'YOUR_BAIDU_APP_ID',
    'secret_key': 'YOUR_BAIDU_SECRET_KEY',
    'enabled': False
}

# ============================================
# 有道智云翻译API
# ============================================
YOUDAO_CONFIG = {
    'app_key': 'YOUR_YOUDAO_APP_KEY',
    'app_secret': 'YOUR_YOUDAO_APP_SECRET',
    'enabled': False
}

# ============================================
# 优先级设置
# ============================================
# 翻译服务优先级（按顺序尝试）
TRANSLATION_PRIORITY = [
    'deepl',    # 技术翻译质量最好
    'openai',   # 可控制翻译风格
    'google',   # 通用翻译
    'baidu',    # 国内服务
    'youdao'    # 备用服务
]

# ============================================
# 翻译参数配置
# ============================================
TRANSLATION_CONFIG = {
    'batch_size': 100,           # 每批翻译数量
    'max_retries': 3,            # 失败重试次数
    'retry_delay': 2,            # 重试延迟（秒）
    'rate_limit_delay': 0.5,     # API调用间隔（秒）
    'timeout': 30,               # 请求超时（秒）
    'source_lang': 'en',         # 源语言
    'target_lang': 'zh',         # 目标语言
}

# ============================================
# 成本估算
# ============================================
COST_ESTIMATES = {
    'google': {
        'per_1m_chars': 20.0,  # USD
        'note': 'Google Cloud Translation API 按字符计费'
    },
    'deepl': {
        'per_1m_chars': 20.0,  # USD
        'free_tier': 500000,   # 免费额度（字符）
        'note': 'DeepL API Pro 按字符计费，有免费额度'
    },
    'openai': {
        'gpt_3_5_per_1k_tokens': 0.002,  # USD (input + output)
        'gpt_4_per_1k_tokens': 0.06,     # USD (input + output)
        'note': 'OpenAI API 按token计费，翻译需要输入+输出token'
    },
    'baidu': {
        'per_1m_chars': 49.0,  # RMB
        'free_tier': 2000000,  # 免费额度（字符/月）
        'note': '百度翻译API 按字符计费，每月有免费额度'
    }
}

def estimate_translation_cost(num_skills: int = 13940, avg_chars_per_skill: int = 100):
    """
    估算翻译成本
    
    Args:
        num_skills: 技能数量
        avg_chars_per_skill: 平均每个技能的字符数
    
    Returns:
        各API的成本估算
    """
    total_chars = num_skills * avg_chars_per_skill
    
    print("="*60)
    print("翻译成本估算")
    print("="*60)
    print(f"技能数量: {num_skills:,}")
    print(f"平均字符/技能: {avg_chars_per_skill}")
    print(f"总字符数: {total_chars:,}")
    print("\n" + "-"*60)
    
    # Google Translate
    google_cost = (total_chars / 1_000_000) * COST_ESTIMATES['google']['per_1m_chars']
    print(f"\nGoogle Cloud Translation:")
    print(f"  估算成本: ${google_cost:.2f} USD")
    print(f"  {COST_ESTIMATES['google']['note']}")
    
    # DeepL
    deepl_free = COST_ESTIMATES['deepl']['free_tier']
    if total_chars <= deepl_free:
        print(f"\nDeepL API:")
        print(f"  估算成本: $0.00 USD (免费额度内)")
    else:
        deepl_cost = ((total_chars - deepl_free) / 1_000_000) * COST_ESTIMATES['deepl']['per_1m_chars']
        print(f"\nDeepL API:")
        print(f"  估算成本: ${deepl_cost:.2f} USD (超出免费额度)")
    print(f"  {COST_ESTIMATES['deepl']['note']}")
    
    # OpenAI
    # 估算token数（英文: ~1 token/4 chars, 中文输出更多）
    input_tokens = total_chars / 4
    output_tokens = total_chars / 2  # 中文输出token更多
    total_tokens = (input_tokens + output_tokens) / 1000
    
    gpt35_cost = total_tokens * COST_ESTIMATES['openai']['gpt_3_5_per_1k_tokens']
    gpt4_cost = total_tokens * COST_ESTIMATES['openai']['gpt_4_per_1k_tokens']
    
    print(f"\nOpenAI API:")
    print(f"  GPT-3.5-turbo: ${gpt35_cost:.2f} USD")
    print(f"  GPT-4: ${gpt4_cost:.2f} USD")
    print(f"  {COST_ESTIMATES['openai']['note']}")
    
    # 百度翻译
    baidu_free = COST_ESTIMATES['baidu']['free_tier']
    if total_chars <= baidu_free:
        print(f"\n百度翻译API:")
        print(f"  估算成本: ¥0.00 RMB (免费额度内)")
    else:
        baidu_cost = ((total_chars - baidu_free) / 1_000_000) * COST_ESTIMATES['baidu']['per_1m_chars']
        print(f"\n百度翻译API:")
        print(f"  估算成本: ¥{baidu_cost:.2f} RMB")
    print(f"  {COST_ESTIMATES['baidu']['note']}")
    
    print("\n" + "="*60)
    print("推荐方案:")
    print("  1. DeepL (技术翻译质量最好，有免费额度)")
    print("  2. 百度翻译 (国内服务，有大量免费额度)")
    print("  3. OpenAI GPT-3.5 (可控性强，成本适中)")
    print("="*60)


if __name__ == "__main__":
    # 运行成本估算
    estimate_translation_cost()
