#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实用翻译工具 - 支持多种API的批量翻译
"""

import pandas as pd
import time
import json
from typing import List, Optional
from tqdm import tqdm
import hashlib


class UniversalTranslator:
    """通用翻译器 - 支持多种翻译API"""
    
    def __init__(self, api_config: dict = None):
        """
        初始化通用翻译器
        
        Args:
            api_config: API配置字典
        """
        self.api_config = api_config or {}
        self.cache = {}
        self.translation_stats = {
            'total': 0,
            'cached': 0,
            'api_calls': 0,
            'errors': 0
        }
    
    def translate_batch_baidu(self, texts: List[str]) -> List[str]:
        """使用百度翻译API（免费额度大）"""
        try:
            import hashlib
            import random
            import requests
            
            app_id = self.api_config.get('baidu', {}).get('app_id')
            secret_key = self.api_config.get('baidu', {}).get('secret_key')
            
            if not app_id or not secret_key:
                return [f"[需要百度API配置] {t}" for t in texts]
            
            results = []
            for text in texts:
                # 百度翻译API签名
                salt = str(random.randint(32768, 65536))
                sign_str = app_id + text + salt + secret_key
                sign = hashlib.md5(sign_str.encode()).hexdigest()
                
                url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
                params = {
                    'q': text,
                    'from': 'en',
                    'to': 'zh',
                    'appid': app_id,
                    'salt': salt,
                    'sign': sign
                }
                
                response = requests.get(url, params=params, timeout=10)
                result = response.json()
                
                if 'trans_result' in result:
                    translated = result['trans_result'][0]['dst']
                    results.append(translated)
                else:
                    results.append(f"[翻译失败] {text}")
                
                time.sleep(0.5)  # API速率限制
            
            return results
            
        except Exception as e:
            print(f"百度翻译错误: {e}")
            return [f"[错误] {t}" for t in texts]
    
    def translate_batch_deepl(self, texts: List[str]) -> List[str]:
        """使用DeepL API（质量高）"""
        try:
            import deepl
            
            api_key = self.api_config.get('deepl', {}).get('api_key')
            if not api_key:
                return [f"[需要DeepL API配置] {t}" for t in texts]
            
            translator = deepl.Translator(api_key)
            results = translator.translate_text(
                texts,
                source_lang='EN',
                target_lang='ZH'
            )
            
            return [r.text for r in results]
            
        except Exception as e:
            print(f"DeepL翻译错误: {e}")
            return [f"[错误] {t}" for t in texts]
    
    def translate_batch_simple(self, texts: List[str]) -> List[str]:
        """简单翻译（使用googletrans库，免费但不稳定）"""
        try:
            from googletrans import Translator
            translator = Translator()
            
            results = []
            for text in texts:
                try:
                    result = translator.translate(text, src='en', dest='zh-cn')
                    results.append(result.text)
                    time.sleep(0.3)  # 避免被限制
                except:
                    results.append(f"[翻译失败] {text}")
            
            return results
            
        except Exception as e:
            print(f"Google翻译错误: {e}")
            return [f"[错误] {t}" for t in texts]
    
    def translate_skills_file(
        self,
        input_csv: str,
        output_csv: str,
        method: str = 'baidu',
        batch_size: int = 10,
        save_interval: int = 100
    ):
        """
        翻译技能CSV文件
        
        Args:
            input_csv: 输入文件路径
            output_csv: 输出文件路径
            method: 翻译方法 ['baidu', 'deepl', 'simple']
            batch_size: 批处理大小
            save_interval: 保存间隔
        """
        print(f"\n开始翻译技能文件...")
        print(f"方法: {method}")
        print(f"输入: {input_csv}")
        print(f"输出: {output_csv}")
        
        # 读取数据
        df = pd.read_csv(input_csv)
        total = len(df)
        
        print(f"总计: {total} 条技能描述")
        
        # 添加中文列
        if 'description_cn' not in df.columns:
            df['description_cn'] = ''
        
        # 选择翻译方法
        if method == 'baidu':
            translate_func = self.translate_batch_baidu
        elif method == 'deepl':
            translate_func = self.translate_batch_deepl
        else:
            translate_func = self.translate_batch_simple
        
        # 批量翻译
        for i in tqdm(range(0, total, batch_size), desc="翻译进度"):
            batch_end = min(i + batch_size, total)
            batch_texts = df.iloc[i:batch_end]['description'].tolist()
            
            # 翻译
            translations = translate_func(batch_texts)
            
            # 保存结果
            for j, translation in enumerate(translations):
                df.at[i + j, 'description_cn'] = translation
            
            # 定期保存
            if (i // batch_size) % (save_interval // batch_size) == 0:
                df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        # 最终保存
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ 翻译完成！已保存至: {output_csv}")
        
        # 统计
        completed = df['description_cn'].notna().sum()
        print(f"✓ 成功翻译: {completed}/{total}")


def quick_translate_with_baidu():
    """快速翻译 - 使用百度API（推荐）"""
    print("\n" + "="*80)
    print("快速翻译工具 - 百度翻译API")
    print("="*80)
    
    print("\n请按以下步骤操作：")
    print("\n1. 获取百度翻译API密钥:")
    print("   访问: https://fanyi-api.baidu.com/")
    print("   注册并创建应用，获取 APP ID 和 密钥")
    print("   每月有200万字符免费额度")
    
    print("\n2. 配置API密钥:")
    app_id = input("   请输入百度 APP ID (或按Enter跳过): ").strip()
    secret_key = input("   请输入百度 密钥 (或按Enter跳过): ").strip()
    
    if not app_id or not secret_key:
        print("\n⚠️ 未配置API密钥，将使用示例模式")
        return
    
    # 配置API
    api_config = {
        'baidu': {
            'app_id': app_id,
            'secret_key': secret_key
        }
    }
    
    # 创建翻译器
    translator = UniversalTranslator(api_config)
    
    # 使用集中路径配置
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.paths import ESCO_CHINESE_DIR
    ESCO_CHINESE_DIR.mkdir(parents=True, exist_ok=True)

    # 文件路径
    input_csv = '/Users/yu/code/miniconda3/lib/python3.13/site-packages/esco_skill_extractor/data/skills.csv'
    output_csv = str(ESCO_CHINESE_DIR / 'skills_chinese.csv')
    
    print("\n3. 开始翻译...")
    print(f"   这将翻译约13,940条技能描述")
    print(f"   预计时间: 2-4小时")
    print(f"   费用: 免费（在200万字符额度内）")
    
    confirm = input("\n确认开始翻译? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        translator.translate_skills_file(
            input_csv=input_csv,
            output_csv=output_csv,
            method='baidu',
            batch_size=10,
            save_interval=100
        )
    else:
        print("\n已取消")


if __name__ == "__main__":
    quick_translate_with_baidu()
