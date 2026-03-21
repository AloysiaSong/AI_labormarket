#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用百度翻译API翻译ESCO技能库
完全免费（200万字符/月额度）
"""

import pandas as pd
import hashlib
import random
import requests
import time
import json
from tqdm import tqdm


class BaiduTranslator:
    """百度翻译API"""
    
    def __init__(self, app_id: str, secret_key: str):
        self.app_id = app_id
        self.secret_key = secret_key
        self.api_url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
        # 使用集中路径配置
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
        from config.paths import TRANSLATION_CACHE_DIR
        TRANSLATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_file = str(TRANSLATION_CACHE_DIR / 'baidu_translation_cache.json')
        self.cache = self._load_cache()
        self.stats = {'total': 0, 'cached': 0, 'translated': 0, 'errors': 0}
        
        print(f"✓ 百度翻译器已初始化")
        print(f"✓ APP ID: {app_id[:10]}...")
        print(f"✓ 已缓存: {len(self.cache)} 条")
    
    def _load_cache(self):
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def translate(self, text: str) -> str:
        """翻译单个文本"""
        self.stats['total'] += 1
        
        # 检查缓存
        if text in self.cache:
            self.stats['cached'] += 1
            return self.cache[text]
        
        # 调用百度API
        try:
            salt = str(random.randint(32768, 65536))
            sign_str = self.app_id + text + salt + self.secret_key
            sign = hashlib.md5(sign_str.encode()).hexdigest()
            
            params = {
                'q': text,
                'from': 'en',
                'to': 'zh',
                'appid': self.app_id,
                'salt': salt,
                'sign': sign
            }
            
            response = requests.get(self.api_url, params=params, timeout=10)
            result = response.json()
            
            if 'trans_result' in result:
                translated = result['trans_result'][0]['dst']
                self.cache[text] = translated
                self.stats['translated'] += 1
                return translated
            else:
                error_msg = result.get('error_msg', 'Unknown error')
                print(f"❌ API错误: {error_msg}")
                self.stats['errors'] += 1
                return f"[翻译失败] {text}"
        
        except Exception as e:
            print(f"❌ 错误: {e}")
            self.stats['errors'] += 1
            return f"[翻译失败] {text}"
    
    def translate_skills_csv(self, input_csv: str, output_csv: str):
        """翻译技能CSV"""
        print("\n" + "="*80)
        print("百度翻译 - ESCO技能库")
        print("="*80)
        
        # 加载数据
        df = pd.read_csv(input_csv)
        if 'description_cn' not in df.columns:
            df['description_cn'] = ''
        
        total = len(df)
        already_done = (df['description_cn'] != '').sum()
        
        print(f"总计: {total}")
        print(f"已完成: {already_done}")
        print(f"待翻译: {total - already_done}")
        
        if already_done > 0:
            resume = input("\n继续翻译? (y/n): ").lower()
            if resume != 'y':
                return
        
        # 翻译
        print("\n开始翻译...")
        for idx in tqdm(range(len(df)), desc="进度"):
            if df.at[idx, 'description_cn'] == '' or pd.isna(df.at[idx, 'description_cn']):
                en_text = df.at[idx, 'description']
                cn_text = self.translate(en_text)
                df.at[idx, 'description_cn'] = cn_text
                
                # 定期保存
                if (idx + 1) % 50 == 0:
                    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                    self._save_cache()
                    tqdm.write(f"✓ 已保存: {idx+1}/{total}")
                
                time.sleep(1)  # API限制：QPS=1
        
        # 最终保存
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        self._save_cache()
        
        print("\n" + "="*80)
        print("翻译完成！")
        print("="*80)
        print(f"✓ 总计: {self.stats['total']}")
        print(f"✓ 缓存: {self.stats['cached']}")
        print(f"✓ 新翻译: {self.stats['translated']}")
        print(f"✓ 错误: {self.stats['errors']}")
        print(f"✓ 文件: {output_csv}")


def main():
    print("="*80)
    print("百度翻译API - 完全免费方案")
    print("="*80)
    print("\n获取API密钥:")
    print("1. 访问: https://fanyi-api.baidu.com/")
    print("2. 注册/登录百度账号")
    print("3. 进入'管理控制台' → '开发者信息'")
    print("4. 获取 APP ID 和 密钥")
    print("5. 免费额度: 200万字符/月（足够翻译20次ESCO库）\n")
    
    app_id = input("请输入百度 APP ID: ").strip()
    secret_key = input("请输入百度 密钥: ").strip()
    
    if not app_id or not secret_key:
        print("\n❌ 需要提供API密钥")
        return
    
    # 测试API
    print("\n测试API...")
    translator = BaiduTranslator(app_id, secret_key)
    test_result = translator.translate("test")
    
    if "翻译失败" in test_result:
        print("❌ API测试失败，请检查密钥是否正确")
        return
    
    print(f"✓ API测试成功: 'test' → '{test_result}'")
    
    # 开始翻译
    # 使用集中路径配置
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.paths import ESCO_CHINESE_DIR
    ESCO_CHINESE_DIR.mkdir(parents=True, exist_ok=True)

    input_csv = '/Users/yu/code/miniconda3/lib/python3.13/site-packages/esco_skill_extractor/data/skills.csv'
    output_csv = str(ESCO_CHINESE_DIR / 'skills_chinese.csv')
    
    print(f"\n配置:")
    print(f"  输入: {input_csv}")
    print(f"  输出: {output_csv}")
    print(f"  预计时间: 4-5小时 (每条1秒)")
    print(f"  费用: ¥0 (免费)")
    
    confirm = input("\n确认开始? (yes/no): ").lower()
    if confirm == 'yes':
        translator.translate_skills_csv(input_csv, output_csv)
    else:
        print("已取消")


if __name__ == "__main__":
    main()
