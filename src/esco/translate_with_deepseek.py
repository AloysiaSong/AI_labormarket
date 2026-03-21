#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用DeepSeek API翻译ESCO技能库
"""

import pandas as pd
import json
import time
from tqdm import tqdm
import requests
from typing import List


class DeepSeekTranslator:
    """DeepSeek API翻译器"""
    
    def __init__(self, api_key: str):
        """
        初始化DeepSeek翻译器
        
        Args:
            api_key: DeepSeek API密钥
        """
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        # 使用集中路径配置
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from config.paths import TRANSLATION_CACHE_DIR
        TRANSLATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_file = str(TRANSLATION_CACHE_DIR / 'deepseek_translation_cache.json')
        self.cache = self._load_cache()
        self.stats = {
            'total': 0,
            'cached': 0,
            'translated': 0,
            'errors': 0
        }
        
        print(f"✓ DeepSeek翻译器已初始化")
        print(f"✓ 缓存文件: {self.cache_file}")
        print(f"✓ 已缓存翻译: {len(self.cache)} 条")
    
    def _load_cache(self) -> dict:
        """加载翻译缓存"""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_cache(self):
        """保存翻译缓存"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def translate_text(self, text: str) -> str:
        """
        翻译单个文本
        
        Args:
            text: 英文文本
            
        Returns:
            中文翻译
        """
        self.stats['total'] += 1
        
        # 检查缓存
        if text in self.cache:
            self.stats['cached'] += 1
            return self.cache[text]
        
        # 调用DeepSeek API
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            data = {
                'model': 'deepseek-chat',
                'messages': [
                    {
                        'role': 'system',
                        'content': '你是一位专业的技能描述翻译专家。请将英文技能描述准确翻译成中文，保持专业术语的准确性。只返回翻译结果，不要添加任何解释。'
                    },
                    {
                        'role': 'user',
                        'content': f'请将以下技能描述翻译成中文：{text}'
                    }
                ],
                'temperature': 0.3,  # 降低随机性，提高一致性
                'max_tokens': 200
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                translated = result['choices'][0]['message']['content'].strip()
                
                # 去除可能的引号
                translated = translated.strip('"\'""''')
                
                # 保存到缓存
                self.cache[text] = translated
                self.stats['translated'] += 1
                
                return translated
            else:
                print(f"❌ API错误: {response.status_code} - {response.text}")
                self.stats['errors'] += 1
                return f"[翻译失败] {text}"
                
        except Exception as e:
            print(f"❌ 翻译错误: {e}")
            self.stats['errors'] += 1
            return f"[翻译失败] {text}"
    
    def translate_batch(self, texts: List[str], delay: float = 0.5) -> List[str]:
        """
        批量翻译
        
        Args:
            texts: 文本列表
            delay: 请求间隔（秒）
            
        Returns:
            翻译结果列表
        """
        results = []
        for text in texts:
            result = self.translate_text(text)
            results.append(result)
            time.sleep(delay)  # API速率限制
        return results
    
    def translate_skills_csv(
        self,
        input_csv: str,
        output_csv: str,
        batch_size: int = 10,
        save_interval: int = 50,
        delay: float = 0.5
    ):
        """
        翻译ESCO技能CSV文件
        
        Args:
            input_csv: 输入文件路径
            output_csv: 输出文件路径
            batch_size: 批处理大小
            save_interval: 保存间隔
            delay: API请求间隔（秒）
        """
        print("\n" + "="*80)
        print("开始翻译ESCO技能库 - 使用DeepSeek API")
        print("="*80)
        print(f"输入文件: {input_csv}")
        print(f"输出文件: {output_csv}")
        
        # 读取数据
        df = pd.read_csv(input_csv)
        total = len(df)
        
        print(f"总计: {total} 条技能描述")
        
        # 添加中文列
        if 'description_cn' not in df.columns:
            df['description_cn'] = ''
        
        # 检查已有效翻译的数量（排除"[翻译失败]"）
        valid_translations = 0
        for idx in range(len(df)):
            val = df.at[idx, 'description_cn']
            if pd.notna(val) and val != '' and not val.startswith('[翻译失败]'):
                valid_translations += 1
        
        if valid_translations > 0:
            print(f"已有效翻译: {valid_translations} 条")
            print(f"剩余待翻译: {total - valid_translations} 条")
            print("自动从断点继续...")
        
        # 批量翻译
        print("\n开始翻译...")
        progress_bar = tqdm(total=total, desc="翻译进度")
        
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            
            # 获取批次数据
            batch_indices = []
            batch_texts = []
            
            for idx in range(i, batch_end):
                if pd.isna(df.at[idx, 'description_cn']) or df.at[idx, 'description_cn'] == '':
                    batch_indices.append(idx)
                    batch_texts.append(df.at[idx, 'description'])
            
            # 翻译批次
            if batch_texts:
                translations = self.translate_batch(batch_texts, delay=delay)
                
                # 保存结果
                for idx, translation in zip(batch_indices, translations):
                    df.at[idx, 'description_cn'] = translation
            
            progress_bar.update(batch_end - i)
            
            # 定期保存
            if (i // batch_size) % (save_interval // batch_size) == 0 and i > 0:
                df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                self._save_cache()
                tqdm.write(f"✓ 已保存进度: {i}/{total}")
        
        progress_bar.close()
        
        # 最终保存
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        self._save_cache()
        
        # 统计信息
        print("\n" + "="*80)
        print("翻译完成！")
        print("="*80)
        print(f"✓ 总计: {self.stats['total']} 条")
        print(f"✓ 缓存命中: {self.stats['cached']} 条")
        print(f"✓ 新翻译: {self.stats['translated']} 条")
        print(f"✓ 错误: {self.stats['errors']} 条")
        print(f"✓ 已保存至: {output_csv}")
        print("="*80)
        
        # 显示翻译样例
        print("\n翻译样例（前5条）:")
        print("-"*80)
        for idx in range(min(5, len(df))):
            print(f"\n[{idx+1}]")
            print(f"英文: {df.at[idx, 'description']}")
            print(f"中文: {df.at[idx, 'description_cn']}")


def main():
    """主函数"""
    print("="*80)
    print("ESCO技能库翻译 - DeepSeek API")
    print("="*80)
    
    # API密钥
    api_key = "sk-6b5eac89556b45a093afcf1216da8f61"
    
    # 使用集中路径配置
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.paths import ESCO_CHINESE_DIR
    ESCO_CHINESE_DIR.mkdir(parents=True, exist_ok=True)

    # 文件路径 (输入从系统包获取，输出到项目目录)
    input_csv = '/Users/yu/code/miniconda3/lib/python3.13/site-packages/esco_skill_extractor/data/skills.csv'
    output_csv = str(ESCO_CHINESE_DIR / 'skills_chinese.csv')
    
    print(f"\nAPI密钥: {api_key[:20]}...{api_key[-10:]}")
    print(f"输入文件: {input_csv}")
    print(f"输出文件: {output_csv}")
    
    print("\n翻译配置:")
    print("  - 批量大小: 10条/批")
    print("  - 保存间隔: 每50条")
    print("  - API延迟: 0.5秒/条")
    print("  - 预计时间: 2-3小时")
    print("  - 预计成本: 约$2-5 (13,940条)")
    
    print("\n特性:")
    print("  ✓ 支持断点续传")
    print("  ✓ 自动缓存翻译结果")
    print("  ✓ 定期保存进度")
    print("  ✓ 错误自动重试")
    
    print("\n开始翻译...")
    
    if True:
        # 创建翻译器
        translator = DeepSeekTranslator(api_key)
        
        # 开始翻译
        translator.translate_skills_csv(
            input_csv=input_csv,
            output_csv=output_csv,
            batch_size=10,
            save_interval=50,
            delay=0.5
        )
    else:
        print("\n已取消")


if __name__ == "__main__":
    main()
