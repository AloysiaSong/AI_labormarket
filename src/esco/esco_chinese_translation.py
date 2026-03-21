#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESCO技能库完整汉化系统
将13,940个ESCO技能描述翻译成中文
"""

import pandas as pd
import os
import time
from typing import List, Dict
import json

# 翻译API配置（需要配置你的翻译服务）
# 可选方案：
# 1. Google Translate API
# 2. DeepL API
# 3. OpenAI GPT API
# 4. 本地翻译模型

class ESCOTranslator:
    """ESCO技能库翻译器"""
    
    def __init__(self, translation_method='local'):
        """
        初始化翻译器
        
        Args:
            translation_method: 翻译方法 ['local', 'google', 'deepl', 'openai']
        """
        self.translation_method = translation_method
        self.cache_file = '/Users/yu/code/code2601/TY/Test_ESCO/esco_translations_cache.json'
        self.cache = self._load_cache()
        
        print(f"初始化翻译器，方法: {translation_method}")
        print(f"缓存文件: {self.cache_file}")
        print(f"已缓存翻译: {len(self.cache)} 条")
    
    def _load_cache(self) -> Dict[str, str]:
        """加载翻译缓存"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
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
        # 检查缓存
        if text in self.cache:
            return self.cache[text]
        
        # 执行翻译
        if self.translation_method == 'local':
            # 使用本地规则翻译（简化版）
            translated = self._translate_local(text)
        elif self.translation_method == 'google':
            translated = self._translate_google(text)
        elif self.translation_method == 'openai':
            translated = self._translate_openai(text)
        else:
            translated = text  # 不翻译
        
        # 保存到缓存
        self.cache[text] = translated
        return translated
    
    def _translate_local(self, text: str) -> str:
        """
        本地规则翻译（示例实现）
        实际应该调用专业翻译API
        """
        # 这里返回原文 + 中文标记
        # 实际项目中应该使用真实的翻译服务
        return f"[待翻译] {text}"
    
    def _translate_google(self, text: str) -> str:
        """使用Google Translate API"""
        try:
            from googletrans import Translator
            translator = Translator()
            result = translator.translate(text, src='en', dest='zh-cn')
            return result.text
        except Exception as e:
            print(f"Google翻译失败: {e}")
            return text
    
    def _translate_openai(self, text: str) -> str:
        """使用OpenAI API翻译"""
        try:
            import openai
            # 需要设置 openai.api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是专业的技能描述翻译专家，将英文技能描述翻译成准确的中文。"},
                    {"role": "user", "content": f"请将以下技能描述翻译成中文：{text}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI翻译失败: {e}")
            return text
    
    def translate_skills_csv(self, input_csv: str, output_csv: str, batch_size: int = 100):
        """
        翻译ESCO技能CSV文件
        
        Args:
            input_csv: 原始英文CSV路径
            output_csv: 输出中文CSV路径
            batch_size: 批量翻译大小
        """
        print(f"\n开始翻译ESCO技能库...")
        print(f"输入文件: {input_csv}")
        print(f"输出文件: {output_csv}")
        
        # 读取原始数据
        df = pd.read_csv(input_csv)
        total = len(df)
        
        print(f"总共需要翻译 {total} 条技能描述")
        
        # 添加中文列
        df['description_cn'] = ''
        
        # 批量翻译
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            print(f"\n处理 {i+1}-{batch_end}/{total}...")
            
            for idx in range(i, batch_end):
                english_text = df.at[idx, 'description']
                chinese_text = self.translate_text(english_text)
                df.at[idx, 'description_cn'] = chinese_text
                
                if (idx + 1) % 10 == 0:
                    print(f"  已翻译: {idx + 1}/{total}")
            
            # 定期保存
            if (i // batch_size) % 5 == 0:
                df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                self._save_cache()
                print(f"  ✓ 已保存进度")
        
        # 最终保存
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        self._save_cache()
        
        print(f"\n✓ 翻译完成！")
        print(f"✓ 共翻译 {total} 条")
        print(f"✓ 已保存至: {output_csv}")


def create_chinese_esco_extractor():
    """创建中文版ESCO提取器"""
    
    print("\n" + "="*80)
    print("创建中文版ESCO技能提取器")
    print("="*80)
    
    code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文版ESCO技能提取器
使用中文Sentence Transformer和中文技能描述
"""

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import torch
import pickle
import os
import warnings
from typing import List, Union

class ChineseESCOSkillExtractor:
    """中文ESCO技能提取器"""
    
    def __init__(
        self, 
        skills_csv: str = None,
        model_name: str = "shibing624/text2vec-base-chinese",
        threshold: float = 0.5,
        device: str = None
    ):
        """
        初始化中文技能提取器
        
        Args:
            skills_csv: 中文技能CSV文件路径
            model_name: 中文Sentence Transformer模型名称
            threshold: 相似度阈值
            device: 计算设备 (cuda/cpu)
        """
        print("初始化中文ESCO技能提取器...")
        
        self.threshold = threshold
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # 加载中文Sentence Transformer模型
        print(f"加载中文模型: {model_name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = SentenceTransformer(model_name, device=self.device)
        
        # 加载中文技能数据
        print(f"加载中文技能数据...")
        if skills_csv and os.path.exists(skills_csv):
            self.skills_df = pd.read_csv(skills_csv)
            print(f"✓ 加载了 {len(self.skills_df)} 条技能")
        else:
            raise FileNotFoundError(f"技能文件不存在: {skills_csv}")
        
        # 生成或加载嵌入
        self.embeddings_file = skills_csv.replace('.csv', '_embeddings.pkl')
        self._load_or_create_embeddings()
    
    def _load_or_create_embeddings(self):
        """加载或创建技能嵌入"""
        if os.path.exists(self.embeddings_file):
            print(f"加载已有嵌入: {self.embeddings_file}")
            with open(self.embeddings_file, 'rb') as f:
                self.skill_embeddings = pickle.load(f).to(self.device)
        else:
            print(f"生成中文技能嵌入（首次运行，需要较长时间）...")
            
            # 使用中文描述生成嵌入
            descriptions = self.skills_df['description_cn'].tolist()
            
            self.skill_embeddings = self.model.encode(
                descriptions,
                device=self.device,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            
            # 保存嵌入
            print(f"保存嵌入到: {self.embeddings_file}")
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.skill_embeddings.cpu(), f)
    
    def extract_skills(self, texts: List[str]) -> List[List[str]]:
        """
        从中文文本中提取技能
        
        Args:
            texts: 中文职位描述列表
            
        Returns:
            每个文本匹配的技能ID列表
        """
        if not texts or all(not t for t in texts):
            return [[] for _ in texts]
        
        # 将文本编码为嵌入
        text_embeddings = self.model.encode(
            texts,
            device=self.device,
            normalize_embeddings=True,
            convert_to_tensor=True
        )
        
        # 计算相似度
        similarities = util.dot_score(text_embeddings, self.skill_embeddings)
        
        # 提取超过阈值的技能
        results = []
        for i, text_sim in enumerate(similarities):
            matched_indices = torch.where(text_sim > self.threshold)[0]
            matched_ids = self.skills_df.iloc[matched_indices.cpu().numpy()]['id'].tolist()
            results.append(matched_ids)
        
        return results
    
    def get_skill_names(self, skill_ids: List[str]) -> List[str]:
        """获取技能的中文名称"""
        names = []
        for skill_id in skill_ids:
            skill = self.skills_df[self.skills_df['id'] == skill_id]
            if not skill.empty:
                names.append(skill.iloc[0]['description_cn'])
        return names


if __name__ == "__main__":
    # 使用示例
    extractor = ChineseESCOSkillExtractor(
        skills_csv="/Users/yu/code/code2601/TY/Test_ESCO/skills_chinese.csv"
    )
    
    # 测试
    test_texts = [
        "负责软件开发，精通Python和Java编程",
        "需要良好的沟通能力和团队协作精神"
    ]
    
    results = extractor.extract_skills(test_texts)
    
    for text, skills in zip(test_texts, results):
        print(f"\\n文本: {text}")
        print(f"识别技能数: {len(skills)}")
        if skills:
            names = extractor.get_skill_names(skills[:3])
            print(f"技能示例: {names}")
'''
    
    output_file = '/Users/yu/code/code2601/TY/Test_ESCO/chinese_esco_extractor.py'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    print(f"✓ 已创建: {output_file}")


if __name__ == "__main__":
    print("="*80)
    print("ESCO技能库完整汉化系统")
    print("="*80)
    
    # 配置
    original_skills_csv = '/Users/yu/code/miniconda3/lib/python3.13/site-packages/esco_skill_extractor/data/skills.csv'
    translated_skills_csv = '/Users/yu/code/code2601/TY/Test_ESCO/skills_chinese.csv'
    
    print("\n⚠️ 重要提示：")
    print("1. 完整翻译13,940条技能描述需要使用专业翻译API")
    print("2. 推荐方案：")
    print("   - Google Cloud Translation API (付费，质量高)")
    print("   - DeepL API (付费，技术翻译质量好)")
    print("   - OpenAI GPT API (付费，可控制翻译风格)")
    print("3. 预计成本：$50-200 (取决于API选择)")
    print("4. 预计时间：2-6小时 (取决于API速率限制)")
    
    print("\n当前配置: 使用示例翻译器（不会真实翻译）")
    print("如需真实翻译，请：")
    print("  1. 配置翻译API密钥")
    print("  2. 修改 translation_method 参数")
    print("  3. 运行完整翻译流程")
    
    choice = input("\n是否继续创建框架文件？(y/n): ")
    
    if choice.lower() == 'y':
        # 创建翻译器实例
        translator = ESCOTranslator(translation_method='local')
        
        # 创建中文提取器代码
        create_chinese_esco_extractor()
        
        print("\n" + "="*80)
        print("框架创建完成！")
        print("="*80)
        print("\n下一步:")
        print("1. 配置翻译API (Google/DeepL/OpenAI)")
        print("2. 运行翻译:")
        print("   translator.translate_skills_csv(original_skills_csv, translated_skills_csv)")
        print("3. 测试中文提取器:")
        print("   python chinese_esco_extractor.py")
        print("4. 集成到数据处理流程")
    else:
        print("\n已取消。")
