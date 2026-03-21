#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA技能组合识别
基于"Occupations as bundles of skills"理论框架
"""

import pandas as pd
import numpy as np
import jieba
import jieba.posseg as pseg
from gensim import corpora, models
from gensim.models import CoherenceModel
import logging
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


class SkillBundleLDA:
    """LDA技能组合识别器"""
    
    def __init__(
        self,
        input_csv: str,
        output_dir: str,
        num_topics: int = 80,
        sample_size: int = None,
        random_state: int = 42
    ):
        """
        初始化
        
        Args:
            input_csv: 输入CSV文件
            output_dir: 输出目录
            num_topics: 主题数量（技能组合数）
            sample_size: 抽样数量（None=全量）
            random_state: 随机种子
        """
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.num_topics = num_topics
        self.sample_size = sample_size
        self.random_state = random_state
        
        # 停用词
        self.stopwords = self._load_stopwords()
        
        # 模型组件
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        
        print("="*80)
        print("LDA职业技能组合识别")
        print("理论框架: Occupations as bundles of skills")
        print("="*80)
        print(f"输入: {input_csv}")
        print(f"输出: {output_dir}")
        print(f"技能组合数: {num_topics}")
        print(f"抽样: {sample_size if sample_size else '全量'}")
    
    def _load_stopwords(self):
        """加载停用词"""
        # 通用停用词
        stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '年', '月',
            '日', '能', '及', '等', '、', '。', '，', '；', '：', '？',
            '！', '…', '"', '"', ''', ''', '（', '）', '【', '】'
        ])
        
        # 职位描述特定停用词
        job_stopwords = set([
            '工作', '职位', '招聘', '岗位', '公司', '企业', '要求',
            '负责', '进行', '相关', '具有', '以上', '以下', '优先',
            '熟悉', '了解', '掌握', '具备', '较强', '良好', '优秀',
            '能够', '可以', '需要', '包括', '提供', '完成', '协助',
            '欢迎', '加入', '团队', '待遇', '面议', '联系', '有限'
        ])
        
        return stopwords | job_stopwords
    
    def load_data(self):
        """加载数据"""
        print("\n" + "="*80)
        print("阶段1: 数据加载")
        print("="*80)
        
        docs = []
        metadata = []
        chunk_size = 100000
        
        print(f"分批读取（每批 {chunk_size:,} 行）...")
        
        for i, chunk in enumerate(pd.read_csv(
            self.input_csv,
            usecols=['职位描述', '工作城市', '招聘类别'],
            chunksize=chunk_size,
            encoding='utf-8-sig'
        )):
            # 清理
            chunk = chunk.dropna(subset=['职位描述'])
            chunk = chunk[chunk['职位描述'].str.strip() != '']
            
            # 保存
            docs.extend(chunk['职位描述'].tolist())
            metadata.extend(chunk[['工作城市', '招聘类别']].to_dict('records'))
            
            if (i + 1) % 10 == 0:
                print(f"  已读取 {(i+1)*chunk_size:,} 行")
            
            # 抽样限制
            if self.sample_size and len(docs) >= self.sample_size:
                break
        
        # 抽样
        if self.sample_size and len(docs) > self.sample_size:
            print(f"\n随机抽样 {self.sample_size:,} 条...")
            indices = np.random.choice(len(docs), self.sample_size, replace=False)
            docs = [docs[i] for i in indices]
            metadata = [metadata[i] for i in indices]
        
        print(f"\n✓ 加载完成: {len(docs):,} 条职位描述")
        
        return docs, metadata
    
    def preprocess(self, docs):
        """文本预处理"""
        print("\n" + "="*80)
        print("阶段2: 文本预处理与词典构建")
        print("="*80)
        
        texts = []
        
        print("中文分词与词性标注...")
        for i, doc in enumerate(docs):
            # 分词 + 词性标注
            words = pseg.cut(str(doc))
            
            # 保留名词、动词、形容词
            filtered = [
                w for w, flag in words
                if len(w) >= 2 
                and w not in self.stopwords
                and flag.startswith(('n', 'v', 'a'))  # 名词/动词/形容词
            ]
            
            texts.append(filtered)
            
            if (i + 1) % 10000 == 0:
                print(f"  已处理 {i+1:,}/{len(docs):,}")
        
        print(f"✓ 分词完成")
        
        # 构建词典
        print("\n构建词典...")
        dictionary = corpora.Dictionary(texts)
        
        print(f"原始词典大小: {len(dictionary)}")
        
        # 过滤极端词
        dictionary.filter_extremes(
            no_below=10,      # 至少出现在10个文档
            no_above=0.5,     # 不超过50%文档
            keep_n=50000      # 保留最常见的5万词
        )
        
        print(f"过滤后词典大小: {len(dictionary)}")
        
        # 转换为词袋
        print("\n转换为词袋表示...")
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        print(f"✓ 语料库构建完成")
        print(f"  文档数: {len(corpus):,}")
        print(f"  词汇数: {len(dictionary):,}")
        
        self.dictionary = dictionary
        self.corpus = corpus
        
        return texts, dictionary, corpus
    
    def train_lda(self):
        """训练LDA模型"""
        print("\n" + "="*80)
        print("阶段3: LDA模型训练")
        print("="*80)
        print(f"主题数: {self.num_topics}")
        print(f"文档数: {len(self.corpus):,}")
        print("\n⚠️  训练可能需要1-3小时...")
        
        # 训练LDA
        lda_model = models.LdaMulticore(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            chunksize=2000,
            passes=10,
            alpha='auto',
            eta='auto',
            per_word_topics=True,
            workers=4
        )
        
        print("\n✓ 训练完成！")
        
        self.lda_model = lda_model
        
        return lda_model
    
    def evaluate_model(self):
        """评估模型"""
        print("\n" + "="*80)
        print("阶段4: 模型评估")
        print("="*80)
        
        # Perplexity
        perplexity = self.lda_model.log_perplexity(self.corpus)
        print(f"Perplexity: {perplexity:.4f} (越低越好)")
        
        # Coherence
        print("\n计算Coherence Score...")
        # 需要原始文本
        texts = [[self.dictionary[id] for id, _ in doc] for doc in self.corpus[:1000]]
        
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        coherence = coherence_model.get_coherence()
        print(f"Coherence Score: {coherence:.4f} (越高越好)")
        
        return perplexity, coherence
    
    def extract_skill_profiles(self):
        """提取技能组合profile"""
        print("\n" + "="*80)
        print("阶段5: 提取技能组合")
        print("="*80)
        
        skill_profiles = []
        
        for topic_id in range(self.num_topics):
            # 获取主题的Top 30关键词
            top_words = self.lda_model.show_topic(topic_id, topn=30)
            
            keywords = [word for word, prob in top_words]
            probabilities = [prob for word, prob in top_words]
            
            skill_profiles.append({
                'skill_id': topic_id,
                'keywords': ', '.join(keywords),
                'top_10': ', '.join(keywords[:10]),
                'avg_probability': np.mean(probabilities)
            })
        
        skill_df = pd.DataFrame(skill_profiles)
        
        print(f"✓ 提取了 {len(skill_df)} 个技能组合")
        
        return skill_df
    
    def compute_document_distributions(self, docs):
        """计算文档-技能组合分布（θ）"""
        print("\n" + "="*80)
        print("阶段6: 计算文档-技能分布")
        print("="*80)
        
        doc_topics = []
        
        print("为每个文档计算技能profile...")
        
        for i, doc_bow in enumerate(self.corpus):
            # 获取文档的主题分布
            topic_dist = self.lda_model.get_document_topics(doc_bow, minimum_probability=0.01)
            
            # 转换为字典
            topic_dict = {f'skill_{tid}': prob for tid, prob in topic_dist}
            
            # 添加原始文档
            topic_dict['document'] = docs[i]
            
            doc_topics.append(topic_dict)
            
            if (i + 1) % 10000 == 0:
                print(f"  已处理 {i+1:,}/{len(self.corpus):,}")
        
        doc_topic_df = pd.DataFrame(doc_topics).fillna(0)
        
        print(f"✓ 完成文档-技能分布计算")
        
        return doc_topic_df
    
    def save_results(self, skill_df, doc_topic_df, metadata):
        """保存结果"""
        print("\n" + "="*80)
        print("阶段7: 保存结果")
        print("="*80)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. 保存技能组合关键词
        skill_file = os.path.join(self.output_dir, 'skill_keywords.csv')
        skill_df.to_csv(skill_file, index=False, encoding='utf-8-sig')
        print(f"✓ 技能关键词: {skill_file}")
        
        # 2. 保存文档-技能分布
        doc_file = os.path.join(self.output_dir, 'document_skill_profiles.csv')
        doc_topic_df.to_csv(doc_file, index=False, encoding='utf-8-sig')
        print(f"✓ 文档技能分布: {doc_file}")
        
        # 3. 保存模型
        model_file = os.path.join(self.output_dir, 'lda_model')
        self.lda_model.save(model_file)
        print(f"✓ LDA模型: {model_file}")
        
        # 4. 保存词典
        dict_file = os.path.join(self.output_dir, 'dictionary.dict')
        self.dictionary.save(dict_file)
        print(f"✓ 词典: {dict_file}")
        
        # 5. 保存语料库
        corpus_file = os.path.join(self.output_dir, 'corpus.mm')
        corpora.MmCorpus.serialize(corpus_file, self.corpus)
        print(f"✓ 语料库: {corpus_file}")
        
        # 6. 生成人工审查报告
        report_file = os.path.join(self.output_dir, 'skill_review_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LDA技能组合识别 - 人工审查报告\n")
            f.write("理论框架: Occupations as bundles of skills\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"总文档数: {len(doc_topic_df):,}\n")
            f.write(f"技能组合数: {self.num_topics}\n")
            f.write(f"词汇数: {len(self.dictionary):,}\n\n")
            
            f.write("="*80 + "\n")
            f.write("技能组合详情\n")
            f.write("="*80 + "\n\n")
            
            for idx, row in skill_df.iterrows():
                f.write(f"技能组合 #{row['skill_id']}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Top 30 关键词:\n{row['keywords']}\n\n")
                
                # 找到该技能组合最强的文档
                skill_col = f"skill_{row['skill_id']}"
                if skill_col in doc_topic_df.columns:
                    top_docs = doc_topic_df.nlargest(3, skill_col)
                    f.write("代表性职位描述:\n")
                    for i, (_, doc_row) in enumerate(top_docs.iterrows(), 1):
                        doc_preview = str(doc_row['document'])[:200]
                        f.write(f"  {i}. {doc_preview}...\n")
                        f.write(f"     (技能强度: {doc_row[skill_col]:.3f})\n\n")
                
                f.write("\n" + "-"*80 + "\n\n")
        
        print(f"✓ 审查报告: {report_file}")
        
        print("\n" + "="*80)
        print("所有结果已保存！")
        print("="*80)
    
    def run(self):
        """执行完整流程"""
        # 1. 加载数据
        docs, metadata = self.load_data()
        
        # 2. 预处理
        texts, dictionary, corpus = self.preprocess(docs)
        
        # 3. 训练LDA
        lda_model = self.train_lda()
        
        # 4. 评估
        perplexity, coherence = self.evaluate_model()
        
        # 5. 提取技能组合
        skill_df = self.extract_skill_profiles()
        
        # 6. 计算文档分布
        doc_topic_df = self.compute_document_distributions(docs)
        
        # 7. 保存结果
        self.save_results(skill_df, doc_topic_df, metadata)
        
        return lda_model, skill_df, doc_topic_df


def main():
    """主函数"""
    print("="*80)
    print("LDA职业技能组合识别")
    print("="*80)

    # 使用集中路径配置
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.paths import CLEANED_DATA, LDA_OUTPUT_DIR

    # 确保输出目录存在
    LDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_csv = str(CLEANED_DATA)
    output_dir = str(LDA_OUTPUT_DIR)
    
    print(f"\n配置:")
    print(f"  输入: {input_csv}")
    print(f"  输出: {output_dir}")
    print(f"  技能组合数: 80")
    
    # 抽样选择
    print(f"\n数据量: 1300万条")
    print(f"抽样方案:")
    print(f"  1. 快速测试: 10万条（30分钟）")
    print(f"  2. 标准分析: 50万条（2-3小时）")
    print(f"  3. 全量分析: 1300万条（8-12小时）")
    
    choice = input(f"\n请选择 (1/2/3): ").strip()
    
    if choice == '1':
        sample_size = 100000
    elif choice == '2':
        sample_size = 500000
    elif choice == '3':
        sample_size = None
    else:
        print("无效选择，使用默认: 10万条")
        sample_size = 100000
    
    # 执行
    lda = SkillBundleLDA(
        input_csv=input_csv,
        output_dir=output_dir,
        num_topics=80,
        sample_size=sample_size
    )
    
    lda_model, skill_df, doc_topic_df = lda.run()
    
    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n下一步:")
    print(f"  1. 查看 skill_review_report.txt")
    print(f"  2. 人工标注技能类别 (build_codebook.py)")
    print(f"  3. 趋势与异质性分析")


if __name__ == "__main__":
    main()
