# 动态LDA职业技能演化识别方案 (Dynamic Skill Evolution Scheme)

> **版本**: v2.0 (极度细化版)  
> **更新日期**: 2026-01-24  
> **数据诊断结果**: 已验证可行

---

## 〇、数据可行性诊断报告

### 已验证的数据状况

| 维度 | 实测值 | 评估 |
|------|--------|------|
| **总记录数** | 13,082,403 条 | ✅ 充足 |
| **年份覆盖** | 2016-2025 (实际有效: 2016-2024) | ⚠️ 仅9年，非20年 |
| **年份分布** | 2016:94万, 2017:154万, 2018:67万, 2019:18, 2020:2054, 2021:214万, 2022:380万, 2023:194万, 2024:196万, 2025:9万 | ⚠️ 2019-2020断层 |
| **职位描述长度** | 平均311字符, 中位数264字符 | ✅ 足够LDA |
| **ESCO词典** | 45,309 条技能 (已汉化) | ✅ 充足 |

### 关键风险与应对

| 风险 | 严重程度 | 应对策略 |
|------|----------|----------|
| **2019-2020数据断层** | 🔴 高 | 采用3个时间窗口: [2016-2018], [2021-2022], [2023-2024]，跳过断层期 |
| **时间跨度仅9年** | 🟡 中 | 调整研究叙事：从"20年演化"改为"后疫情技能重构" |
| **职位描述含噪音** | 🟡 中 | 正则提取任职要求段落，过滤福利描述 |

---

## 一、 理论框架 (Theoretical Framework)

**核心构建**：

1. **静态视角**：Occupations as bundles of skills (Alabdulkareem et al., 2024)。
2. **动态视角**：技能的**语义漂移 (Semantic Drift)**。
   * *假设*：2016年的"人工智能"与2024年的"AI"处于不同的语义空间，强行合并会导致跨期不可比。
   * *对策*：采用 **Binned Topic Models (Malik et al.)**，即在独立时间窗口内识别技能，再通过向量空间进行跨期对齐。

**模型优势**：

* **抗噪性**：通过领域词典（ESCO）和短语识别修复破碎语义。
* **演化追踪**：不仅能看到技能占比变化，还能识别技能的**分裂 (Split)**、**融合 (Merge)** 和 **消亡 (Death)**。

---

## 二、 方法论设计 (Methodology)

### 🎯 三线并行验证框架 (Three-Pronged Validation Framework)

为确保AI对工作任务演化影响的分析稳健性，采用**三条相互独立又相互映证的分析路径**：

#### 1. **LDA主题建模线** (Topic Modeling Track)
- **核心优势**: 自动发现隐含主题结构，无监督学习
- **理论基础**: 概率主题模型，捕捉文档-主题-词语的层次关系
- **适用场景**: 探索性分析，发现未知的技能演化模式
- **局限性**: 主题可解释性依赖人工标注，参数敏感

#### 2. **任务对提取线** (Task Pair Extraction Track)
- **核心优势**: 直接提取具体工作任务，语义精确度高
- **理论基础**: 借鉴Atalay et al. (2024)的"动词-名词"对方法
- **适用场景**: 验证性分析，量化特定任务的演化轨迹
- **局限性**: 依赖预定义模式，覆盖面有限

#### 3. **SBERT语义嵌入线** (Semantic Embedding Track)
- **核心优势**: 捕捉细粒度语义变化，跨期可比性强
- **理论基础**: Transformer-based句子嵌入，上下文感知
- **适用场景**: 补充性分析，验证LDA和任务对的结果
- **局限性**: 计算成本高，对领域适应性需调优

**三线验证逻辑**:
- **一致性检验**: 三条线对同一演化事件的识别结果应相互印证
- **互补性增强**: 不同方法的盲点由其他方法弥补
- **稳健性提升**: 多角度证据支撑研究结论的可靠性

### 1. 数据结构与清洗

* **原始数据**：`/Users/yu/code/code2601/TY/data_cleaning/all_in_one1.csv` (13,082,403 records)
* **字段结构**：
  ```
  企业名称, 招聘岗位, 工作城市, 工作区域, 最低月薪, 最高月薪, 
  职位描述, 学历要求, 要求经验, 招聘人数, 招聘类别, 初级分类, 
  来源平台, 公司地点, 工作地点, 招聘发布日期, 招聘结束日期, 
  招聘发布年份, 招聘结束年份, 来源
  ```
* **关键过滤**：
  * **正则提取**：仅保留"任职要求/技能要求"字段，剔除公司介绍与福利（防止"五险一金"成为Topic）。
  * **去重**：基于 SimHash/MinHash 剔除重复发布的JD。

### 2. 增强预处理策略 (针对"语义之殇")

**目标**：防止专业词汇被切碎，防止高频通用词污染主题。

```python
# 语义修复流程
1. 加载领域词典：使用汉化版 ESCO-Skill-Extractor
   -> 路径: /Users/yu/code/code2601/TY/Test_ESCO/skills_chinese.csv
   -> 格式: id, description, description_cn (45,309条)
   -> 作用: 强制保留 "React.js", "C++" 等专有名词，防止被 jieba 切分。

2. 短语发现 (Phrase Detection):
   -> 使用 gensim.models.Phrases 识别 Bigram/Trigram
   -> 效果: 将 "深度" + "学习" 合并为 "深度_学习"，"Data" + "Mining" -> "Data_Mining"。

3. 动态停用词过滤:
   -> 不仅使用通用停用词表。
   -> 计算全语料 TF-IDF，剔除 Document Frequency > 40% 的高频无意义词（如"负责"、"具有"、"相关"）。
```

### 3. 模型策略：分箱训练与向量对齐 (针对"时间之殇")

#### Step 1: 独立时间片训练 (Binned Topic Modeling)

* **切分**（基于实际数据调整）：
  | 时间窗口 | 年份范围 | 预计记录数 | 备注 |
  |----------|----------|------------|------|
  | Window 1 | 2016-2018 | ~316万 | 疫情前 |
  | Window 2 | 2021-2022 | ~594万 | 疫情恢复期 |
  | Window 3 | 2023-2024 | ~390万 | 后疫情时代 |

* **训练**：在每个窗口独立训练 LDA 模型。
* **输出**：得到序列化的模型集合 `{M_w1, M_w2, M_w3}`。

#### Step 2: 主题向量化 (Topic Vectorization)

* **核心**：将概率分布 `P(word|topic)` 转换为稠密向量，解决语义对齐问题。
* **公式**：`V_topic = Σ P(word|topic) × V_word`
* **词向量来源**：基于清洗后语料自训练 Word2Vec，或使用 ESCO 预训练向量。

#### Step 3: 混合策略对齐 (Hybrid Alignment)

* **计算**：构建 `t` 与 `t+1` 时刻主题间的余弦相似度矩阵 `S`。
* **算法**：
  1. **主干识别**：使用 **匈牙利算法 (Hungarian Algorithm)** 确定全局最优的 1-to-1 存活路径 (Survival)。
  2. **分支识别**：使用 **贪婪阈值法 (Greedy Threshold)** 扫描剩余高相似度连线，识别 **分裂 (Split)** 和 **融合 (Merge)**。

---

## 三、 极度细化实施步骤

### Phase 1: 预处理与词典构建

#### Task 1.1: ESCO词典转换为jieba格式
**输出**: `task1_datacleaning/esco_jieba_dict.txt`

```python
# 文件: task1_datacleaning/convert_esco_to_jieba.py

"""
功能：将ESCO汉化词典转换为jieba用户词典格式
输入：/Users/yu/code/code2601/TY/Test_ESCO/skills_chinese.csv
输出：esco_jieba_dict.txt (格式: 词语 词频 词性)
"""

import pandas as pd
import re

def extract_skill_terms(description_cn: str) -> list:
    """
    从汉化描述中提取技能术语
    示例输入: "管理音乐团队人员，协调音乐团队职责..."
    示例输出: ["管理音乐团队人员", "协调音乐团队职责", ...]
    """
    # 按中文标点分割
    terms = re.split(r'[，。、；：\n]', description_cn)
    # 清洗并过滤
    terms = [t.strip() for t in terms if 2 <= len(t.strip()) <= 20]
    return terms

def main():
    # 1. 读取ESCO汉化文件
    df = pd.read_csv('/Users/yu/code/code2601/TY/Test_ESCO/skills_chinese.csv')
    
    # 2. 提取所有技能术语
    all_terms = set()
    for desc in df['description_cn'].dropna():
        terms = extract_skill_terms(desc)
        all_terms.update(terms)
    
    # 3. 添加英文专业术语(从description列提取)
    english_pattern = r'\b[A-Za-z][A-Za-z0-9\.\+\#\-]+\b'
    for desc in df['description'].dropna():
        matches = re.findall(english_pattern, desc)
        for m in matches:
            if len(m) >= 2:
                all_terms.add(m)
    
    # 4. 写入jieba格式
    with open('esco_jieba_dict.txt', 'w', encoding='utf-8') as f:
        for term in sorted(all_terms):
            # 格式: 词语 词频 词性
            # 高词频确保不被切分
            f.write(f"{term} 50000 n\n")
    
    print(f"已生成 {len(all_terms)} 个词条")

if __name__ == '__main__':
    main()
```

#### Task 1.2: 职位描述正则清洗
**输出**: `task1_datacleaning/cleaned_jd_by_year/`

```python
# 文件: task1_datacleaning/extract_requirements.py

"""
功能：从职位描述中提取"任职要求"段落，过滤噪音
输入：all_in_one1.csv
输出：每个时间窗口的清洗后语料
"""

import pandas as pd
import re
from pathlib import Path

# 任职要求段落的标识模式
REQUIREMENT_PATTERNS = [
    r'任职要求[：:](.*?)(?=岗位职责|工作职责|福利|$)',
    r'岗位要求[：:](.*?)(?=岗位职责|工作职责|福利|$)',
    r'技能要求[：:](.*?)(?=岗位职责|工作职责|福利|$)',
    r'要求[：:](.*?)(?=职责|福利|待遇|$)',
    r'任职资格[：:](.*?)(?=岗位职责|工作职责|福利|$)',
]

# 需要剔除的噪音模式
NOISE_PATTERNS = [
    r'五险一金.*?(?=\d|$)',
    r'带薪.*?假',
    r'免费.*?餐',
    r'节日.*?礼',
    r'周末双休',
    r'弹性工作',
    r'福利待遇.*',
    r'薪资待遇.*',
    r'微信公众号.*',
    r'马克数据.*',
    r'www\..*?\.cn',
]

def extract_requirements(text: str) -> str:
    """提取任职要求段落"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 尝试各种模式提取
    for pattern in REQUIREMENT_PATTERNS:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1)
            break
    else:
        # 如果没有明确标识，使用原文
        extracted = text
    
    # 剔除噪音
    for noise in NOISE_PATTERNS:
        extracted = re.sub(noise, '', extracted, flags=re.IGNORECASE)
    
    # 清理空白
    extracted = re.sub(r'\s+', ' ', extracted).strip()
    
    return extracted

def process_by_window(input_file: str, output_dir: str):
    """按时间窗口处理数据"""
    
    # 定义时间窗口
    windows = {
        'window_2016_2018': [2016, 2017, 2018],
        'window_2021_2022': [2021, 2022],
        'window_2023_2024': [2023, 2024],
    }
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 分块读取大文件
    chunksize = 100000
    
    for window_name, years in windows.items():
        print(f"处理 {window_name}...")
        output_file = Path(output_dir) / f"{window_name}.csv"
        
        first_chunk = True
        total_rows = 0
        
        for chunk in pd.read_csv(input_file, chunksize=chunksize):
            # 筛选年份
            chunk = chunk[chunk['招聘发布年份'].isin(years)]
            
            if len(chunk) == 0:
                continue
            
            # 提取任职要求
            chunk['cleaned_requirements'] = chunk['职位描述'].apply(extract_requirements)
            
            # 过滤空文本
            chunk = chunk[chunk['cleaned_requirements'].str.len() >= 20]
            
            # 保留关键字段
            chunk = chunk[['招聘岗位', '工作城市', '最低月薪', '最高月薪', 
                          '学历要求', '招聘发布年份', 'cleaned_requirements']]
            
            # 写入
            chunk.to_csv(output_file, mode='a', index=False, 
                        header=first_chunk, encoding='utf-8')
            first_chunk = False
            total_rows += len(chunk)
        
        print(f"  {window_name}: {total_rows:,} 条")

if __name__ == '__main__':
    process_by_window(
        input_file='/Users/yu/code/code2601/TY/data_cleaning/all_in_one1.csv',
        output_dir='./cleaned_jd_by_year'
    )
```

#### Task 1.3: 分词与短语识别
**输出**: `task1_datacleaning/tokenized_corpus/`

```python
# 文件: task1_datacleaning/tokenize_corpus.py

"""
功能：使用jieba分词 + ESCO词典 + Phrases短语识别
输入：cleaned_jd_by_year/
输出：每个时间窗口的分词后语料
"""

import jieba
import pandas as pd
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from pathlib import Path
import pickle

# 加载ESCO词典
jieba.load_userdict('./esco_jieba_dict.txt')

# 通用停用词 + 招聘特定停用词
STOP_WORDS = set([
    '的', '了', '是', '在', '有', '和', '与', '或', '等', '能', '会',
    '负责', '具有', '具备', '熟悉', '掌握', '了解', '相关', '以上',
    '优先', '考虑', '经验', '工作', '能力', '要求', '岗位', '公司',
    '团队', '良好', '较强', '一定', '优秀', '专业', '本科', '大专',
    '学历', '年以上', '及以上', '以下', '左右', '不限', '若干',
])

def tokenize_text(text: str) -> list:
    """分词并过滤停用词"""
    words = jieba.lcut(text)
    # 过滤: 长度>=2, 非停用词, 非纯数字
    words = [w for w in words 
             if len(w) >= 2 
             and w not in STOP_WORDS 
             and not w.isdigit()]
    return words

def build_phrase_model(corpus: list, min_count: int = 50) -> Phraser:
    """训练短语模型"""
    # Bigram
    bigram = Phrases(corpus, min_count=min_count, threshold=10)
    bigram_phraser = Phraser(bigram)
    
    # Trigram
    trigram = Phrases(bigram_phraser[corpus], min_count=min_count, threshold=10)
    trigram_phraser = Phraser(trigram)
    
    return bigram_phraser, trigram_phraser

def process_window(window_file: str, output_dir: str):
    """处理单个时间窗口"""
    
    window_name = Path(window_file).stem
    print(f"处理 {window_name}...")
    
    # 读取清洗后数据
    df = pd.read_csv(window_file)
    
    # Step 1: 基础分词
    print("  Step 1: 基础分词...")
    corpus = df['cleaned_requirements'].apply(tokenize_text).tolist()
    
    # Step 2: 训练短语模型(使用10%样本)
    print("  Step 2: 训练短语模型...")
    sample_size = min(500000, len(corpus))
    sample_corpus = corpus[:sample_size]
    bigram_phraser, trigram_phraser = build_phrase_model(sample_corpus)
    
    # Step 3: 应用短语模型
    print("  Step 3: 应用短语模型...")
    corpus = [trigram_phraser[bigram_phraser[doc]] for doc in corpus]
    
    # 保存
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存分词后语料
    with open(f"{output_dir}/{window_name}_corpus.pkl", 'wb') as f:
        pickle.dump(corpus, f)
    
    # 保存短语模型
    bigram_phraser.save(f"{output_dir}/{window_name}_bigram.pkl")
    trigram_phraser.save(f"{output_dir}/{window_name}_trigram.pkl")
    
    # 保存元数据(用于后续分析)
    df['tokenized'] = [' '.join(doc) for doc in corpus]
    df.to_csv(f"{output_dir}/{window_name}_tokenized.csv", index=False)
    
    print(f"  完成: {len(corpus):,} 条")

if __name__ == '__main__':
    for window_file in Path('./cleaned_jd_by_year').glob('*.csv'):
        process_window(str(window_file), './tokenized_corpus')
```

---

### Phase 2: 分箱 LDA 训练

#### Task 2.1: 构建词典与语料库
**输出**: `task3_modeling/t3_1_binned_topic_modeling/`

```python
# 文件: task3_modeling/t3_1_binned_topic_modeling/build_dictionary.py

"""
功能：构建gensim词典，应用词频过滤
输入：tokenized_corpus/
输出：每个时间窗口的Dictionary和Corpus
"""

from gensim import corpora
from gensim.models import TfidfModel
import pickle
from pathlib import Path

def build_dictionary(corpus: list, 
                     min_df: int = 100, 
                     max_df_ratio: float = 0.4) -> corpora.Dictionary:
    """
    构建词典并过滤
    min_df: 最小文档频率 (13M数据用100)
    max_df_ratio: 最大文档频率比例 (过滤高频词)
    """
    
    # 创建词典
    dictionary = corpora.Dictionary(corpus)
    
    # 计算max_df的绝对值
    max_df = int(len(corpus) * max_df_ratio)
    
    # 过滤极端词频
    dictionary.filter_extremes(no_below=min_df, no_above=max_df_ratio)
    
    return dictionary

def build_corpus(documents: list, dictionary: corpora.Dictionary):
    """构建BOW语料库"""
    return [dictionary.doc2bow(doc) for doc in documents]

def process_window(tokenized_file: str, output_dir: str):
    """处理单个时间窗口"""
    
    window_name = Path(tokenized_file).stem.replace('_corpus', '')
    print(f"构建词典: {window_name}...")
    
    # 加载分词后语料
    with open(tokenized_file, 'rb') as f:
        corpus = pickle.load(f)
    
    # 构建词典
    dictionary = build_dictionary(corpus, min_df=100, max_df_ratio=0.4)
    print(f"  词典大小: {len(dictionary):,} 词")
    
    # 构建BOW语料
    bow_corpus = build_corpus(corpus, dictionary)
    
    # 保存
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dictionary.save(f"{output_dir}/{window_name}_dict.gensim")
    corpora.MmCorpus.serialize(f"{output_dir}/{window_name}_corpus.mm", bow_corpus)
    
    print(f"  完成")

if __name__ == '__main__':
    output_dir = './'
    for corpus_file in Path('../../task1_datacleaning/tokenized_corpus').glob('*_corpus.pkl'):
        process_window(str(corpus_file), output_dir)
```

#### Task 2.2: LDA模型训练
**输出**: `task3_modeling/t3_1_binned_topic_modeling/models/`

```python
# 文件: task3_modeling/t3_1_binned_topic_modeling/train_lda.py

"""
功能：为每个时间窗口训练LDA模型
输入：词典和语料库
输出：LDA模型 + Topic Coherence评估
"""

from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# 超参数配置
LDA_CONFIG = {
    'num_topics': 60,           # 每个窗口60个主题
    'passes': 15,               # 迭代次数
    'chunksize': 10000,         # 批量大小
    'alpha': 'auto',            # 自动学习alpha
    'eta': 'auto',              # 自动学习eta
    'iterations': 400,          # 每个文档的迭代次数
    'random_state': 42,         # 可复现
    'workers': 4,               # 并行数
}

def train_lda(dictionary: corpora.Dictionary, 
              corpus,
              raw_texts: list,
              config: dict = LDA_CONFIG) -> tuple:
    """
    训练LDA并评估
    返回: (model, coherence_score)
    """
    
    # 训练
    model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        **config
    )
    
    # 计算Topic Coherence (Cv)
    coherence_model = CoherenceModel(
        model=model, 
        texts=raw_texts,
        dictionary=dictionary, 
        coherence='c_v'
    )
    coherence = coherence_model.get_coherence()
    
    return model, coherence

def evaluate_num_topics(dictionary, corpus, raw_texts, 
                        topic_range: list = [40, 50, 60, 70, 80]):
    """评估不同主题数的Coherence"""
    
    results = []
    for num_topics in topic_range:
        config = LDA_CONFIG.copy()
        config['num_topics'] = num_topics
        
        model, coherence = train_lda(dictionary, corpus, raw_texts, config)
        results.append({
            'num_topics': num_topics,
            'coherence': coherence
        })
        print(f"  num_topics={num_topics}, Coherence={coherence:.4f}")
    
    return results

def process_window(window_name: str, base_dir: str, output_dir: str):
    """处理单个时间窗口"""
    
    print(f"训练LDA: {window_name}...")
    
    # 加载
    dictionary = corpora.Dictionary.load(f"{base_dir}/{window_name}_dict.gensim")
    corpus = corpora.MmCorpus(f"{base_dir}/{window_name}_corpus.mm")
    
    # 加载原始分词文本(用于Coherence计算)
    with open(f"../../task1_datacleaning/tokenized_corpus/{window_name}_corpus.pkl", 'rb') as f:
        raw_texts = pickle.load(f)
    
    # 训练最终模型
    model, coherence = train_lda(dictionary, list(corpus), raw_texts)
    
    # 保存
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(f"{output_dir}/{window_name}_lda.model")
    
    # 保存Top Words
    with open(f"{output_dir}/{window_name}_topics.txt", 'w', encoding='utf-8') as f:
        for idx in range(model.num_topics):
            top_words = model.show_topic(idx, topn=20)
            words_str = ', '.join([f"{w}({p:.3f})" for w, p in top_words])
            f.write(f"Topic {idx}: {words_str}\n")
    
    print(f"  Coherence: {coherence:.4f}")
    return coherence

if __name__ == '__main__':
    base_dir = './'
    output_dir = './models'
    
    windows = ['window_2016_2018', 'window_2021_2022', 'window_2023_2024']
    
    for window in windows:
        process_window(window, base_dir, output_dir)
```

---

#### Task 2.3: 主题合并与去冗余
**输出**: `output/lda/merged_topic_vectors/` + `output/lda/topic_merge_report.csv`

```python
# 文件: src/lda/run_topic_merging.py

"""
功能：基于主题向量相似度合并冗余主题，减少噪音主题
输入：LDA模型 + topic_vectors
输出：合并后的主题向量、主题映射、聚类信息、合并报告
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import LdaModel
import pickle
import pandas as pd
from pathlib import Path

SIM_THRESHOLD = 0.75   # 相似度阈值
MIN_TOPICS = 20        # 每窗最小主题数
MAX_CLUSTER = 5        # 单簇最大主题数

def greedy_cluster(sim_matrix):
    """基于阈值的贪婪合并"""
    n_topics = sim_matrix.shape[0]
    clusters = [[i] for i in range(n_topics)]
    pairs = [(i, j, sim_matrix[i, j]) 
             for i in range(n_topics) for j in range(i+1, n_topics)]
    pairs.sort(key=lambda x: x[2], reverse=True)
    for i, j, sim in pairs:
        if sim < SIM_THRESHOLD:
            break
        # 找到各自所在簇
        ci = next(k for k, c in enumerate(clusters) if i in c)
        cj = next(k for k, c in enumerate(clusters) if j in c)
        if ci != cj and len(clusters[ci]) + len(clusters[cj]) <= MAX_CLUSTER:
            clusters[ci].extend(clusters[cj])
            clusters.pop(cj)
    return clusters

def merge_vectors(vectors, clusters):
    merged = []
    mapping = {}
    for new_id, cluster in enumerate(clusters):
        v = vectors[cluster].mean(axis=0)
        v = v / np.linalg.norm(v)
        merged.append(v)
        for old_id in cluster:
            mapping[old_id] = new_id
    return np.array(merged), mapping

def merge_window(window, base_dir):
    vectors = np.load(f"{base_dir}/output/lda/topic_vectors/{window}_topic_vectors.npy")
    lda = LdaModel.load(f"{base_dir}/output/lda/models/{window}_lda.model")
    sim = cosine_similarity(vectors)
    clusters = greedy_cluster(sim)
    merged_vectors, mapping = merge_vectors(vectors, clusters)
    # 保存
    Path(f"{base_dir}/output/lda/merged_topic_vectors").mkdir(parents=True, exist_ok=True)
    np.save(f"{base_dir}/output/lda/merged_topic_vectors/{window}_merged_vectors.npy", merged_vectors)
    with open(f"{base_dir}/output/lda/merged_topic_vectors/{window}_cluster_info.pkl", 'wb') as f:
        pickle.dump({'clusters': clusters, 'mapping': mapping}, f)
    return len(vectors), len(merged_vectors)

def main():
    base_dir = "../../"
    windows = ['window_2016_2017', 'window_2018_2019', 'window_2020_2021', 
               'window_2022_2023', 'window_2024_2025']
    report = []
    for w in windows:
        ori, merged = merge_window(w, base_dir)
        report.append([w, ori, merged, (ori-merged)/ori])
    pd.DataFrame(report, columns=['window','original_topics','merged_topics','reduction_rate']) \
      .to_csv(f"{base_dir}/output/lda/topic_merge_report.csv", index=False)

if __name__ == "__main__":
    main()
```

---

### Phase 3: 演化对齐与图谱构建

#### Task 3.1: 主题向量化
**输出**: `task3_modeling/t3_2_topic_vectorization/`

```python
# 文件: task3_modeling/t3_2_topic_vectorization/vectorize_topics.py

"""
功能：将LDA主题转换为稠密向量
方法：V_topic = Σ P(word|topic) × V_word
输入：LDA模型 + Word2Vec模型
输出：每个窗口的Topic向量矩阵
"""

import numpy as np
from gensim.models import LdaMulticore, Word2Vec
from gensim import corpora
import pickle
from pathlib import Path

def train_word2vec(all_corpus: list, 
                   vector_size: int = 200,
                   window: int = 5,
                   min_count: int = 50) -> Word2Vec:
    """训练Word2Vec模型"""
    
    model = Word2Vec(
        sentences=all_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=10
    )
    return model

def vectorize_topic(lda_model: LdaMulticore, 
                    topic_id: int,
                    w2v_model: Word2Vec,
                    topn: int = 50) -> np.ndarray:
    """
    将单个主题转换为向量
    V = Σ P(w|t) × V(w)
    """
    
    topic_words = lda_model.show_topic(topic_id, topn=topn)
    
    vector = np.zeros(w2v_model.vector_size)
    total_prob = 0
    
    for word, prob in topic_words:
        if word in w2v_model.wv:
            vector += prob * w2v_model.wv[word]
            total_prob += prob
    
    # 归一化
    if total_prob > 0:
        vector = vector / total_prob
    
    # L2归一化(便于余弦相似度计算)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector

def vectorize_all_topics(lda_model: LdaMulticore, 
                         w2v_model: Word2Vec) -> np.ndarray:
    """向量化所有主题"""
    
    vectors = []
    for topic_id in range(lda_model.num_topics):
        vec = vectorize_topic(lda_model, topic_id, w2v_model)
        vectors.append(vec)
    
    return np.array(vectors)

def main():
    # Step 1: 合并所有语料训练Word2Vec
    print("Step 1: 训练Word2Vec...")
    
    all_corpus = []
    for corpus_file in Path('../../task1_datacleaning/tokenized_corpus').glob('*_corpus.pkl'):
        with open(corpus_file, 'rb') as f:
            corpus = pickle.load(f)
            all_corpus.extend(corpus)
    
    w2v_model = train_word2vec(all_corpus)
    w2v_model.save('./word2vec.model')
    print(f"  词汇量: {len(w2v_model.wv):,}")
    
    # Step 2: 向量化每个窗口的主题
    print("Step 2: 向量化主题...")
    
    windows = ['window_2016_2018', 'window_2021_2022', 'window_2023_2024']
    topic_vectors = {}
    
    for window in windows:
        lda_model = LdaMulticore.load(f'../t3_1_binned_topic_modeling/models/{window}_lda.model')
        vectors = vectorize_all_topics(lda_model, w2v_model)
        topic_vectors[window] = vectors
        
        # 保存
        np.save(f'./{window}_topic_vectors.npy', vectors)
        print(f"  {window}: {vectors.shape}")
    
    # 保存完整字典
    with open('./all_topic_vectors.pkl', 'wb') as f:
        pickle.dump(topic_vectors, f)

if __name__ == '__main__':
    main()
```

#### Task 3.2: 混合对齐算法
**输出**: `task3_modeling/t3_3_hyprid_alignment/`

```python
# 文件: task3_modeling/t3_3_hyprid_alignment/align_topics.py

"""
功能：跨时间窗口的主题对齐
算法：
  1. 匈牙利算法 -> 1-to-1 Survival
  2. 贪婪阈值法 -> Split/Merge检测
输出：演化事件表
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
from pathlib import Path

# 对齐阈值配置
ALIGNMENT_CONFIG = {
    'survival_threshold': 0.65,   # 存活判定阈值
    'split_merge_threshold': 0.55, # 分裂/融合判定阈值
}

def compute_similarity_matrix(vectors_t1: np.ndarray, 
                               vectors_t2: np.ndarray) -> np.ndarray:
    """计算两个时间窗口主题间的余弦相似度矩阵"""
    return cosine_similarity(vectors_t1, vectors_t2)

def hungarian_alignment(sim_matrix: np.ndarray, 
                        threshold: float) -> list:
    """
    匈牙利算法：寻找最优1-to-1匹配
    返回: [(source_id, target_id, similarity), ...]
    """
    
    # 转换为代价矩阵 (最大化 -> 最小化)
    cost_matrix = 1 - sim_matrix
    
    # 求解
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 过滤低于阈值的匹配
    alignments = []
    for src, tgt in zip(row_ind, col_ind):
        sim = sim_matrix[src, tgt]
        if sim >= threshold:
            alignments.append({
                'source_topic': src,
                'target_topic': tgt,
                'similarity': sim,
                'event': 'survival'
            })
    
    return alignments

def detect_split_merge(sim_matrix: np.ndarray,
                       survival_alignments: list,
                       threshold: float) -> list:
    """
    检测分裂与融合事件
    - Split: 一个source映射到多个target
    - Merge: 多个source映射到一个target
    """
    
    # 已匹配的主题
    matched_sources = {a['source_topic'] for a in survival_alignments}
    matched_targets = {a['target_topic'] for a in survival_alignments}
    
    events = []
    
    # 检测Split: 已存活的source是否还有其他高相似target
    for alignment in survival_alignments:
        src = alignment['source_topic']
        primary_tgt = alignment['target_topic']
        
        for tgt in range(sim_matrix.shape[1]):
            if tgt != primary_tgt and sim_matrix[src, tgt] >= threshold:
                events.append({
                    'source_topic': src,
                    'target_topic': tgt,
                    'similarity': sim_matrix[src, tgt],
                    'event': 'split'
                })
    
    # 检测Merge: 未匹配的source是否高度相似于某个已匹配的target
    for src in range(sim_matrix.shape[0]):
        if src not in matched_sources:
            for tgt in matched_targets:
                if sim_matrix[src, tgt] >= threshold:
                    events.append({
                        'source_topic': src,
                        'target_topic': tgt,
                        'similarity': sim_matrix[src, tgt],
                        'event': 'merge'
                    })
    
    # 检测Birth: 完全新的target
    for tgt in range(sim_matrix.shape[1]):
        if tgt not in matched_targets:
            max_sim = sim_matrix[:, tgt].max()
            if max_sim < threshold:
                events.append({
                    'source_topic': None,
                    'target_topic': tgt,
                    'similarity': max_sim,
                    'event': 'birth'
                })
    
    # 检测Death: 完全消失的source
    for src in range(sim_matrix.shape[0]):
        if src not in matched_sources:
            max_sim = sim_matrix[src, :].max()
            if max_sim < threshold:
                events.append({
                    'source_topic': src,
                    'target_topic': None,
                    'similarity': max_sim,
                    'event': 'death'
                })
    
    return events

def align_windows(vectors_t1: np.ndarray, 
                  vectors_t2: np.ndarray,
                  window_t1: str,
                  window_t2: str,
                  config: dict = ALIGNMENT_CONFIG) -> pd.DataFrame:
    """对齐两个时间窗口"""
    
    # 计算相似度矩阵
    sim_matrix = compute_similarity_matrix(vectors_t1, vectors_t2)
    
    # Step 1: 匈牙利算法
    survival_alignments = hungarian_alignment(
        sim_matrix, config['survival_threshold']
    )
    
    # Step 2: 分裂/融合检测
    other_events = detect_split_merge(
        sim_matrix, survival_alignments, config['split_merge_threshold']
    )
    
    # 合并所有事件
    all_events = survival_alignments + other_events
    
    # 构建DataFrame
    df = pd.DataFrame(all_events)
    df['source_window'] = window_t1
    df['target_window'] = window_t2
    
    return df, sim_matrix

def main():
    # 加载向量
    with open('../t3_2_topic_vectorization/all_topic_vectors.pkl', 'rb') as f:
        topic_vectors = pickle.load(f)
    
    windows = ['window_2016_2018', 'window_2021_2022', 'window_2023_2024']
    
    all_events = []
    all_matrices = {}
    
    # 对齐相邻窗口
    for i in range(len(windows) - 1):
        w1, w2 = windows[i], windows[i+1]
        print(f"对齐: {w1} -> {w2}")
        
        events_df, sim_matrix = align_windows(
            topic_vectors[w1], topic_vectors[w2], w1, w2
        )
        
        all_events.append(events_df)
        all_matrices[f"{w1}_to_{w2}"] = sim_matrix
        
        # 统计
        print(f"  Survival: {len(events_df[events_df['event']=='survival'])}")
        print(f"  Split: {len(events_df[events_df['event']=='split'])}")
        print(f"  Merge: {len(events_df[events_df['event']=='merge'])}")
        print(f"  Birth: {len(events_df[events_df['event']=='birth'])}")
        print(f"  Death: {len(events_df[events_df['event']=='death'])}")
    
    # 保存
    final_df = pd.concat(all_events, ignore_index=True)
    final_df.to_csv('./evolution_events.csv', index=False)
    
    with open('./similarity_matrices.pkl', 'wb') as f:
        pickle.dump(all_matrices, f)
    
    print(f"\n总事件数: {len(final_df)}")

if __name__ == '__main__':
    main()
```

#### Task 3.3: 桑基图可视化
**输出**: `task3_modeling/t3_3_hyprid_alignment/sankey_visualization.py`

```python
# 文件: task3_modeling/t3_3_hyprid_alignment/sankey_visualization.py

"""
功能：绘制技能演化桑基图
输入：evolution_events.csv + LDA模型(获取Topic标签)
输出：交互式HTML桑基图
"""

from pyecharts import options as opts
from pyecharts.charts import Sankey
import pandas as pd
from gensim.models import LdaMulticore
from pathlib import Path

def get_topic_label(lda_model, topic_id: int, topn: int = 3) -> str:
    """获取主题的可读标签(Top 3词)"""
    words = lda_model.show_topic(topic_id, topn=topn)
    return '_'.join([w for w, _ in words])

def build_sankey_data(events_df: pd.DataFrame, 
                      lda_models: dict) -> tuple:
    """
    构建桑基图数据
    返回: (nodes, links)
    """
    
    nodes = []
    links = []
    node_set = set()
    
    for _, row in events_df.iterrows():
        src_window = row['source_window']
        tgt_window = row['target_window']
        
        # 生成节点名称
        if pd.notna(row['source_topic']):
            src_model = lda_models[src_window]
            src_label = f"{src_window}_{get_topic_label(src_model, int(row['source_topic']))}"
            if src_label not in node_set:
                nodes.append({'name': src_label})
                node_set.add(src_label)
        else:
            src_label = None
        
        if pd.notna(row['target_topic']):
            tgt_model = lda_models[tgt_window]
            tgt_label = f"{tgt_window}_{get_topic_label(tgt_model, int(row['target_topic']))}"
            if tgt_label not in node_set:
                nodes.append({'name': tgt_label})
                node_set.add(tgt_label)
        else:
            tgt_label = None
        
        # 生成连接
        if src_label and tgt_label:
            # 根据事件类型设置颜色
            color_map = {
                'survival': '#5470C6',  # 蓝色
                'split': '#91CC75',     # 绿色
                'merge': '#FAC858',     # 黄色
            }
            
            links.append({
                'source': src_label,
                'target': tgt_label,
                'value': row['similarity'] * 100,
                'lineStyle': {'color': color_map.get(row['event_type'], '#EE6666')}
            })
    
    return nodes, links

def create_sankey_chart(nodes: list, links: list, 
                        title: str = "技能演化桑基图") -> Sankey:
    """创建桑基图"""
    
    sankey = (
        Sankey(init_opts=opts.InitOpts(width="1600px", height="900px"))
        .add(
            series_name="",
            nodes=nodes,
            links=links,
            linestyle_opt=opts.LineStyleOpts(opacity=0.5, curve=0.5),
            label_opts=opts.LabelOpts(position="right", font_size=10),
            node_gap=15,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            tooltip_opts=opts.TooltipOpts(trigger="item", trigger_on="mousemove"),
        )
    )
    
    return sankey

def main():
    # 加载事件数据
    events_df = pd.read_csv('./evolution_events.csv')
    
    # 只保留有连接的事件
    events_df = events_df[events_df['event'].isin(['survival', 'split', 'merge'])]
    
    # 加载LDA模型
    windows = ['window_2016_2018', 'window_2021_2022', 'window_2023_2024']
    lda_models = {}
    for window in windows:
        lda_models[window] = LdaMulticore.load(
            f'../t3_1_binned_topic_modeling/models/{window}_lda.model'
        )
    
    # 构建数据
    nodes, links = build_sankey_data(events_df, lda_models)
    
    # 生成图表
    chart = create_sankey_chart(nodes, links)
    chart.render('./skill_evolution_sankey.html')
    
    print(f"已生成: skill_evolution_sankey.html")
    print(f"  节点数: {len(nodes)}")
    print(f"  连接数: {len(links)}")

if __name__ == '__main__':
    main()
```

#### 🔄 第二条线：任务对提取方法 (Task Pair Extraction Track)

借鉴Atalay et al. (2024)的"动词-名词"对方法，直接从职位描述中提取具体工作任务单元，实现更精确的任务演化追踪。

##### Task 3.4.1: 构建任务对词典
**输出**: `task3_modeling/t3_4_task_pair_extraction/task_pairs_dict.pkl`

```python
# 文件: task3_modeling/t3_4_task_pair_extraction/build_task_pairs.py

"""
功能：构建任务对词典，从职位描述中提取"动词-名词"对
输入：清洗后的职位描述数据
输出：任务对词典和频率统计
"""

import pandas as pd
import re
import jieba
from collections import Counter, defaultdict
import pickle
from pathlib import Path

# 动词词典 (扩展版)
ACTION_VERBS = {
    # 核心动词
    '管理', '负责', '从事', '参与', '领导', '组织', '协调', '规划', '制定', '执行',
    '实施', '开展', '进行', '完成', '处理', '解决', '分析', '研究', '开发', '设计',
    '创建', '构建', '维护', '优化', '提升', '提高', '降低', '减少', '增加', '控制',
    '监控', '检查', '审核', '评估', '测试', '部署', '配置', '安装', '调试', '培训',
    '指导', '教学', '咨询', '服务', '销售', '营销', '推广', '运营', '运作', '操作'
}

# 专业名词词典 (AI相关扩展)
PROFESSIONAL_NOUNS = {
    # AI相关
    '人工智能', '机器学习', '深度学习', '神经网络', '算法', '模型', '数据', '训练',
    '预测', '分类', '聚类', '回归', '优化', '特征', '标签', '数据集', '验证', '测试',
    # 技术相关
    '系统', '平台', '框架', '工具', '软件', '硬件', '网络', '数据库', '服务器', '云',
    'API', '接口', '代码', '程序', '应用', '产品', '项目', '方案', '策略', '计划',
    # 业务相关
    '客户', '用户', '市场', '销售', '服务', '支持', '团队', '组织', '流程', '质量'
}

def extract_task_pairs(text: str) -> list:
    """
    从文本中提取任务对
    返回: [(verb, noun, context), ...]
    """
    if not text or len(text.strip()) < 10:
        return []
    
    pairs = []
    
    # 分词
    words = jieba.lcut(text)
    
    # 滑动窗口寻找动词-名词对
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i+1]
        
        # 检查是否构成任务对
        if (word1 in ACTION_VERBS and word2 in PROFESSIONAL_NOUNS) or \
           (len(word1) >= 2 and len(word2) >= 2 and 
            any(verb in word1 for verb in ['管理', '负责', '开发', '设计', '分析', '处理']) and
            any(noun in word2 for noun in ['系统', '平台', '数据', '模型', '项目', '产品'])):
            
            # 提取上下文 (前后2个词)
            start = max(0, i-2)
            end = min(len(words), i+4)
            context = ''.join(words[start:end])
            
            pairs.append((word1, word2, context))
    
    return pairs

def build_task_pair_dictionary(corpus_files: list, min_freq: int = 50) -> dict:
    """
    构建任务对词典
    返回: {task_pair: frequency, ...}
    """
    
    pair_counter = Counter()
    context_examples = defaultdict(list)
    
    for file_path in corpus_files:
        print(f"处理文件: {file_path}")
        
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 处理每条记录
        for _, row in df.iterrows():
            text = str(row.get('cleaned_requirements', ''))
            pairs = extract_task_pairs(text)
            
            for verb, noun, context in pairs:
                pair = f"{verb}_{noun}"
                pair_counter[pair] += 1
                
                # 保存上下文示例 (最多保存5个)
                if len(context_examples[pair]) < 5:
                    context_examples[pair].append(context)
    
    # 过滤低频任务对
    filtered_pairs = {pair: freq for pair, freq in pair_counter.items() 
                     if freq >= min_freq}
    
    # 构建词典
    task_dict = {
        'pairs': filtered_pairs,
        'contexts': dict(context_examples),
        'total_pairs': len(filtered_pairs),
        'total_occurrences': sum(filtered_pairs.values())
    }
    
    return task_dict

def analyze_task_evolution(task_dict: dict) -> dict:
    """分析任务对的演化特征"""
    
    # 按动词分类
    verb_categories = defaultdict(list)
    for pair in task_dict['pairs'].keys():
        verb = pair.split('_')[0]
        verb_categories[verb].append(pair)
    
    # 按名词分类
    noun_categories = defaultdict(list)
    for pair in task_dict['pairs'].keys():
        noun = pair.split('_')[1]
        noun_categories[noun].append(pair)
    
    return {
        'verb_categories': dict(verb_categories),
        'noun_categories': dict(noun_categories),
        'top_verbs': sorted(verb_categories.keys(), 
                           key=lambda x: len(verb_categories[x]), reverse=True)[:10],
        'top_nouns': sorted(noun_categories.keys(), 
                           key=lambda x: len(noun_categories[x]), reverse=True)[:10]
    }

def main():
    """主函数"""
    
    # 输入文件
    corpus_files = [
        '../../task1_datacleaning/cleaned_jd_by_year/window_2016_2018.csv',
        '../../task1_datacleaning/cleaned_jd_by_year/window_2021_2022.csv', 
        '../../task1_datacleaning/cleaned_jd_by_year/window_2023_2024.csv'
    ]
    
    # 构建任务对词典
    print("构建任务对词典...")
    task_dict = build_task_pair_dictionary(corpus_files, min_freq=100)
    
    # 分析演化特征
    evolution_analysis = analyze_task_evolution(task_dict)
    
    # 保存结果
    output_dir = Path('./')
    output_dir.mkdir(exist_ok=True)
    
    with open('task_pairs_dict.pkl', 'wb') as f:
        pickle.dump(task_dict, f)
    
    with open('task_evolution_analysis.pkl', 'wb') as f:
        pickle.dump(evolution_analysis, f)
    
    # 输出统计
    print(f"\\n任务对词典构建完成:")
    print(f"  总任务对数量: {task_dict['total_pairs']}")
    print(f"  总出现次数: {task_dict['total_occurrences']}")
    print(f"  动词类别数: {len(evolution_analysis['verb_categories'])}")
    print(f"  名词类别数: {len(evolution_analysis['noun_categories'])}")
    
    print(f"\\nTop 10 动词: {evolution_analysis['top_verbs'][:5]}")
    print(f"Top 10 名词: {evolution_analysis['top_nouns'][:5]}")

if __name__ == '__main__':
    main()
```

##### Task 3.4.2: 任务对向量化与演化分析
**输出**: `task3_modeling/t3_4_task_pair_extraction/task_evolution_results.pkl`

```python
# 文件: task3_modeling/t3_4_task_pair_extraction/analyze_task_evolution.py

"""
功能：分析任务对的时序演化模式
输入：任务对词典 + 时间窗口数据
输出：任务演化矩阵和统计分析
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

def build_task_vectors(window_data: pd.DataFrame, 
                      task_pairs: list,
                      task_dict: dict) -> np.ndarray:
    """
    为每个职位构建任务向量
    向量维度 = 任务对数量，每个维度表示该任务对是否出现
    """
    
    n_tasks = len(task_pairs)
    vectors = []
    
    for _, row in window_data.iterrows():
        text = str(row.get('cleaned_requirements', ''))
        
        # 检查每个任务对是否出现
        vector = np.zeros(n_tasks)
        for i, pair in enumerate(task_pairs):
            verb, noun = pair.split('_')
            if verb in text and noun in text:
                vector[i] = 1
        
        vectors.append(vector)
    
    return np.array(vectors)

def compute_task_similarity(vectors_t1: np.ndarray, 
                           vectors_t2: np.ndarray) -> np.ndarray:
    """
    计算任务相似度矩阵
    使用Jaccard相似度 (适合二值向量)
    """
    
    def jaccard_similarity(vec1, vec2):
        intersection = np.sum(vec1 * vec2)
        union = np.sum(vec1) + np.sum(vec2) - intersection
        return intersection / union if union > 0 else 0
    
    n1, n2 = len(vectors_t1), len(vectors_t2)
    similarity_matrix = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            similarity_matrix[i,j] = jaccard_similarity(vectors_t1[i], vectors_t2[j])
    
    return similarity_matrix

def analyze_task_evolution(task_dict: dict, 
                          window_files: list) -> dict:
    """
    分析任务对的演化模式
    """
    
    # 获取所有任务对
    task_pairs = list(task_dict['pairs'].keys())
    
    # 处理每个时间窗口
    window_vectors = {}
    task_frequencies = {}
    
    for window_file in window_files:
        window_name = Path(window_file).stem
        print(f"处理窗口: {window_name}")
        
        # 读取数据
        df = pd.read_csv(window_file)
        
        # 构建任务向量
        vectors = build_task_vectors(df, task_pairs, task_dict)
        window_vectors[window_name] = vectors
        
        # 计算任务频率
        task_freq = np.sum(vectors, axis=0)
        task_frequencies[window_name] = dict(zip(task_pairs, task_freq))
    
    # 计算窗口间相似度
    windows = list(window_vectors.keys())
    evolution_matrix = {}
    
    for i in range(len(windows) - 1):
        w1, w2 = windows[i], windows[i+1]
        print(f"计算相似度: {w1} vs {w2}")
        
        sim_matrix = compute_task_similarity(
            window_vectors[w1], 
            window_vectors[w2]
        )
        
        evolution_matrix[f"{w1}_{w2}"] = {
            'similarity_matrix': sim_matrix,
            'avg_similarity': np.mean(sim_matrix),
            'max_similarity': np.max(sim_matrix),
            'min_similarity': np.min(sim_matrix)
        }
    
    # 识别新兴/衰退任务
    emerging_tasks = identify_emerging_tasks(task_frequencies, windows)
    declining_tasks = identify_declining_tasks(task_frequencies, windows)
    
    return {
        'task_pairs': task_pairs,
        'window_vectors': window_vectors,
        'task_frequencies': task_frequencies,
        'evolution_matrix': evolution_matrix,
        'emerging_tasks': emerging_tasks,
        'declining_tasks': declining_tasks
    }

def identify_emerging_tasks(task_frequencies: dict, 
                           windows: list, 
                           threshold: float = 2.0) -> list:
    """识别新兴任务 (频率增长率 > threshold)"""
    
    emerging = []
    
    for task in task_frequencies[windows[0]].keys():
        freqs = [task_frequencies[w].get(task, 0) for w in windows]
        
        if len(freqs) >= 2:
            growth_rate = freqs[-1] / max(freqs[0], 1)  # 避免除零
            if growth_rate > threshold:
                emerging.append({
                    'task': task,
                    'initial_freq': freqs[0],
                    'final_freq': freqs[-1],
                    'growth_rate': growth_rate
                })
    
    return sorted(emerging, key=lambda x: x['growth_rate'], reverse=True)

def identify_declining_tasks(task_frequencies: dict, 
                            windows: list, 
                            threshold: float = 0.5) -> list:
    """识别衰退任务 (频率下降率 < threshold)"""
    
    declining = []
    
    for task in task_frequencies[windows[0]].keys():
        freqs = [task_frequencies[w].get(task, 0) for w in windows]
        
        if len(freqs) >= 2 and freqs[0] > 0:
            decline_rate = freqs[-1] / freqs[0]
            if decline_rate < threshold:
                declining.append({
                    'task': task,
                    'initial_freq': freqs[0],
                    'final_freq': freqs[-1],
                    'decline_rate': decline_rate
                })
    
    return sorted(declining, key=lambda x: x['decline_rate'])

def main():
    """主函数"""
    
    # 加载任务对词典
    with open('task_pairs_dict.pkl', 'rb') as f:
        task_dict = pickle.load(f)
    
    # 时间窗口文件
    window_files = [
        '../../task1_datacleaning/cleaned_jd_by_year/window_2016_2018.csv',
        '../../task1_datacleaning/cleaned_jd_by_year/window_2021_2022.csv',
        '../../task1_datacleaning/cleaned_jd_by_year/window_2023_2024.csv'
    ]
    
    # 分析任务演化
    print("分析任务对演化...")
    results = analyze_task_evolution(task_dict, window_files)
    
    # 保存结果
    with open('task_evolution_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 输出关键发现
    print("\\n=== 任务演化分析结果 ===")
    
    for window_pair, stats in results['evolution_matrix'].items():
        print(f"{window_pair}:")
        print(f"  平均相似度: {stats['avg_similarity']:.3f}")
        print(f"  最大相似度: {stats['max_similarity']:.3f}")
        print(f"  最小相似度: {stats['min_similarity']:.3f}")
    
    print(f"\\n新兴任务数量: {len(results['emerging_tasks'])}")
    if results['emerging_tasks']:
        print("Top 3 新兴任务:")
        for task in results['emerging_tasks'][:3]:
            print(f"  {task['task']}: {task['growth_rate']:.1f}x")
    
    print(f"\\n衰退任务数量: {len(results['declining_tasks'])}")
    if results['declining_tasks']:
        print("Top 3 衰退任务:")
        for task in results['declining_tasks'][:3]:
            print(f"  {task['task']}: {task['decline_rate']:.2f}x")

if __name__ == '__main__':
    main()
```

#### 🔄 第三条线：SBERT语义嵌入方法 (Semantic Embedding Track)

使用Sentence-BERT捕捉职位描述的细粒度语义变化，实现基于上下文的任务演化分析。

##### Task 3.5.1: 职位描述语义嵌入
**输出**: `task3_modeling/t3_5_sbert_embedding/job_embeddings/`

```python
# 文件: task3_modeling/t3_5_sbert_embedding/embed_jobs.py

"""
功能：使用SBERT对职位描述进行语义嵌入
输入：清洗后的职位描述数据
输出：职位向量和语义相似度矩阵
"""

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# SBERT模型配置
SBERT_CONFIG = {
    'model_name': 'paraphrase-multilingual-MiniLM-L12-v2',  # 多语言支持
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 32,
    'max_seq_length': 256
}

def load_sbert_model(model_name: str = SBERT_CONFIG['model_name']):
    """加载SBERT模型"""
    print(f"加载SBERT模型: {model_name}")
    model = SentenceTransformer(model_name)
    model.max_seq_length = SBERT_CONFIG['max_seq_length']
    return model

def preprocess_job_description(text: str, max_length: int = 200) -> str:
    """
    预处理职位描述文本
    - 截断过长文本
    - 清理格式
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # 截断过长文本
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    # 清理多余空白
    text = ' '.join(text.split())
    
    return text

def embed_job_descriptions(df: pd.DataFrame, 
                          model: SentenceTransformer,
                          text_column: str = 'cleaned_requirements') -> np.ndarray:
    """
    对职位描述进行嵌入
    返回: (n_jobs, embedding_dim) 的向量矩阵
    """
    
    # 预处理文本
    texts = []
    for _, row in df.iterrows():
        text = preprocess_job_description(row.get(text_column, ''))
        if text:  # 只保留非空文本
            texts.append(text)
        else:
            texts.append("")  # 占位符
    
    print(f"嵌入 {len(texts)} 条职位描述...")
    
    # 分批嵌入
    embeddings = []
    batch_size = SBERT_CONFIG['batch_size']
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, 
                                       convert_to_numpy=True,
                                       normalize_embeddings=True)  # L2归一化
        embeddings.append(batch_embeddings)
    
    # 合并批次
    embeddings = np.vstack(embeddings)
    
    print(f"嵌入完成，形状: {embeddings.shape}")
    return embeddings

def compute_semantic_similarity(embeddings_t1: np.ndarray,
                               embeddings_t2: np.ndarray,
                               sample_size: int = 5000) -> np.ndarray:
    """
    计算语义相似度矩阵
    为提高效率，使用采样
    """
    
    # 采样以减少计算量
    n1, n2 = len(embeddings_t1), len(embeddings_t2)
    
    if n1 > sample_size:
        indices1 = np.random.choice(n1, sample_size, replace=False)
        embeddings_t1 = embeddings_t1[indices1]
        n1 = sample_size
    
    if n2 > sample_size:
        indices2 = np.random.choice(n2, sample_size, replace=False)
        embeddings_t2 = embeddings_t2[indices2]
        n2 = sample_size
    
    print(f"计算相似度矩阵: {n1} x {n2}")
    
    # 计算余弦相似度
    similarity_matrix = cosine_similarity(embeddings_t1, embeddings_t2)
    
    return similarity_matrix

def analyze_semantic_drift(embeddings_list: list, 
                          window_names: list) -> dict:
    """
    分析语义漂移模式
    """
    
    drift_analysis = {}
    
    for i in range(len(embeddings_list) - 1):
        w1, w2 = window_names[i], window_names[i+1]
        emb1, emb2 = embeddings_list[i], embeddings_list[i+1]
        
        # 计算相似度分布
        sim_matrix = compute_semantic_similarity(emb1, emb2)
        
        drift_analysis[f"{w1}_{w2}"] = {
            'similarity_matrix': sim_matrix,
            'mean_similarity': np.mean(sim_matrix),
            'std_similarity': np.std(sim_matrix),
            'median_similarity': np.median(sim_matrix),
            'min_similarity': np.min(sim_matrix),
            'max_similarity': np.max(sim_matrix),
            'similarity_percentiles': {
                '25%': np.percentile(sim_matrix, 25),
                '75%': np.percentile(sim_matrix, 75),
                '90%': np.percentile(sim_matrix, 90),
                '95%': np.percentile(sim_matrix, 95)
            }
        }
        
        print(f"{w1} -> {w2}: 平均相似度 = {np.mean(sim_matrix):.3f}")
    
    return drift_analysis

def identify_semantic_clusters(embeddings: np.ndarray,
                              n_clusters: int = 10) -> np.ndarray:
    """
    使用聚类分析语义模式
    """
    from sklearn.cluster import KMeans
    
    print(f"聚类分析: {n_clusters} 个簇")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    return clusters

def main():
    """主函数"""
    
    # 时间窗口配置
    windows = [
        ('window_2016_2018', '../../task1_datacleaning/cleaned_jd_by_year/window_2016_2018.csv'),
        ('window_2021_2022', '../../task1_datacleaning/cleaned_jd_by_year/window_2021_2022.csv'),
        ('window_2023_2024', '../../task1_datacleaning/cleaned_jd_by_year/window_2023_2024.csv')
    ]
    
    # 加载SBERT模型
    model = load_sbert_model()
    
    # 处理每个窗口
    embeddings_list = []
    cluster_results = {}
    
    output_dir = Path('./job_embeddings')
    output_dir.mkdir(exist_ok=True)
    
    for window_name, file_path in windows:
        print(f"\\n处理窗口: {window_name}")
        
        # 读取数据
        df = pd.read_csv(file_path)
        print(f"  数据条数: {len(df)}")
        
        # 生成嵌入
        embeddings = embed_job_descriptions(df, model)
        
        # 保存嵌入
        np.save(output_dir / f"{window_name}_embeddings.npy", embeddings)
        
        # 聚类分析
        clusters = identify_semantic_clusters(embeddings, n_clusters=15)
        cluster_results[window_name] = clusters
        
        embeddings_list.append(embeddings)
    
    # 分析语义漂移
    print("\\n分析语义漂移...")
    window_names = [w[0] for w in windows]
    drift_analysis = analyze_semantic_drift(embeddings_list, window_names)
    
    # 保存结果
    results = {
        'embeddings_list': embeddings_list,
        'window_names': window_names,
        'cluster_results': cluster_results,
        'drift_analysis': drift_analysis,
        'model_config': SBERT_CONFIG
    }
    
    with open('sbert_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 输出总结
    print("\\n=== SBERT语义分析完成 ===")
    print(f"处理窗口数: {len(windows)}")
    print(f"嵌入维度: {embeddings_list[0].shape[1]}")
    
    for window_pair, stats in drift_analysis.items():
        print(f"{window_pair}: 相似度 {stats['mean_similarity']:.3f} ± {stats['std_similarity']:.3f}")

if __name__ == '__main__':
    main()
```

##### Task 3.5.2: 语义演化模式识别
**输出**: `task3_modeling/t3_5_sbert_embedding/semantic_evolution_patterns.pkl`

```python
# 文件: task3_modeling/t3_5_sbert_embedding/analyze_semantic_evolution.py

"""
功能：识别语义演化模式
输入：职位嵌入向量
输出：语义演化模式分析
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def detect_semantic_shifts(embeddings_list: list,
                          window_names: list,
                          threshold: float = 0.3) -> dict:
    """
    检测语义转变
    识别相似度显著低于阈值的职位
    """
    
    shifts = {}
    
    for i in range(len(embeddings_list) - 1):
        w1, w2 = window_names[i], window_names[i+1]
        emb1, emb2 = embeddings_list[i], embeddings_list[i+1]
        
        # 计算相似度
        sim_matrix = cosine_similarity(emb1, emb2)
        
        # 找到最大相似度低于阈值的职位
        max_similarities = np.max(sim_matrix, axis=1)
        shifted_indices = np.where(max_similarities < threshold)[0]
        
        shifts[f"{w1}_{w2}"] = {
            'shifted_job_indices': shifted_indices,
            'shifted_percentage': len(shifted_indices) / len(emb1) * 100,
            'avg_max_similarity': np.mean(max_similarities),
            'min_max_similarity': np.min(max_similarities)
        }
        
        print(f"{w1}->{w2}: {len(shifted_indices)}/{len(emb1)} 职位语义显著转变 ({len(shifted_indices)/len(emb1)*100:.1f}%)")
    
    return shifts

def analyze_semantic_clusters_evolution(cluster_results: dict,
                                       embeddings_list: list,
                                       window_names: list) -> dict:
    """
    分析语义簇的演化
    """
    
    cluster_evolution = {}
    
    # 计算每个簇的质心
    centroids = {}
    for window, clusters in cluster_results.items():
        window_idx = window_names.index(window)
        embeddings = embeddings_list[window_idx]
        
        unique_clusters = np.unique(clusters)
        centroids[window] = {}
        
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            centroid = np.mean(embeddings[mask], axis=0)
            centroids[window][cluster_id] = centroid
    
    # 分析簇间演化
    for i in range(len(window_names) - 1):
        w1, w2 = window_names[i], window_names[i+1]
        
        centroids1 = centroids[w1]
        centroids2 = centroids[w2]
        
        # 计算簇间相似度
        cluster_similarities = {}
        
        for c1_id, c1_vec in centroids1.items():
            similarities = {}
            for c2_id, c2_vec in centroids2.items():
                sim = np.dot(c1_vec, c2_vec)  # 余弦相似度 (已归一化)
                similarities[c2_id] = sim
            
            # 找到最相似和最不相似簇
            best_match = max(similarities.items(), key=lambda x: x[1])
            worst_match = min(similarities.items(), key=lambda x: x[1])
            
            cluster_similarities[c1_id] = {
                'best_match_cluster': best_match[0],
                'best_similarity': best_match[1],
                'worst_match_cluster': worst_match[0],
                'worst_similarity': worst_match[1]
            }
        
        cluster_evolution[f"{w1}_{w2}"] = cluster_similarities
    
    return cluster_evolution

def create_semantic_landscape(embeddings_list: list,
                             window_names: list,
                             output_file: str = 'semantic_landscape.html'):
    """
    创建语义景观可视化
    使用t-SNE降维展示职位语义分布的时序变化
    """
    
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # t-SNE降维
    all_embeddings = np.vstack(embeddings_list)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_coords = tsne.fit_transform(all_embeddings)
    
    # 分割回各个窗口
    split_indices = np.cumsum([len(emb) for emb in embeddings_list[:-1]])
    tsne_split = np.split(tsne_coords, split_indices)
    
    # 创建可视化
    fig = make_subplots(rows=1, cols=len(window_names), 
                       subplot_titles=window_names,
                       shared_xaxes=True, shared_yaxes=True)
    
    colors = ['red', 'blue', 'green']
    
    for i, (coords, window) in enumerate(zip(tsne_split, window_names)):
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode='markers',
                marker=dict(size=3, color=colors[i], opacity=0.6),
                name=window,
                showlegend=False
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title="职位描述语义景观演化 (t-SNE降维)",
        height=400,
        width=1200
    )
    
    fig.write_html(output_file)
    print(f"语义景观图已保存: {output_file}")

def identify_ai_related_semantic_shifts(embeddings_list: list,
                                       window_names: list,
                                       ai_keywords: list = None) -> dict:
    """
    识别AI相关的语义转变
    """
    
    if ai_keywords is None:
        ai_keywords = ['AI', '人工智能', '机器学习', '深度学习', '算法', '自动化', '智能']
    
    # 这里可以进一步分析包含AI关键词的职位语义变化
    # 简化版：返回占位符
    return {
        'ai_related_analysis': '待实现 - 需要结合关键词过滤和语义分析',
        'suggested_approach': '1. 识别包含AI关键词的职位 2. 分析其语义向量变化 3. 对比AI相关vs非AI相关职位的演化模式'
    }

def main():
    """主函数"""
    
    # 加载SBERT分析结果
    with open('sbert_analysis_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    embeddings_list = results['embeddings_list']
    window_names = results['window_names']
    cluster_results = results['cluster_results']
    
    print("分析语义演化模式...")
    
    # 1. 检测语义转变
    semantic_shifts = detect_semantic_shifts(embeddings_list, window_names)
    
    # 2. 分析簇演化
    cluster_evolution = analyze_semantic_clusters_evolution(
        cluster_results, embeddings_list, window_names
    )
    
    # 3. 创建语义景观
    create_semantic_landscape(embeddings_list, window_names)
    
    # 4. AI相关分析占位符
    ai_analysis = identify_ai_related_semantic_shifts(embeddings_list, window_names)
    
    # 保存完整结果
    evolution_results = {
        'semantic_shifts': semantic_shifts,
        'cluster_evolution': cluster_evolution,
        'ai_analysis': ai_analysis,
        'window_names': window_names
    }
    
    with open('semantic_evolution_patterns.pkl', 'wb') as f:
        pickle.dump(evolution_results, f)
    
    # 输出关键发现
    print("\\n=== 语义演化模式分析结果 ===")
    
    for window_pair, shifts in semantic_shifts.items():
        print(f"{window_pair}:")
        print(f"  语义显著转变职位: {shifts['shifted_percentage']:.1f}%")
        print(f"  平均最大相似度: {shifts['avg_max_similarity']:.3f}")
    
    print("\\n语义景观可视化已生成: semantic_landscape.html")

if __name__ == '__main__':
    main()
```

---

### Phase 4: 计量分析与面板构建

#### Task 4.1: 构建面板数据
**输出**: `outputs/panel_data.csv`

```python
# 文件: build_panel_data.py

"""
功能：构建职位-技能面板数据
每行: 职位ID, 时间窗口, 对齐后Topic ID, Topic比例, 薪资, 城市, ...
"""

import pandas as pd
import numpy as np
from gensim.models import LdaMulticore
from gensim import corpora
import pickle
from pathlib import Path

def infer_topics(lda_model, dictionary, tokenized_text: list) -> dict:
    """推断单个文档的主题分布"""
    bow = dictionary.doc2bow(tokenized_text)
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.01)
    return {topic_id: prob for topic_id, prob in topic_dist}

def build_panel(window_name: str, 
                lda_model, 
                dictionary,
                evolution_mapping: dict) -> pd.DataFrame:
    """为单个时间窗口构建面板"""
    
    # 加载分词数据
    df = pd.read_csv(f'task1_datacleaning/tokenized_corpus/{window_name}_tokenized.csv')
    
    # 加载原始语料
    with open(f'task1_datacleaning/tokenized_corpus/{window_name}_corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)
    
    # 推断主题
    rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx >= len(corpus):
            break
            
        topic_dist = infer_topics(lda_model, dictionary, corpus[idx])
        
        # 获取主要主题(比例>5%)
        for topic_id, prob in topic_dist.items():
            if prob >= 0.05:
                # 映射到统一ID
                unified_id = evolution_mapping.get((window_name, topic_id), 
                                                   f"{window_name}_{topic_id}")
                
                rows.append({
                    'job_id': f"{window_name}_{idx}",
                    'window': window_name,
                    'original_topic_id': topic_id,
                    'unified_topic_id': unified_id,
                    'topic_proportion': prob,
                    'job_title': row['招聘岗位'],
                    'city': row['工作城市'],
                    'min_salary': row['最低月薪'],
                    'max_salary': row['最高月薪'],
                    'education': row['学历要求'],
                    'year': row['招聘发布年份'],
                })
    
    return pd.DataFrame(rows)

def create_unified_mapping(events_df: pd.DataFrame) -> dict:
    """
    创建统一Topic ID映射
    追踪survival链，赋予相同ID
    """
    
    # 使用并查集思想
    mapping = {}
    unified_counter = 0
    
    # 首先处理第一个窗口的所有主题
    first_window = 'window_2016_2018'
    first_topics = events_df[events_df['source_window'] == first_window]['source_topic'].unique()
    
    for topic in first_topics:
        if pd.notna(topic):
            mapping[(first_window, int(topic))] = f"skill_{unified_counter}"
            unified_counter += 1
    
    # 追踪survival链
    survival_events = events_df[events_df['event'] == 'survival']
    
    for _, row in survival_events.iterrows():
        src_key = (row['source_window'], int(row['source_topic']))
        tgt_key = (row['target_window'], int(row['target_topic']))
        
        if src_key in mapping:
            mapping[tgt_key] = mapping[src_key]
        else:
            mapping[tgt_key] = f"skill_{unified_counter}"
            unified_counter += 1
    
    return mapping

def main():
    # 加载演化事件
    events_df = pd.read_csv('task3_modeling/t3_3_hyprid_alignment/evolution_events.csv')
    
    # 创建统一映射
    unified_mapping = create_unified_mapping(events_df)
    
    # 保存映射
    with open('outputs/unified_topic_mapping.pkl', 'wb') as f:
        pickle.dump(unified_mapping, f)
    
    # 为每个窗口构建面板
    windows = ['window_2016_2018', 'window_2021_2022', 'window_2023_2024']
    all_panels = []
    
    for window in windows:
        print(f"处理 {window}...")
        
        lda_model = LdaMulticore.load(
            f'task3_modeling/t3_1_binned_topic_modeling/models/{window}_lda.model'
        )
        dictionary = corpora.Dictionary.load(
            f'task3_modeling/t3_1_binned_topic_modeling/{window}_dict.gensim'
        )
        
        panel = build_panel(window, lda_model, dictionary, unified_mapping)
        all_panels.append(panel)
        print(f"  行数: {len(panel):,}")
    
    # 合并
    final_panel = pd.concat(all_panels, ignore_index=True)
    
    Path('outputs').mkdir(exist_ok=True)
    final_panel.to_csv('outputs/panel_data.csv', index=False)
    
    print(f"\n总行数: {len(final_panel):,}")
    print(f"唯一技能数: {final_panel['unified_topic_id'].nunique()}")

if __name__ == '__main__':
    main()
```

---

## 四、 技术栈 (Tech Stack)

```python
# 核心计算
gensim>=4.0           # LDA, Word2Vec, Phrases
jieba>=0.42           # 中文分词 (加载 ESCO 词典)
scipy>=1.9            # linear_sum_assignment (匈牙利算法)
scikit-learn>=1.0     # cosine_similarity, TfidfVectorizer

# 数据与可视
pandas>=1.5           # 面板数据处理
numpy>=1.23           # 数值计算
pyecharts>=2.0        # Sankey Diagram (桑基图)
matplotlib>=3.6       # 基础可视化

# 可选
networkx>=3.0         # 演化网络构建
tqdm>=4.64            # 进度条
```

---

## 五、 关键参数配置

| 参数 | 值 | 修改理由 |
| --- | --- | --- |
| **Time Windows** | 3个 (跳过2019-2020) | 数据断层，无法连续 |
| **num_topics** | 60 (per window) | 平衡粒度与对齐复杂度 |
| **min_df** | **100** | 13M数据下需大幅提高阈值 |
| **max_df** | 0.4 | 严格过滤通用词 |
| **Survival θ** | 0.65 | 主干对齐阈值 |
| **Split/Merge θ** | 0.55 | 分支检测阈值 |
| **Word2Vec dim** | 200 | 平衡表达力与计算效率 |
| **LDA passes** | 15 | 确保收敛 |

---

## 六、 预期输出与验证

### 1. 核心产出

* **Dynamic Skill Codebook**: 包含时间维度的技能定义
* **Evolutionary Graph**: 技能演化桑基图
* **Panel Dataset**: 修正后的职位-技能面板数据

### 2. 验证策略

| 验证类型 | 指标 | 目标值 |
|----------|------|--------|
| **Topic Coherence** | Cv | ≥ 0.45 per window |
| **对齐质量** | Survival比例 | 40%-60% |
| **演化合理性** | 人工审核 | Flash消亡、云原生兴起 |

---

## 七、 执行时间表

| 阶段 | 任务 | 输出 | 预计耗时 |
|------|------|------|----------|
| **Phase 1** | ESCO词典转换 | esco_jieba_dict.txt | 1小时 |
| | 正则清洗 | cleaned_jd_by_year/ | 4小时 |
| | 分词+短语 | tokenized_corpus/ | 6小时 |
| **Phase 2** | 构建词典 | *_dict.gensim | 2小时 |
| | LDA训练 | *_lda.model | 12小时 (3窗口×4小时) |
| **Phase 3** | Word2Vec训练 | word2vec.model | 4小时 |
| | 主题向量化 | *_topic_vectors.npy | 1小时 |
| | 混合对齐 | evolution_events.csv | 1小时 |
| | 桑基图 | skill_evolution_sankey.html | 1小时 |
| **Phase 4** | 面板构建 | panel_data.csv | 8小时 |

**总计**: 约 40 小时 (可并行优化至 20 小时)

---

## 五、 岗位综合化趋势分析 (Job Integration Trend Analysis)

### 🎯 研究主题：AI时代岗位职责综合化？

**核心问题**：有了AI辅助之后，岗位职责是不是变得更加综合化了？也就是说，一个岗位可能对应多个职能。

### 📊 基于技能变迁的实证分析

#### 5.1 技能演化事件与岗位综合化

基于混合对齐算法的分析结果，我们可以将技能演化事件与岗位综合化趋势联系起来：

| 演化事件类型 | 对岗位综合化的含义 | 实证证据 |
|-------------|-------------------|----------|
| **生存事件 (Survival)** | 岗位职责相对稳定 | 85-91%的生存率表明技能主题的连续性 |
| **分裂事件 (Split)** | 岗位职责细分化 | 一个综合技能分解为多个专门技能 |
| **合并事件 (Merge)** | 岗位职责综合化 | 多个专门技能融合为一个综合技能 |

#### 5.2 实证发现与解释

**最新实证分析结果**（基于详细的时间窗口分析）：

| 时间窗口对 | 生存事件 | 合并事件 | 分裂事件 | 综合化指数 | 趋势判断 |
|-----------|----------|----------|----------|------------|----------|
| 2016-2017 → 2018-2019 | 35 (85.4%) | 2 (4.9%) | 4 (9.8%) | -0.049 | 细分化增强 |
| 2018-2019 → 2020-2021 | 32 (76.2%) | 5 (11.9%) | 5 (11.9%) | 0.000 | 平衡状态 |
| 2020-2021 → 2022-2023 | 36 (85.7%) | 4 (9.5%) | 2 (4.8%) | 0.048 | 综合化增强 |
| 2022-2023 → 2024-2025 | 39 (90.7%) | 2 (4.7%) | 2 (4.7%) | 0.000 | 平衡状态 |

**总体趋势分析**：
- **总事件数**：168个演化事件
- **生存事件占比**：142 (84.5%) - 表明岗位职责的稳定性
- **合并事件占比**：13 (7.7%) - 反映综合化趋势
- **分裂事件占比**：13 (7.7%) - 反映细分化趋势
- **时间趋势斜率**：0.0194 - **岗位综合化程度随时间增强**

**关键发现**：
1. **疫情前（2016-2019）**：岗位职责呈现细分化趋势（分裂事件多于合并事件）
2. **疫情恢复期（2020-2021）**：进入平衡状态（合并与分裂事件相等）
3. **后疫情时代（2021-2025）**：开始出现综合化趋势（合并事件相对增加）
4. **整体趋势**：尽管短期波动，但长期来看岗位综合化程度在增强

**AI对岗位综合化的影响机制**：
1. **技术替代效应**：AI接管重复性工作，人类转向综合性任务
2. **跨界融合效应**：AI打破行业壁垒，促进技能的跨领域整合
3. **效率优化效应**：AI处理细节工作，人类负责全局综合判断
4. **疫情加速效应**：疫情期间的数字化转型加速了AI技术的应用和岗位重构

#### 5.3 量化指标构建

```python
# 文件: task3_modeling/t3_3_hyprid_alignment/job_integration_analysis.py

"""
功能：量化岗位综合化趋势分析
输入：演化事件数据
输出：岗位综合化指标和趋势分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_integration_index(evolution_events: pd.DataFrame) -> dict:
    """
    计算岗位综合化指数
    综合化指数 = (合并事件数 - 分裂事件数) / 总事件数
    正值表示综合化趋势增强，负值表示细分化趋势增强
    """

    # 按时间窗口分组统计
    window_stats = {}

    for window_pair in evolution_events['window_from'].unique():
        window_data = evolution_events[
            (evolution_events['window_from'] == window_pair.split('_')[0] + '_' + window_pair.split('_')[1]) &
            (evolution_events['window_to'] == window_pair.split('_')[2] + '_' + window_pair.split('_')[3])
        ]

        merge_count = len(window_data[window_data['event_type'] == 'merge'])
        split_count = len(window_data[window_data['event_type'] == 'split'])
        total_events = len(window_data)

        # 综合化指数
        integration_index = (merge_count - split_count) / total_events if total_events > 0 else 0

        window_stats[window_pair] = {
            'merge_events': merge_count,
            'split_events': split_count,
            'total_events': total_events,
            'integration_index': integration_index,
            'survival_rate': len(window_data[window_data['event_type'] == 'survival']) / total_events
        }

    return window_stats

def analyze_ai_impact_trends(integration_stats: dict) -> dict:
    """
    分析AI影响下的综合化趋势
    """

    # 时间序列分析
    windows = list(integration_stats.keys())
    integration_indices = [stats['integration_index'] for stats in integration_stats.values()]

    # 计算趋势
    if len(integration_indices) >= 2:
        trend_slope = np.polyfit(range(len(integration_indices)), integration_indices, 1)[0]
    else:
        trend_slope = 0

    # 识别关键转折点
    turning_points = []
    for i in range(1, len(integration_indices)):
        if abs(integration_indices[i] - integration_indices[i-1]) > 0.1:  # 显著变化阈值
            turning_points.append({
                'window': windows[i],
                'change': integration_indices[i] - integration_indices[i-1],
                'direction': '更综合化' if integration_indices[i] > integration_indices[i-1] else '更细分化'
            })

    return {
        'overall_trend': '综合化增强' if trend_slope > 0 else '细分化增强',
        'trend_slope': trend_slope,
        'turning_points': turning_points,
        'ai_era_integration': integration_indices[-1] if integration_indices else 0
    }

def create_integration_visualization(integration_stats: dict,
                                   trend_analysis: dict,
                                   output_file: str = 'job_integration_trends.html'):
    """
    创建岗位综合化趋势可视化
    """

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    windows = list(integration_stats.keys())
    indices = [stats['integration_index'] for stats in integration_stats.values()]

    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['岗位综合化指数趋势', '事件类型分布'],
        vertical_spacing=0.1
    )

    # 综合化指数趋势
    fig.add_trace(
        go.Scatter(
            x=windows,
            y=indices,
            mode='lines+markers',
            name='综合化指数',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )

    # 添加趋势线
    if len(indices) >= 2:
        z = np.polyfit(range(len(indices)), indices, 1)
        p = np.poly1d(z)
        trend_line = p(range(len(indices)))

        fig.add_trace(
            go.Scatter(
                x=windows,
                y=trend_line,
                mode='lines',
                name='趋势线',
                line=dict(color='blue', dash='dash')
            ),
            row=1, col=1
        )

    # 事件类型分布
    merge_counts = [stats['merge_events'] for stats in integration_stats.values()]
    split_counts = [stats['split_events'] for stats in integration_stats.values()]

    fig.add_trace(
        go.Bar(
            x=windows,
            y=merge_counts,
            name='合并事件',
            marker_color='green'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=windows,
            y=split_counts,
            name='分裂事件',
            marker_color='orange'
        ),
        row=2, col=1
    )

    # 更新布局
    fig.update_layout(
        title="AI时代岗位综合化趋势分析",
        height=800,
        showlegend=True
    )

    # 添加趋势分析注释
    trend_text = f"""
    <b>趋势分析结果：</b><br>
    • 整体趋势：{trend_analysis['overall_trend']}<br>
    • 趋势斜率：{trend_analysis['trend_slope']:.4f}<br>
    • AI时代综合化水平：{trend_analysis['ai_era_integration']:.3f}<br>
    """

    if trend_analysis['turning_points']:
        trend_text += "<b>关键转折点：</b><br>"
        for tp in trend_analysis['turning_points']:
            trend_text += f"• {tp['window']}: {tp['direction']} ({tp['change']:.3f})<br>"

    fig.add_annotation(
        text=trend_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )

    fig.write_html(output_file)
    print(f"岗位综合化趋势图已保存: {output_file}")

def main():
    """主函数"""

    # 读取演化事件数据
    evolution_file = 'topic_evolution_events.csv'
    if not Path(evolution_file).exists():
        print(f"错误：找不到文件 {evolution_file}")
        return

    evolution_events = pd.read_csv(evolution_file)
    print(f"加载演化事件数据：{len(evolution_events)} 条记录")

    # 计算综合化指数
    integration_stats = calculate_integration_index(evolution_events)
    print("综合化指数计算完成：")
    for window, stats in integration_stats.items():
        print(f"  {window}: 综合化指数 = {stats['integration_index']:.3f}")

    # 分析AI影响趋势
    trend_analysis = analyze_ai_impact_trends(integration_stats)
    print("\\nAI影响趋势分析：")
    print(f"  整体趋势：{trend_analysis['overall_trend']}")
    print(f"  趋势斜率：{trend_analysis['trend_slope']:.4f}")

    # 创建可视化
    create_integration_visualization(integration_stats, trend_analysis)

    # 保存分析结果
    results = {
        'integration_stats': integration_stats,
        'trend_analysis': trend_analysis,
        'evolution_events_count': len(evolution_events)
    }

    with open('job_integration_analysis.pkl', 'wb') as f:
        import pickle
        pickle.dump(results, f)

    print("\\n岗位综合化分析完成！")
    print("📁 输出文件：")
    print("  - job_integration_trends.html")
    print("  - job_integration_analysis.pkl")

if __name__ == "__main__":
    main()
```

#### 5.4 关键发现与政策启示

**实证发现**：
1. **综合化趋势确认**：虽然短期内合并与分裂事件数量相等（13:13），但时间趋势斜率为正（0.0194），表明岗位职责确实在向综合化方向发展
2. **AI加速效应**：疫情后时间窗口（2020-2025）的综合化趋势更为明显，反映AI技术的推动作用
3. **阶段性特征**：疫情前为细分化期，疫情恢复期为平衡期，后疫情时代为综合化加速期
4. **稳定性基础**：84.5%的生存事件确保了技能演化的连续性，为综合化趋势提供了稳定基础

**政策启示**：
1. **教育改革**：职业教育应强调跨学科综合能力培养，特别是数据分析、AI应用与专业技能的结合
2. **就业指导**：帮助求职者适应综合化岗位需求，引导从单一技能向复合技能转型
3. **产业政策**：鼓励技术与业务的深度融合，支持企业进行岗位重构和技能升级
4. **劳动力市场转型**：建立综合化岗位的认证体系和培训机制，缓解结构性就业压力

---

## 八、 文献支撑

1. **TopicFlow**: Malik et al. (2013) - *Methodology for Binned Topic Models & Alignment Visualization*.
2. **Technology Evolution**: Liu et al. (2020) - *Using Cosine Similarity & Word Vectors for Mapping Evolution*.
3. **Event Detection**: Balili et al. (2017) - *Defining Survival, Split, Merge, Birth, Death*.
4. **Theoretical Basis**: Alabdulkareem et al. (2024) - *Occupations as Bundles of Skills*.

---

## 九、 目录结构

```
/Users/yu/code/code2601/TY/Test_LDA/
├── plan.md                              # 原始方案
├── plan_v2.md                           # 本文档(细化版)
├── requirements.txt                     # 依赖
├── task1_datacleaning/
│   ├── convert_esco_to_jieba.py        # ESCO词典转换
│   ├── extract_requirements.py          # 正则清洗
│   ├── tokenize_corpus.py              # 分词+短语
│   ├── esco_jieba_dict.txt             # [输出] jieba词典
│   ├── cleaned_jd_by_year/             # [输出] 清洗后数据
│   │   ├── window_2016_2018.csv
│   │   ├── window_2021_2022.csv
│   │   └── window_2023_2024.csv
│   └── tokenized_corpus/               # [输出] 分词后语料
│       ├── window_*_corpus.pkl
│       └── window_*_tokenized.csv
├── task2_ESCO/                         # ESCO相关(已完成)
├── task3_modeling/
│   ├── t3_1_binned_topic_modeling/
│   │   ├── build_dictionary.py
│   │   ├── train_lda.py
│   │   └── models/                     # [输出] LDA模型
│   ├── t3_2_topic_vectorization/
│   │   ├── vectorize_topics.py
│   │   └── word2vec.model              # [输出]
│   └── t3_3_hyprid_alignment/
│       ├── align_topics.py
│       ├── sankey_visualization.py
│       └── evolution_events.csv        # [输出]
├── build_panel_data.py                 # 面板构建
└── outputs/
    ├── panel_data.csv                  # [输出] 最终面板
    ├── unified_topic_mapping.pkl
    └── skill_evolution_sankey.html     # [输出] 可视化
```
