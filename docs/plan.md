
# 动态LDA职业技能演化识别方案 (Dynamic Skill Evolution Scheme)

## 一、 理论框架 (Theoretical Framework)

**核心构建**：

1. **静态视角**：Occupations as bundles of skills (Alabdulkareem et al., 2024)。
2. **动态视角**：技能的**语义漂移 (Semantic Drift)**。
* *假设*：20年前的“诺基亚”与今天的“AI”处于不同的语义空间，强行合并会导致跨期不可比。
* *对策*：采用 **Binned Topic Models (Malik et al.)**，即在独立时间窗口内识别技能，再通过向量空间进行跨期对齐。



**模型优势**：

* **抗噪性**：通过领域词典（ESCO）和短语识别修复破碎语义。
* **演化追踪**：不仅能看到技能占比变化，还能识别技能的**分裂 (Split)**、**融合 (Merge)** 和 **消亡 (Death)**。

---

## 二、 方法论设计 (Methodology)

### 1. 数据结构与清洗

* **原始数据**：`all_in_one1.csv` (13M records, 清洗后)
* **关键过滤**：
* **正则提取**：仅保留“任职要求/技能要求”字段，剔除公司介绍与福利（防止“五险一金”成为Topic）。
* **去重**：基于 SimHash/MinHash 剔除重复发布的JD。



### 2. 增强预处理策略 (针对“语义之殇”)

**目标**：防止专业词汇被切碎，防止高频通用词污染主题。

```python
# 语义修复流程
1. 加载领域词典：使用汉化版 ESCO-Skill-Extractor
   -> 路径: /Users/yu/code/code2601/TY/Test_ESCO
   -> 作用: 强制保留 "React.js", "C++" 等专有名词，防止被 jieba 切分。

2. 短语发现 (Phrase Detection):
   -> 使用 gensim.models.Phrases 识别 Bigram/Trigram
   -> 效果: 将 "深度" + "学习" 合并为 "深度_学习"，"Data" + "Mining" -> "Data_Mining"。

3. 动态停用词过滤:
   -> 不仅使用通用停用词表。
   -> 计算全语料 TF-IDF，剔除 Document Frequency > 50% 的高频无意义词（如“负责”、“具有”、“相关”）。

```

### 3. 模型策略：分箱训练与向量对齐 (针对“时间之殇”)

#### Step 1: 独立时间片训练 (Binned Topic Modeling)

* **切分**：将数据按  个时间窗口切分（建议 3年为一窗口，如 2010-2012, 2013-2015...）。
* **训练**：在每个窗口独立训练 LDA 模型。
* **输出**：得到序列化的模型集合 。

#### Step 2: 主题向量化 (Topic Vectorization)

* **核心**：将概率分布  转换为稠密向量，解决语义对齐问题。
* **公式**：
* **词向量来源**：基于清洗后语料自训练 Word2Vec，或使用 ESCO 预训练向量。

#### Step 3: 混合策略对齐 (Hybrid Alignment)

* **计算**：构建  与  时刻主题间的余弦相似度矩阵 。
* **算法**：
1. **主干识别**：使用 **匈牙利算法 (Hungarian Algorithm)** 确定全局最优的 1-to-1 存活路径 (Survival)。
2. **分支识别**：使用 **贪婪阈值法 (Greedy Threshold)** 扫描剩余高相似度连线，识别 **分裂 (Split)** 和 **融合 (Merge)**。



---

## 三、 实施步骤 (Implementation Pipeline)

### Phase 1: 预处理与词典构建 (Week 1)

1. **词典集成**：读取 `/Users/yu/code/code2601/TY/Test_ESCO`，格式化为 jieba `user_dict`。
2. **短语建模**：抽取 50万 条样本训练 `Phrases` 模型，固化词组搭配。
3. **文本清洗**：执行正则清洗和分词，生成序列化的 Tokenized Corpus。

### Phase 2: 分箱 LDA 训练 (Week 1-2)

1. **时间切片**：按 `[2005-2009], [2010-2014], [2015-2019], [2020-2024]` (示例) 切分数据。
2. **并行训练**：对每个时间片训练 LDA ()。
3. **质量控制**：检查每个时间片的 Top Topic，确保无“噪音主题”。

### Phase 3: 演化对齐与图谱构建 (Week 3)

1. **向量化**：加载训练好的 Word2Vec，计算所有 Topic 的向量 。
2. **执行对齐算法**：
* 计算 Cosine Similarity Matrix。
* 运行 Hybrid Alignment (Hungarian + Threshold)。
* 输出演化事件表：`{Year: 2010, Source: Topic_5, Target: Topic_12, Event: "Split"}`。


3. **Sankey 可视化**：绘制技能演化桑基图 (TopicFlow)。

### Phase 4: 计量分析 (Week 4)

1. **构建面板数据**：将对齐后的 Topic ID 作为统一变量。
2. **回归分析**：分析技能兴衰（Topic Proportion）与外部变量（薪资、地区GDP）的关系。

---

## 四、 技术栈 (Tech Stack)

```python
# 核心计算
gensim            # LDA, Word2Vec, Phrases
jieba             # 中文分词 (加载 ESCO 词典)
scipy             # linear_sum_assignment (匈牙利算法)
sklearn           # cosine_similarity, TfidfVectorizer

# 数据与可视
pandas            # 面板数据处理
networkx          # 演化网络构建
pyecharts / d3.js # Sankey Diagram (桑基图)

```

---

## 五、 关键参数配置

| 参数 | 值 | 修改理由 |
| --- | --- | --- |
| **Time Windows** | 3-5年 | 捕捉中长期技能演变，避免年度波动噪音 |
| **num_topics** | 50-80 (per window) | 保持粒度适中，方便跨期对齐 |
| **min_df** | **100** | (原为10) 13M数据下需大幅提高阈值，过滤拼写错误 |
| **max_df** | 0.4 | 严格过滤通用词 |
| **Alignment ** | 0.65 (Survival) <br>

<br> 0.75 (Split/Merge) | 设定对齐阈值，低于阈值判定为新生/消亡 |

---

## 六、 预期输出与验证

### 1. 核心产出

* **Dynamic Skill Codebook**: 包含时间维度的技能定义（如：2010年的"大数据" vs 2020年的"大数据"）。
* **Evolutionary Graph**: 技能演化桑基图（展示 "后端" 如何分裂出 "云原生"）。
* **Panel Dataset**: 修正后的职位-技能面板数据。

### 2. 验证策略

* **内部验证**: Topic Coherence (Cv) 在每个时间窗口内是否达标。
* **演化合理性**: 检查检测到的 Split/Merge 事件是否符合行业常识（例如：Flash 是否在 2015 左右消亡？）。

---

## 七、 文献支撑

1. **TopicFlow**: Malik et al. (2013) - *Methodology for Binned Topic Models & Alignment Visualization*.
2. **Technology Evolution**: Liu et al. (2020) - *Using Cosine Similarity & Word Vectors for Mapping Evolution*.
3. **Event Detection**: Balili et al. (2017) - *Defining Survival, Split, Merge, Birth, Death*.
4. **Theoretical Basis**: Alabdulkareem et al. (2024) - *Occupations as Bundles of Skills*.