# 📊 TY项目进度报告 (2026年2月3日)

## 🎯 项目目标
**研究问题**: 有了AI辅助之后，岗位职责是不是变得更加综合化了？

## ✅ 已完成工作

### 1️⃣ 数据准备 (100% 完成)
- ✅ **原始数据**: 13,492,246 条招聘广告 (all_in_one.csv)
- ✅ **时间跨度**: 2015-2025 (11年)
- ✅ **数据清洗管道**: 已建立 (run_pipeline.py)
- ✅ **ESCO词典**: 已准备 (data/esco/chinese/, jieba_dict/)

### 2️⃣ LDA 主题建模 (40% 完成)
**已训练的时间窗口**:
- ✅ **Window 2016-2018**: LDA模型已训练 (60 topics)
- ✅ **Window 2021-2022**: LDA模型已训练 (60 topics)
- ⚠️ **Window 2023-2024**: 仅有分词数据，模型待训练

**缺失的时间窗口**:
- ❌ Window 2015 (单年)
- ❌ Window 2019-2020
- ❌ Window 2025 (最新数据)

**技术参数**:
```python
num_topics: 60
passes: 15
alpha: 'symmetric'
workers: 4
coherence: c_v (目标 > 0.40)
```

### 3️⃣ SBERT 语义分析 (0% 完成)
- ❌ `src/sbert/` 文件夹为空
- ⚠️ 可复用 ICM_F 项目的代码框架

## 🚧 当前核心问题

### 问题1: 主题对齐 (Critical)
**现状**: 每个时间窗口独立训练 60 个主题，**无法跨期对比**

**解决方案**:
1. **锚点主题法** (推荐) - 参考 Autor et al. (2021, AER)
   - 使用 2016-2018 的 top 20 关键词作为锚点
   - 在后续窗口中用锚点词初始化主题

2. **动态主题模型** (更学术) - Blei & Lafferty (2006)
   - 使用 `gensim.ldaseqmodel`
   - 一次性处理所有年份，自动捕捉主题演化

3. **后验对齐** (折中方案)
   - 训练完成后用余弦相似度匹配主题
   - 工具: `sklearn.metrics.pairwise_distances`

### 问题2: 语义漂移
**现象**: "AI"、"云计算"等词在 2015 和 2025 年含义不同

**解决方案**:
- 保持当前的时间窗口分割策略 ✅ (正确)
- 在对齐时允许主题 **部分重叠** 而非完全匹配

### 问题3: 综合化指标构建
**待设计指标**:
- **LDA 侧**: 主题熵 (Topic Entropy)
- **SBERT 侧**: 技能熵 (Skill Entropy)

## 📋 下一步行动计划

### 🔥 本周优先级（已准备好代码工具！）

#### **周一-周二: 补全 LDA 训练**
```bash
# 1. 训练缺失的时间窗口（需要您先准备分词数据）
cd /Users/yu/code/code2601/TY/src/lda
python train_lda.py  # 自动处理所有已分词窗口

# 2. 检查模型质量
ls -lh /Users/yu/code/code2601/TY/output/lda/models/
```

#### **周三: 实现主题对齐 ✅ (代码已创建)**
```bash
# 运行主题对齐脚本
cd /Users/yu/code/code2601/TY/src/lda
python align_topics.py

# 输出:
# - output/lda/alignment/alignment_matrix.csv
# - output/lda/alignment/alignment_quality.csv
# - output/lda/alignment/topic_labels_template.csv (需人工标注)
```

#### **周四: 构建 SBERT 管道 ✅ (代码已创建)**
```bash
# 运行SBERT技能指纹计算
cd /Users/yu/code/code2601/TY/src/sbert
python compute_skill_fingerprint.py

# 输出:
# - output/sbert/fingerprint_evolution_TY.csv
# - output/sbert/detail_window_*.npy (每个窗口的详细矩阵)
```

#### **周五: 综合化指标 + 可视化 ✅ (代码已创建)**
```bash
# 方案1: 一键运行全流程
cd /Users/yu/code/code2601/TY/scripts
python run_full_analysis.py

# 方案2: 分步运行
# Step 1: 计算综合化指标
python ../src/analysis/build_comprehensiveness_index.py

# Step 2: 生成可视化报告
python ../src/analysis/visualize_results.py
```

### 📊 Week 2: 指标构建与验证

#### **构建综合化指数**
```python
# LDA 综合化指数
def lda_diversification_score(doc_topic_distribution):
    """
    输入: 文档-主题分布 (N_docs, 60)
    输出: 标准化香农熵 [0, 1]
    """
    prob = doc_topic_distribution
    entropy = -np.sum(prob * np.log(prob + 1e-10), axis=1)
    return entropy / np.log(60)  # 标准化

# SBERT 综合化指数
def sbert_diversification_score(skill_similarity_matrix):
    """
    输入: 文档-技能相似度 (N_docs, 20)
    输出: 标准化香农熵 [0, 1]
    """
    prob = skill_similarity_matrix / skill_similarity_matrix.sum(axis=1, keepdims=True)
    entropy = -np.sum(prob * np.log(prob + 1e-10), axis=1)
    return entropy / np.log(20)
```

#### **稳健性检验**
- 计算 LDA 和 SBERT 综合化指数的年度平均值
- Pearson 相关系数目标: **r > 0.70**
- 可视化: 双轴时序图

## 💡 方法论建议

### 论文结构设计
```
Section 3: Methodology
  3.1 LDA Topic Modeling (主分析)
      - ESCO 词典构建
      - 时间窗口设计
      - 主题对齐算法
  
  3.2 Job Comprehensiveness Index
      - 基于主题熵的定义
      - 与 Herfindahl Index 的关系
  
  3.3 Robustness Check (稳健性检验)
      - SBERT 语义相似度方法
      - 双方法相关性验证
```

### 预期贡献点
1. **方法创新**: 多时期 LDA + 锚点对齐
2. **测量创新**: AI 时代的岗位综合化指数
3. **实证发现**: AI 工具对技能需求的影响

## 📚 参考文献（已搜索）

### LDA 时间序列对齐
- Blei & Lafferty (2006) "Dynamic Topic Models", ICML
- Sato et al. (2019) "Anchor Topic Alignment", KDD
- Autor et al. (2021) "The Fall of the Labor Share...", AER

### 技能分析
- Deming & Kahn (2018) "Skill Requirements across Firms...", QJE
- Atalay et al. (2020) "New Technologies and the Labor Market", JPE

## 🎓 给导师的汇报材料

### 当前进展
1. ✅ 已有 1349 万条清洗后的 JD 数据
2. ✅ LDA 框架已搭建（部分窗口已训练）
3. ✅ ESCO 专业词典已准备
4. ⚠️ 主题对齐方案待实施

### 下周交付物
1. 完整的 LDA 训练结果 (所有时间窗口)
2. 主题对齐矩阵可视化
3. LDA vs SBERT 相关性初步结果

### 预期时间线
- **2周后**: 完成方法论部分初稿
- **4周后**: 完成描述性统计 + 稳健性检验
- **6周后**: 完成全文初稿

---

## 📁 项目文件结构
```
TY/
├── dataset/
│   └── all_in_one.csv              [13.5M 条原始数据 ✅]
├── data/
│   ├── esco/                       [ESCO 词典 ✅]
│   └── processed/
│       ├── cleaned/                [清洗后数据]
│       ├── tokenized/              [分词结果 ✅]
│       │   ├── window_2016_2018_tokenized.csv
│       │   ├── window_2021_2022_tokenized.csv
│       │   └── window_2023_2024_tokenized.csv
│       └── windows/                [时间窗口分割]
├── src/
│   ├── lda/
│   │   ├── train_lda.py           [LDA 训练 ✅]
│   │   ├── align_topics.py        [✅ 新建！主题对齐]
│   │   └── build_codebook.py      [主题词典 ✅]
│   ├── sbert/                      
│   │   └── compute_skill_fingerprint.py [✅ 新建！技能指纹]
│   └── analysis/                   
│       ├── build_comprehensiveness_index.py [✅ 新建！综合化指标]
│       └── visualize_results.py   [✅ 新建！可视化]
├── scripts/
│   ├── run_pipeline.py            [数据清洗 ✅]
│   └── run_full_analysis.py       [✅ 新建！一键运行]
└── output/
    ├── lda/
    │   ├── models/                 [部分模型已训练 ⚠️]
    │   │   ├── window_2016_2018_lda.model ✅
    │   │   └── window_2021_2022_lda.model ✅
    │   ├── alignment/              [✅ 待生成：对齐结果]
    │   └── analysis/               [✅ 待生成：综合化指标]
    ├── sbert/                      [✅ 待生成：技能指纹]
    └── reports/
        └── figures/                [✅ 待生成：可视化图表]
```

## 🚀 立即执行命令

```bash
# 1. 检查当前训练进度
ls -lh /Users/yu/code/code2601/TY/output/lda/models/

# 2. 查看已有主题（验证质量）
head -50 /Users/yu/code/code2601/TY/output/lda/models/window_2016_2018_topics.txt

# 3. 准备下一步训练
cd /Users/yu/code/code2601/TY/src/lda
python train_lda.py  # 检查脚本是否支持命令行参数
```

---
**更新时间**: 2026-02-03  
**负责人**: Yu  
**导师要求**: 优先完成 LDA + 主题对齐
