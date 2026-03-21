# LDA职业技能组合识别

## 理论框架

**Occupations as bundles of skills** (Alabdulkareem et al., 2024)

- 职业不是单一技能，而是技能组合
- LDA主题 = 潜在技能profile
- 职位描述 = 多个技能组合的概率分布

**优势**：
- 捕捉职业边界的模糊性
- 追踪技能需求的时间演变
- 分析地区/行业的异质性

---

## 项目结构

```
Test_LDA/
├── plan.md                          # 详细设计文档
├── lda_skill_extraction.py          # 主程序：LDA训练
├── build_codebook.py                # Codebook构建工具
├── requirements.txt                 # 依赖包
├── README.md                        # 本文件
│
├── skill_keywords.csv               # 输出：技能关键词
├── document_skill_profiles.csv      # 输出：文档-技能分布(θ)
├── skill_review_report.txt          # 输出：审查报告
├── skill_codebook_template.xlsx    # 输出：标注模板
├── annotation_guide.txt             # 输出：标注指南
│
├── lda_model                        # 输出：保存的LDA模型
├── dictionary.dict                  # 输出：词典
└── corpus.mm                        # 输出：语料库
```

---

## 使用流程

### Phase 1: LDA训练 (2-3天)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行LDA训练
python lda_skill_extraction.py
```

**选项**：
- 快速测试：10万条（30分钟）
- 标准分析：50万条（2-3小时）
- 全量分析：1300万条（8-12小时）

**输出**：
- `skill_keywords.csv` - 80个技能组合的Top 30关键词
- `document_skill_profiles.csv` - 每条职位的技能分布(θ)
- `skill_review_report.txt` - 详细审查报告
- `lda_model` - 训练好的模型

---

### Phase 2: 技能标注 (2-3天)

```bash
# 生成标注模板
python build_codebook.py
```

**输出**：
- `skill_codebook_template.xlsx` - Excel标注表
- `annotation_guide.txt` - 标注指南

**标注任务**：

参考LDA关键词，为每个技能组合标注：

| 字段 | 说明 | 示例 |
|------|------|------|
| 技能大类 | Level 1 | 技术研发、市场营销、财务管理 |
| 技能子类 | Level 2 | 软件开发、数据分析、品牌推广 |
| 具体技能 | Level 3 | Python开发、SQL分析、数字营销 |
| 技能描述 | 详细说明 | 使用Python进行后端开发... |
| 典型职位 | 职位示例 | Python工程师、数据分析师 |

**标注原则**：
- 基于关键词，而非主观经验
- 技能优先，而非职位名称
- 允许跨界组合
- 不确定时建立新类目

---

### Phase 3: 面板数据分析 (3-5天)

完成标注后，进行：

1. **时间趋势分析**
   ```python
   # 技能需求如何随时间演变？
   skill_demand_by_year = aggregate(theta, by='year')
   ```

2. **地理异质性分析**
   ```python
   # 一线城市 vs 二三线城市的技能差异？
   skill_by_city = aggregate(theta, by='工作城市')
   ```

3. **行业异质性分析**
   ```python
   # 技术行业 vs 传统行业的技能结构？
   skill_by_industry = aggregate(theta, by='招聘类别')
   ```

4. **回归分析**
   ```python
   # 什么因素驱动技能需求变化？
   model = regression(skill_intensity ~ year + city + industry)
   ```

---

## 技术细节

### LDA参数配置

```python
num_topics = 80              # 技能组合数
passes = 10                  # 训练轮数
alpha = 'auto'              # 文档-主题先验（自动学习）
eta = 'auto'                # 主题-词先验（自动学习）
workers = 4                 # 并行核心数
```

### 预处理策略

```python
分词: jieba + 词性标注（保留名词/动词/形容词）
词典: min_df=10, max_df=0.5, keep_n=50000
停用词: 通用停用词 + 职位描述特定停用词
```

### 输出说明

**skill_keywords.csv**：
```csv
skill_id,keywords,top_10,avg_probability
0,"软件, 开发, python, java, 代码, ...","软件, 开发, python, ...",0.032
```

**document_skill_profiles.csv**：
```csv
document,skill_0,skill_1,skill_2,...,skill_79
"负责软件开发...",0.45,0.12,0.08,...,0.01
```
- 每行 = 一条职位描述
- skill_X列 = 该职位使用技能X的概率（θ）

---

## 评估指标

1. **Perplexity** (困惑度)
   - 越低越好
   - 衡量模型预测能力

2. **Coherence Score** (一致性)
   - 越高越好
   - 衡量主题可解释性

3. **人工评估**
   - 关键词清晰度
   - 技能组合合理性
   - 与实际职业的对应

---

## 文献参考

**Alabdulkareem et al. (2024)**  
"Are occupations 'bundles of skills'? Identifying latent skill profiles in the labor market using topic modeling"

**方法**：
- LDA主题建模识别潜在技能组合
- 发现职业边界模糊，技能组合重叠
- 应用于劳动力市场技能演化分析

**延伸**：
- 面板数据追踪技能需求变迁
- 异质性分析识别地区/行业特征
- 政策意义：教育培训、就业匹配

---

## 常见问题

**Q: 为什么用LDA而不是BERTopic？**  
A: LDA提供概率分布(θ)，更适合：
- 面板数据分析（时间趋势）
- 异质性分析（跨组比较）
- 回归建模（技能强度作为变量）

**Q: 内存不足？**  
A: 减少sample_size，从10万条开始测试

**Q: 训练太慢？**  
A: 调整workers参数，使用多核并行

**Q: 主题质量不好？**  
A: 调整num_topics (50-150)，或增加训练数据

**Q: 如何解释θ（技能分布）？**  
A: θ[i,j] = 职位i使用技能j的强度  
   - 0.5 = 该技能占职位的50%
   - 可以多个技能共存（如：编程+数据分析）

---

## 时间规划

- **Week 1**: LDA训练 + 技能识别
- **Week 2**: 人工标注Codebook
- **Week 3**: 面板数据构建 + 趋势分析
- **Week 4**: 异质性分析 + 回归建模

---

## 下一步

1. 运行 `lda_skill_extraction.py` 完成模型训练
2. 查看 `skill_review_report.txt` 审查技能组合质量
3. 使用 `build_codebook.py` 生成标注模板
4. 完成人工标注（预计2-3天）
5. 进行面板数据分析和可视化
