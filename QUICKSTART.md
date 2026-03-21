# 🚀 TY项目 - 快速开始指南

**更新时间**: 2026-02-03  
**目标**: 使用LDA + SBERT双方法分析岗位综合化趋势

---

## ✅ 已完成工作

1. ✅ 数据清洗管道 (13.5M 条招聘数据)
2. ✅ ESCO 专业词典构建
3. ✅ 部分时间窗口的 LDA 训练
4. ✅ **核心分析模块全部开发完成**

---

## 🎯 立即开始（三条命令）

### 方案A：一键运行全流程 ⭐ 推荐
```bash
cd /Users/yu/code/code2601/TY
python scripts/run_full_analysis.py
```

### 方案B：分步运行（便于调试）

#### Step 1: 主题对齐
```bash
cd /Users/yu/code/code2601/TY/src/lda
python align_topics.py
```
**输出**:
- `output/lda/alignment/alignment_matrix.csv` - 主题对齐矩阵
- `output/lda/alignment/alignment_quality.csv` - 对齐质量评估
- `output/lda/alignment/topic_labels_template.csv` - 待人工标注

#### Step 2: SBERT 技能指纹
```bash
cd /Users/yu/code/code2601/TY/src/sbert
python compute_skill_fingerprint.py
```
**输出**:
- `output/sbert/fingerprint_evolution_TY.csv` - 年度技能指纹
- `output/sbert/detail_window_*.npy` - 详细相似度矩阵

#### Step 3: 综合化指标
```bash
cd /Users/yu/code/code2601/TY/src/analysis
python build_comprehensiveness_index.py
```
**输出**:
- `output/lda/analysis/comprehensiveness_time_series.csv` - LDA指标
- `output/lda/analysis/robustness_check.csv` - 稳健性检验

#### Step 4: 可视化
```bash
python visualize_results.py
```
**输出**:
- `output/reports/figures/comprehensiveness_trend.png` - 趋势图
- `output/reports/figures/skill_heatmap.png` - 技能热力图
- `output/reports/figures/alignment_quality.png` - 对齐质量
- `output/reports/figures/robustness_scatter.png` - 相关性散点图

---

## 📊 预期结果

### 1. 综合化指数时序图
- 展示 2015-2025 年岗位综合化的演变
- LDA 和 SBERT 双曲线对比
- 误差带显示方差

### 2. 稳健性检验
- **目标**: Pearson 相关系数 **r > 0.70**
- 如果达到 → 证明两种方法高度一致 ✅
- 如果较低 → 检查主题对齐质量或调整参数

### 3. 技能演化热力图
- 20 个技能维度的时序变化
- 识别快速上升的技能（如 GenAI）

---

## ⚠️ 注意事项

### 1. 前置条件
确保以下文件存在：
```bash
# 检查分词数据
ls /Users/yu/code/code2601/TY/data/processed/tokenized/

# 检查LDA模型
ls /Users/yu/code/code2601/TY/output/lda/models/
```

如果缺少某个时间窗口的分词数据，需要先运行：
```bash
cd /Users/yu/code/code2601/TY/src/cleaning
python tokenize_windows.py  # 假设您有此脚本
```

### 2. 硬件要求
- **推荐**: Apple Silicon (M1/M2/M3) 或 NVIDIA GPU
- **最低**: CPU (速度较慢，但可用)
- **内存**: 至少 16GB RAM（处理 13.5M 数据时）

### 3. Python 依赖
```bash
pip install -r requirements.txt

# 核心依赖:
# - gensim >= 4.0
# - sentence-transformers >= 2.0
# - scikit-learn
# - pandas, numpy
# - matplotlib, seaborn
# - scipy
```

---

## 🎓 给导师的汇报材料

### 当前可交付物
1. ✅ **完整的分析框架代码**（LDA + SBERT）
2. ✅ **主题对齐算法**（基于顶刊方法）
3. ✅ **综合化指标公式**（香农熵 + HHI）
4. ⏳ **实证结果**（待运行完整数据）

### 下周可提供
运行完成后，您将获得：
1. 📊 **综合化指数时序图**（2015-2025）
2. 📈 **双方法相关性验证** (r > 0.7 则通过稳健性检验)
3. 📋 **60个主题的标签词典**（需人工审核）
4. 📄 **初步发现总结**（1-2页报告）

---

## 🐛 常见问题排查

### Q1: 运行 align_topics.py 报错 "模型不存在"
**解决**: 确保所有窗口都已训练 LDA 模型
```bash
ls /Users/yu/code/code2601/TY/output/lda/models/
# 应该看到 window_*_lda.model 文件
```

### Q2: SBERT 运行速度慢
**解决**: 
1. 检查是否启用了 MPS/CUDA 加速
2. 减小 batch_size (默认 128 → 改为 64)
3. 使用更小的模型（如 `all-MiniLM-L6-v2`）

### Q3: 稳健性检验相关性低 (r < 0.5)
**可能原因**:
1. 主题对齐质量不佳 → 检查 `alignment_quality.csv`
2. 技能本体定义不匹配 → 调整 20 维技能词典
3. GenAI 门控权重过高 → 调整关键词匹配阈值

---

## 📚 代码架构说明

```
核心模块：
1. align_topics.py          - 锚点主题法 + 匈牙利算法
2. compute_skill_fingerprint.py - SBERT编码 + 余弦相似度
3. build_comprehensiveness_index.py - 香农熵 + HHI指数
4. visualize_results.py     - Matplotlib/Seaborn绘图
```

**参考文献**:
- 主题对齐: Sato et al. (2019, KDD), Autor et al. (2021, AER)
- 综合化指标: Shannon (1948), Herfindahl (1950)
- SBERT: Reimers & Gurevych (2019, EMNLP)

---

## 🚀 立即执行

```bash
# 1. 进入项目目录
cd /Users/yu/code/code2601/TY

# 2. 一键运行（推荐）
python scripts/run_full_analysis.py

# 3. 查看结果
open output/reports/figures/
open output/lda/analysis/comprehensiveness_time_series.csv
```

**预计运行时间**: 2-4 小时（取决于数据量和硬件）

---

**祝研究顺利！有问题随时联系。** 🎉
