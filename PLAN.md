# 下一步规划：描述性趋势分析

## 定位

导师原话："哪怕就是描述一下这个趋势，也有意义。"

**这是一篇描述性研究 (descriptive study)**，不需要因果识别。但需要：
- 趋势是稳健的（不随参数选择而翻转）
- 排除了最明显的混淆（文档长度、行业组成）
- 有足够的异质性分析增加厚度

## 现有资产

| 资产 | 状态 |
|------|------|
| processed_corpus.jsonl (800万条, 含ID/year/tokens) | ✅ 完成 |
| global_lda_fast.bin (K=50, 10%采样训练) | ✅ 完成 |
| final_results_sample.csv (776万条 entropy/HHI) | ✅ 完成 |
| 窗口CSV (含city/education/salary元数据) | ✅ 可用但未关联 |

## 已发现的关键事实

```
Year   Entropy   Tokens   Δ_entropy   Δ_tokens
2016    0.4036    14.4       +0.0%      +0.0%     ← 基线
2017    0.4069    14.4       +0.8%
2018    0.4053    14.6       +0.4%
         -------- 2019-2020 数据断层 --------
2021    0.4329    20.9       +7.3%     +45.5%
2022    0.4373    20.8       +8.3%
2023    0.4390    20.8       +8.8%
2024    0.4411    20.5       +9.3%     +42.4%

Entropy vs token_count: Pearson r = 0.175
```

趋势存在，但文档长度（+45%）跳变与 entropy（+9%）跳变同步出现，
需要控制后才能声称 entropy 的变化不是纯粹的文档变长效应。

---

## 执行计划

### Phase 1：构建分析数据集（带元数据）
**目标**：把 entropy 分数与控制变量（城市、学历、薪资、文档长度）关联起来

**做法**：修改 step1_preprocess.py，在输出 JSONL 时保留元数据字段：
- city (工作城市)
- education (学历要求)
- salary_low / salary_high (最低/最高月薪)
- token_count (分词后token数)

然后重新运行 step3_infer.py，在输出 CSV 中追加这些字段。

**产出**：`analysis_dataset.csv`，每行包含：
id, year, city, education, salary_low, salary_high, token_count, entropy, hhi, dominant_topic

### Phase 2：核心趋势 + 长度控制
**目标**：展示 entropy 趋势在控制文档长度后依然成立

**做法**：
1. OLS回归：entropy = α + β₁·year + β₂·token_count + ε
2. 输出 length-residualized entropy（残差）的年度均值
3. 对比原始趋势 vs 控制长度后的趋势

**预期**：趋势斜率变小但不消失（因为 r=0.175 是中等偏低的相关）

### Phase 3：K 稳健性检验
**目标**：证明趋势不依赖于 K=50 这一特定选择

**做法**：
1. 用 step2_train_fast.py 分别训练 K=30, 40, 60, 80 的模型（每个约30分钟，可后台并行）
2. 用 step3_infer.py 对每个模型跑全量推断（每个约1分钟）
3. 画出不同K下的 entropy 趋势叠加图

**产出**：Figure: K-robustness（5条线趋势一致）

### Phase 4：异质性分析
**目标**：增加论文厚度，回答"哪些岗位变得更综合？"

**维度**：
1. **城市层级**：一线（北上广深）vs 新一线 vs 其他
2. **学历要求**：大专 vs 本科 vs 硕士+
3. **薪资区间**：<5K / 5-10K / 10-20K / >20K

**做法**：按 year × 维度 分组计算 entropy 均值，画 facet 图

### Phase 5：出图出表
**目标**：论文级可视化

| 图表 | 内容 |
|------|------|
| Figure 1 | 主趋势图：原始 entropy + 控制长度后 entropy，含置信带 |
| Figure 2 | K 稳健性：K=30/40/50/60/80 五条趋势线 |
| Figure 3 | 异质性 facet：城市×年份、学历×年份 |
| Table 1 | 描述性统计（N, mean, sd, min, max by year） |
| Table 2 | OLS 回归表（entropy ~ year + token_count + city_FE + edu_FE） |

---

## 执行顺序与时间预估

| 步骤 | 依赖 | 预计耗时 | 方式 |
|------|------|----------|------|
| Phase 1: 重建带元数据的JSONL + 推断 | 无 | 10分钟 | 改代码+运行 |
| Phase 3: 训练K=30,40,60,80模型 | 无 | 2小时（后台并行） | nohup |
| Phase 3: 对4个新模型跑推断 | 模型训练完 | 4分钟 | 串行 |
| Phase 2: 长度控制回归 | Phase 1 | 5分钟 | Python脚本 |
| Phase 4: 异质性分析 | Phase 1 | 10分钟 | Python脚本 |
| Phase 5: 出图出表 | Phase 2,3,4 | 30分钟 | matplotlib |

**总计约 3 小时**（其中 2 小时是模型训练等待时间）

---

## 关键方法论注意事项

1. **论文中诚实声明**：这是描述性趋势，不做因果声明
2. **丢弃 2019 (n=13) 和 2025 (不完整)**，只报告 2016-2024
3. **Analytical folding-in 方法需要交代**：在方法论章节说明用的是 geometric mean variant，引用 Taddy (2012) 的 multinomial inverse regression 作为理论支撑
4. **Global LDA 的合理性**：对于"同一主题空间下的跨年比较"，global LDA 是正确选择（不需要 alignment）
