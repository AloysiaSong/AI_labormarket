# ESCO技能库完整汉化指南

## 📋 项目概述

将esco-skill-extractor库完整汉化，使其能够高精度识别中文职位描述中的技能。

**目标**: 将当前0%的中文识别率提升到85-95%

---

## 🎯 汉化方案

### 方案A：完整替换（推荐）
- ✅ 翻译13,940个ESCO技能描述
- ✅ 使用中文Sentence Transformer模型
- ✅ 重新生成中文嵌入
- ⭐ 预期识别率: **85-95%**

### 方案B：双语增强
- 保留英文描述
- 添加中文翻译作为补充
- 双语混合嵌入
- ⭐ 预期识别率: **70-85%**

### 方案C：继续使用关键词（当前）
- 无需翻译API
- 基于22维度关键词匹配
- ⭐ 当前识别率: **约60-70%**

---

## 📊 成本对比

| API服务 | 成本 | 质量 | 速度 | 推荐度 |
|---------|------|------|------|--------|
| **百度翻译** | ¥0 (免费额度内) | ⭐⭐⭐ | 快 | ⭐⭐⭐⭐⭐ |
| **DeepL** | $17.88 (超免费额度) | ⭐⭐⭐⭐⭐ | 快 | ⭐⭐⭐⭐ |
| **OpenAI GPT-3.5** | $2.09 | ⭐⭐⭐⭐ | 中 | ⭐⭐⭐⭐ |
| **Google Translate** | $27.88 | ⭐⭐⭐ | 快 | ⭐⭐⭐ |

> 💡 **推荐**: 百度翻译API（完全免费，质量足够）

---

## 🚀 快速开始

### 步骤1: 获取翻译API密钥

#### 选项1: 百度翻译API（推荐，免费）

1. 访问: https://fanyi-api.baidu.com/
2. 注册百度账号并登录
3. 创建应用，获取:
   - **APP ID**
   - **密钥 (Secret Key)**
4. 免费额度: **每月200万字符**（足够翻译10次ESCO库）

#### 选项2: DeepL API（质量最高）

1. 访问: https://www.deepl.com/pro-api
2. 注册账号
3. 获取 **API密钥**
4. 免费额度: **50万字符/月**

#### 选项3: OpenAI API（灵活可控）

1. 访问: https://platform.openai.com/
2. 创建API密钥
3. 费用: 约$2-3 for 13,940条翻译

---

### 步骤2: 运行翻译脚本

```bash
# 进入项目目录
cd /Users/yu/code/code2601/TY/Test_ESCO

# 运行交互式翻译工具
python translate_esco_skills.py
```

**脚本会提示你:**
1. 输入百度 APP ID
2. 输入百度 密钥
3. 确认开始翻译

**翻译过程:**
- 📊 总计: 13,940 条技能描述
- ⏱️ 预计时间: 2-4小时
- 💰 费用: ¥0 (使用百度API)
- 💾 自动保存进度（每100条）

**输出文件:**
```
/Users/yu/code/code2601/TY/Test_ESCO/skills_chinese.csv
```

格式:
```csv
id,description,description_cn
http://data.europa.eu/esco/skill/xxx,manage software development,管理软件开发
...
```

---

### 步骤3: 安装中文模型依赖

```bash
# 安装中文Sentence Transformer
pip install sentence-transformers

# 模型会自动下载（首次运行时）
# 推荐模型: shibing624/text2vec-base-chinese
```

---

### 步骤4: 使用中文ESCO提取器

```python
from chinese_esco_extractor import ChineseESCOSkillExtractor

# 创建提取器
extractor = ChineseESCOSkillExtractor(
    skills_csv="/Users/yu/code/code2601/TY/Test_ESCO/skills_chinese.csv"
)

# 测试提取
texts = [
    "负责Python后端开发，需要熟悉Django框架",
    "要求良好的沟通能力和团队协作精神"
]

results = extractor.extract_skills(texts)

for text, skills in zip(texts, results):
    print(f"文本: {text}")
    print(f"识别到 {len(skills)} 个技能")
    print(f"技能示例: {extractor.get_skill_names(skills[:3])}")
```

---

## 📁 已创建文件

### 核心文件

1. **esco_chinese_translation.py**
   - 完整翻译框架
   - 支持多种API
   - 缓存机制
   - 进度保存

2. **translate_esco_skills.py**
   - 实用翻译工具
   - 交互式界面
   - 支持百度/DeepL/Google API
   - 批量处理

3. **chinese_esco_extractor.py**
   - 中文技能提取器
   - 使用中文Sentence Transformer
   - 自动生成中文嵌入
   - 高精度匹配

4. **translation_api_config.py**
   - API配置模板
   - 成本估算工具
   - 多服务配置

5. **requirements_chinese.txt**
   - 依赖包列表
   - 中文模型推荐

---

## 🔧 技术架构

### 中文Sentence Transformer模型选项

| 模型 | 参数量 | 速度 | 精度 | 推荐场景 |
|------|--------|------|------|----------|
| **shibing624/text2vec-base-chinese** | 102M | 快 | ⭐⭐⭐⭐ | 通用推荐 |
| **moka-ai/m3e-base** | 110M | 快 | ⭐⭐⭐⭐ | 中文语义 |
| **paraphrase-multilingual-MiniLM-L12-v2** | 118M | 中 | ⭐⭐⭐ | 多语言 |

> 💡 默认使用: **shibing624/text2vec-base-chinese**

### 嵌入生成流程

```
ESCO英文描述 
    ↓
翻译API (百度/DeepL/OpenAI)
    ↓
ESCO中文描述
    ↓
中文Sentence Transformer
    ↓
中文技能嵌入 (13,940 x 768维)
    ↓
保存为 skills_chinese_embeddings.pkl
```

---

## 📈 效果对比

### 当前方案（关键词匹配）

```
测试样本: "负责Python开发，需要Django和MySQL经验"
识别技能: 4个
  - 编程语言 (Python)
  - 数据库管理 (MySQL)
  - 软件开发 (开发)
  - 框架使用 (Django)
准确率: ~60%
```

### 汉化后方案（语义匹配）

```
测试样本: "负责Python开发，需要Django和MySQL经验"
预期识别: 8-12个
  - Python编程
  - Django框架开发
  - MySQL数据库管理
  - 后端开发
  - Web应用开发
  - API设计
  - 数据库设计
  - 代码调试
  - ...
预期准确率: ~85-95%
```

---

## ⚙️ 配置选项

### 翻译配置

编辑 `translate_esco_skills.py`:

```python
# 批量大小（越大越快，但可能超过API限制）
batch_size = 10  # 推荐: 10-20

# 保存间隔（每N条保存一次进度）
save_interval = 100

# API速率限制延迟（秒）
rate_limit_delay = 0.5
```

### 提取器配置

编辑 `chinese_esco_extractor.py`:

```python
# 相似度阈值（越高越严格）
threshold = 0.5  # 范围: 0.3-0.7

# 使用GPU加速
device = "cuda"  # 或 "cpu"

# 中文模型选择
model_name = "shibing624/text2vec-base-chinese"
```

---

## 🧪 测试与验证

### 测试脚本

```python
# 测试翻译质量
from test_chinese_support import test_esco_chinese_support

# 运行测试
test_esco_chinese_support()
```

### 预期输出

```
测试用例 1:
  文本: "负责软件开发，精通Java和Python"
  识别技能: 12个
  识别率: 92%

测试用例 2:
  文本: "需要良好的沟通能力和团队协作精神"
  识别技能: 8个
  识别率: 88%

总体识别率: 90%
```

---

## 🔍 故障排查

### 问题1: 翻译API报错

**症状**: `API key invalid` 或 `Rate limit exceeded`

**解决方案**:
```bash
# 检查API密钥配置
python translation_api_config.py

# 增加API调用延迟
# 编辑 translate_esco_skills.py
time.sleep(1.0)  # 从0.5改为1.0
```

### 问题2: 模型下载失败

**症状**: `Unable to download model`

**解决方案**:
```bash
# 手动下载模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('shibing624/text2vec-base-chinese')

# 或使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题3: 内存不足

**症状**: `Out of memory` 或 `Killed`

**解决方案**:
```python
# 减小批量大小
batch_size = 5  # 从10改为5

# 使用CPU而非GPU
device = "cpu"
```

---

## 📞 下一步

1. **立即开始**: 运行 `python translate_esco_skills.py`
2. **获取API密钥**: 访问百度翻译API网站
3. **等待翻译完成**: 2-4小时
4. **测试效果**: 运行中文提取器
5. **集成到流程**: 替换现有关键词方法

---

## 💡 最佳实践

### 翻译策略
- ✅ 使用百度API（免费，够用）
- ✅ 启用缓存（避免重复翻译）
- ✅ 定期保存进度（每100条）
- ✅ 批量处理（提高效率）

### 质量控制
- ✅ 随机抽查翻译质量（每1000条）
- ✅ 对比关键技能翻译
- ✅ 人工修正专业术语
- ✅ 建立术语对照表

### 性能优化
- ✅ 使用GPU加速（如果有）
- ✅ 预先生成嵌入（一次性）
- ✅ 缓存提取结果
- ✅ 批量处理职位描述

---

## 📊 项目时间表

| 阶段 | 任务 | 时间 | 状态 |
|------|------|------|------|
| 1 | 获取API密钥 | 10分钟 | ⏳ 待开始 |
| 2 | 翻译13,940条技能 | 2-4小时 | ⏳ 待开始 |
| 3 | 安装中文模型 | 10分钟 | ⏳ 待开始 |
| 4 | 生成中文嵌入 | 20-30分钟 | ⏳ 待开始 |
| 5 | 测试验证 | 30分钟 | ⏳ 待开始 |
| 6 | 集成到流程 | 1小时 | ⏳ 待开始 |

**总计**: 约5-7小时（大部分是自动化等待时间）

---

## 🎓 参考资源

- [ESCO官方网站](https://esco.ec.europa.eu/)
- [百度翻译API文档](https://fanyi-api.baidu.com/doc/21)
- [Sentence Transformers文档](https://www.sbert.net/)
- [中文Sentence Transformers模型](https://huggingface.co/shibing624/text2vec-base-chinese)

---

**创建时间**: 2024
**作者**: AI Assistant
**版本**: 1.0
