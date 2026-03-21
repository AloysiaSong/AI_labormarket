"""
TY项目路径配置文件
集中管理所有数据和输出路径
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# ============ 原始数据目录 (保持原位置) ============
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_ALL_IN_ONE = DATASET_DIR / "all_in_one.csv"
RAW_ALL_IN_ONE_DTA = DATASET_DIR / "all_in_one.dta"
RAW_YEARLY_DIR = DATASET_DIR / "分年份保存数据"

# ============ 处理后数据目录 ============
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# 清洗后数据
CLEANED_DIR = PROCESSED_DIR / "cleaned"
CLEANED_DATA = CLEANED_DIR / "all_in_one1.csv"

# 提取后数据
EXTRACTED_DIR = PROCESSED_DIR / "extracted"
EXTRACTED_DATA = EXTRACTED_DIR / "all_in_one2.csv"

# 去重后数据
DEDUPED_DIR = PROCESSED_DIR / "deduped"
DEDUPED_DATA = DEDUPED_DIR / "all_in_one2_dedup.csv"

# 时间窗口数据
WINDOWS_DIR = PROCESSED_DIR / "windows"
WINDOW_2016_2018 = WINDOWS_DIR / "window_2016_2018.csv"
WINDOW_2021_2022 = WINDOWS_DIR / "window_2021_2022.csv"
WINDOW_2023_2024 = WINDOWS_DIR / "window_2023_2024.csv"

# 分词后数据
TOKENIZED_DIR = PROCESSED_DIR / "tokenized"

# ============ ESCO数据目录 ============
ESCO_DIR = DATA_DIR / "esco"
ESCO_ORIGINAL_DIR = ESCO_DIR / "original"
ESCO_CHINESE_DIR = ESCO_DIR / "chinese"
JIEBA_DICT_DIR = ESCO_DIR / "jieba_dict"

# ESCO相关文件
SKILLS_EN_CSV = ESCO_ORIGINAL_DIR / "skills_en.csv"
SKILLS_CHINESE_CSV = ESCO_CHINESE_DIR / "skills_chinese.csv"
ESCO_JIEBA_DICT = JIEBA_DICT_DIR / "esco_jieba_dict.txt"
SKILL_METADATA_CSV = JIEBA_DICT_DIR / "skill_metadata.csv"

# ============ 输出目录 ============
OUTPUT_DIR = PROJECT_ROOT / "output"
LDA_OUTPUT_DIR = OUTPUT_DIR / "lda"
SBERT_OUTPUT_DIR = OUTPUT_DIR / "sbert"
ESCO_OUTPUT_DIR = OUTPUT_DIR / "esco"
REPORTS_DIR = OUTPUT_DIR / "reports"

# ============ 缓存目录 ============
CACHE_DIR = PROJECT_ROOT / "cache"
TRANSLATION_CACHE_DIR = CACHE_DIR / "translation"

# ============ 源代码目录 ============
SRC_DIR = PROJECT_ROOT / "src"
CLEANING_SRC_DIR = SRC_DIR / "cleaning"
LDA_SRC_DIR = SRC_DIR / "lda"
ESCO_SRC_DIR = SRC_DIR / "esco"

# ============ 辅助函数 ============
def ensure_dirs():
    """确保所有必要的目录存在"""
    dirs = [
        CLEANED_DIR, EXTRACTED_DIR, DEDUPED_DIR, WINDOWS_DIR, TOKENIZED_DIR,
        ESCO_ORIGINAL_DIR, ESCO_CHINESE_DIR, JIEBA_DICT_DIR,
        LDA_OUTPUT_DIR, SBERT_OUTPUT_DIR, ESCO_OUTPUT_DIR, REPORTS_DIR,
        TRANSLATION_CACHE_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def get_yearly_data(year: int) -> Path:
    """获取指定年份的原始数据文件路径"""
    return RAW_YEARLY_DIR / f"智联招聘数据库{year}.csv"
