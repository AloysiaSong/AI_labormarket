#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: 从职位描述中提取任职要求/技能要求段落
功能：
    1. 正则提取任职要求/技能要求/岗位条件等段落
    2. 剔除公司介绍、福利待遇、薪资结构等噪音
    3. 保留清洗后的文本到 all_in_one2.csv

输入: /Users/yu/code/code2601/TY/data_cleaning/all_in_one1.csv
输出: /Users/yu/code/code2601/TY/Test_LDA/task1_datacleaning/all_in_one2.csv
"""

import pandas as pd
import re
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    class _DummyTqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter([])
        def update(self, n=1):
            return None
        def close(self):
            return None
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
    def tqdm(iterable=None, **kwargs):
        return _DummyTqdm(iterable=iterable, **kwargs)
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============== 路径配置 ==============
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import CLEANED_DATA, EXTRACTED_DATA, EXTRACTED_DIR

# 确保输出目录存在
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = str(CLEANED_DATA)
OUTPUT_FILE = str(EXTRACTED_DATA)
CHUNKSIZE = 50000  # 分块读取大文件
PART_ROWS = 500000  # 每个分段文件包含的原始行数（用于断点续跑）

# ============== 正则模式定义 ==============

# 任职要求段落的起始标识（按优先级排序）
REQUIREMENT_START_PATTERNS = [
    # 带分隔符的格式
    r'[【\[]?\s*任职要求\s*[】\]]?\s*[:：]?',
    r'[【\[]?\s*任职资格\s*[】\]]?\s*[:：]?',
    r'[【\[]?\s*岗位要求\s*[】\]]?\s*[:：]?',
    r'[【\[]?\s*技能要求\s*[】\]]?\s*[:：]?',
    r'[【\[]?\s*职位要求\s*[】\]]?\s*[:：]?',
    r'[【\[]?\s*岗位条件\s*[】\]]?\s*[:：]?',
    r'[【\[]?\s*招聘条件\s*[】\]]?\s*[:：]?',
    r'[【\[]?\s*基本要求\s*[】\]]?\s*[:：]?',
    r'[【\[]?\s*能力要求\s*[】\]]?\s*[:：]?',
    r'[【\[]?\s*要求\s*[】\]]?\s*[:：]',  # 单独的"要求："需要冒号
    # 带等号的格式 (== 任职资格 ==)
    r'==\s*任职(?:要求|资格)\s*==',
    r'--\s*任职(?:要求|资格)\s*--',
]

# 任职要求段落的结束标识（遇到这些则停止提取）
REQUIREMENT_END_PATTERNS = [
    r'[【\[]?\s*岗位职责\s*[】\]]?',
    r'[【\[]?\s*工作职责\s*[】\]]?',
    r'[【\[]?\s*职位描述\s*[】\]]?',
    r'[【\[]?\s*工作内容\s*[】\]]?',
    r'[【\[]?\s*薪资待遇\s*[】\]]?',
    r'[【\[]?\s*薪酬待遇\s*[】\]]?',
    r'[【\[]?\s*薪资福利\s*[】\]]?',
    r'[【\[]?\s*薪酬福利\s*[】\]]?',
    r'[【\[]?\s*福利待遇\s*[】\]]?',
    r'[【\[]?\s*福利\s*[】\]]?',
    r'[【\[]?\s*薪资\s*[】\]]?',
    r'[【\[]?\s*薪酬\s*[】\]]?',
    r'[【\[]?\s*待遇\s*[】\]]?',
    r'[【\[]?\s*公司介绍\s*[】\]]?',
    r'[【\[]?\s*公司简介\s*[】\]]?',
    r'[【\[]?\s*企业介绍\s*[】\]]?',
    r'[【\[]?\s*关于我们\s*[】\]]?',
    r'[【\[]?\s*联系方式\s*[】\]]?',
    r'[【\[]?\s*工作地点\s*[】\]]?',
    r'[【\[]?\s*上班地址\s*[】\]]?',
    r'[【\[]?\s*岗位介绍\s*[】\]]?',
    r'-+\s*福利待遇\s*-+',
    r'==\s*工作职责\s*==',
    r'薪资结构\s*[:：]',
    r'(?:薪资|薪酬|待遇|福利)(?:结构|说明)?\s*[:：]',
    r'企业福利\s*[:：]',
    r'人才培养计划\s*[:：]',
    r'我们的承诺\s*[:：]',
]

# 需要完全剔除的噪音内容
NOISE_PATTERNS = [
    # 福利相关
    r'五险一金[^。；]*[。；]?',
    r'(?:标准)?社保[^。；]*[。；]?',
    r'(?:公积金|商业险)[^。；]*[。；]?',
    r'带薪(?:年假|病假|休假)[^。；]*[。；]?',
    r'(?:免费)?(?:工作餐|午餐|餐补)[^。；]*[。；]?',
    r'(?:包吃|包住|住宿|食宿)[^。；]*[。；]?',
    r'节日(?:福利|礼品|礼金)[^。；]*[。；]?',
    r'生日(?:福利|礼遇|礼品)[^。；]*[。；]?',
    r'周末双休',
    r'双休',
    r'弹性工作[^。；]*',
    r'(?:年终奖|绩效奖)[^。；]*',
    r'(?:薪资|薪酬|待遇|工资|底薪|提成|奖金|补贴|津贴)[^。；]*[。；]?',
    r'(?:单双休|大小周)[^。；]*',
    r'(?:高温|交通|住房|餐饮)补贴[^。；]*',
    r'(?:养老|医疗|工伤|失业|生育)保险[^，。；]*',
    r'住房公积金[^。；]*',
    r'(?:入职|岗前|专业)培训[^。；]*',
    r'晋升(?:空间|机会|通道)[^。；]*',
    r'(?:股权|期权)激励[^。；]*',
    
    # 广告/来源相关
    r'微信公众号[^\n]*',
    r'马克数据[^\n]*',
    r'macrodatas[^\n]*',
    r'www\.[a-zA-Z0-9\.\-]+\.(?:com|cn|net)[^\n]*',
    r'来源[：:][^\n]*',
    r'更多数据[：:][^\n]*',
    r'搜索马克[^\n]*',
    r'百度搜索[^\n]*',
    
    # 工作时间描述
    r'(?:工作时间|上班时间)[：:][^。；\n]*',
    r'(?:周一|周二|周三|周四|周五|周六|周日)[至到~\-][^。；\n]{0,30}',
    r'\d{1,2}[:：]\d{2}\s*[-~至到]\s*\d{1,2}[:：]\d{2}',
    
    # 联系方式
    r'(?:联系电话|咨询电话|手机|微信)[：:][^\n]*',
    r'1[3-9]\d{9}',  # 手机号
]

# 无意义的短句/格式残留
CLEANUP_PATTERNS = [
    r'^\d+[、\.．]\s*',  # 行首数字编号
    r'^[a-zA-Z]\s*[、\.．]\s*',  # 行首字母编号
    r'^[-•●○◆★☆·]\s*',  # 行首符号
    r'\s+',  # 多余空白
]


def extract_requirements(text: str) -> str:
    """
    从职位描述中提取任职要求段落
    
    策略：
    1. 优先查找明确的"任职要求"等段落标识
    2. 如果找到，提取该段落直到下一个段落标识
    3. 如果没找到明确标识，尝试提取整体中的要求相关内容
    4. 最后进行噪音清理
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = str(text).strip()
    if len(text) < 10:
        return ""
    
    extracted = ""
    
    # 策略1: 查找明确的任职要求段落
    for start_pattern in REQUIREMENT_START_PATTERNS:
        match = re.search(start_pattern, text, re.IGNORECASE)
        if match:
            start_pos = match.end()
            remaining_text = text[start_pos:]
            
            # 查找结束位置
            end_pos = len(remaining_text)
            for end_pattern in REQUIREMENT_END_PATTERNS:
                end_match = re.search(end_pattern, remaining_text, re.IGNORECASE)
                if end_match:
                    end_pos = min(end_pos, end_match.start())
            
            extracted = remaining_text[:end_pos].strip()
            break
    
    # 策略2: 如果没找到明确段落，检查是否整体就是要求描述
    if not extracted:
        # 检查是否以数字列表开头（常见的要求格式）
        if re.match(r'^[1１一][\s、\.．：:]', text):
            # 可能整段都是要求，但需要过滤掉职责部分
            # 查找是否有"职责"相关内容，如果有则只取要求部分
            for end_pattern in REQUIREMENT_END_PATTERNS:
                end_match = re.search(end_pattern, text, re.IGNORECASE)
                if end_match:
                    # 如果职责在前面，取后面部分
                    if end_match.start() < len(text) / 2:
                        # 再次查找任职要求
                        for start_pattern in REQUIREMENT_START_PATTERNS:
                            req_match = re.search(start_pattern, text[end_match.end():], re.IGNORECASE)
                            if req_match:
                                extracted = text[end_match.end() + req_match.end():].strip()
                                break
                    else:
                        # 职责在后面，取前面部分
                        extracted = text[:end_match.start()].strip()
                    break
            
            # 如果还没提取到，整段使用
            if not extracted:
                extracted = text
    
    # 策略3: 最后尝试 - 如果文本较短且包含技能关键词
    if not extracted:
        skill_keywords = ['熟悉', '掌握', '精通', '了解', '会使用', '具备', '擅长', 
                          '经验', '学历', '专业', '年以上', '本科', '大专', '硕士']
        if any(kw in text for kw in skill_keywords):
            # 提取包含技能关键词的句子
            sentences = re.split(r'[。；\n]', text)
            skill_sentences = [s for s in sentences 
                              if any(kw in s for kw in skill_keywords)]
            if skill_sentences:
                extracted = '。'.join(skill_sentences)
    
    # 如果还是没有提取到内容，返回空
    if not extracted:
        return ""
    
    # 噪音清理
    extracted = clean_noise(extracted)
    
    return extracted


def clean_noise(text: str) -> str:
    """清理噪音内容"""
    if not text:
        return ""
    
    # 应用噪音模式过滤
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 清理格式
    for pattern in CLEANUP_PATTERNS:
        text = re.sub(pattern, ' ', text)
    
    # 规范化空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 去除残留的标点堆积
    text = re.sub(r'[，。；、]{2,}', '。', text)
    text = re.sub(r'^[，。；、\s]+', '', text)
    text = re.sub(r'[，。；、\s]+$', '', text)
    
    return text


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """处理一个数据块"""
    # 提取任职要求
    chunk['cleaned_requirements'] = chunk['职位描述'].apply(extract_requirements)
    
    # 过滤空文本和过短文本
    chunk = chunk[chunk['cleaned_requirements'].str.len() >= 20]
    
    return chunk


def main():
    """主函数"""
    logger.info(f"开始处理数据...")
    logger.info(f"输入文件: {INPUT_FILE}")
    logger.info(f"输出文件: {OUTPUT_FILE}")
    
    # 分段输出目录
    parts_dir = EXTRACTED_DIR / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"分段输出目录: {parts_dir}")
    
    # 确保输出目录存在
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    # 预估总行数（避免耗时统计）
    total_lines = 13082403  # 已知数据量
    logger.info(f"预估总行数: {total_lines:,}")
    
    # 分块处理
    total_processed = 0
    total_kept = 0
    if PART_ROWS % CHUNKSIZE != 0:
        logger.warning("PART_ROWS 不是 CHUNKSIZE 的整数倍，分段边界将按 chunk 对齐")

    # 已完成分段（断点续跑）
    existing_parts = sorted(parts_dir.glob("part_*.csv"))
    completed_parts = set()
    for p in existing_parts:
        try:
            part_idx = int(p.stem.split('_')[-1])
            completed_parts.add(part_idx)
        except Exception:
            continue
    if completed_parts:
        logger.info(f"检测到已完成分段: {sorted(completed_parts)}")

    chunks = pd.read_csv(INPUT_FILE, chunksize=CHUNKSIZE)
    
    with tqdm(total=total_lines, desc="处理进度") as pbar:
        current_part_idx = 0
        current_part_rows = 0
        current_part_path = parts_dir / f"part_{current_part_idx:03d}.csv"
        current_part_header_written = False

        for chunk in chunks:
            if total_processed > 0 and total_processed % PART_ROWS == 0:
                current_part_idx += 1
                current_part_rows = 0
                current_part_path = parts_dir / f"part_{current_part_idx:03d}.csv"
                current_part_header_written = False

            if current_part_idx in completed_parts:
                total_processed += len(chunk)
                pbar.update(len(chunk))
                continue
            # 处理当前块
            processed_chunk = process_chunk(chunk)
            
            # 保留原始字段 + 新字段
            output_columns = [
                '企业名称', '招聘岗位', '工作城市', '工作区域',
                '最低月薪', '最高月薪', '学历要求', '要求经验',
                '招聘发布年份', '来源平台', 'cleaned_requirements'
            ]
            
            # 检查哪些列存在
            available_columns = [col for col in output_columns if col in processed_chunk.columns]
            processed_chunk = processed_chunk[available_columns]
            
            # 分段写入文件
            if not current_part_header_written and current_part_rows == 0:
                mode = 'w'
                header = True
            else:
                mode = 'a'
                header = False
            
            processed_chunk.to_csv(
                current_part_path,
                mode=mode,
                index=False,
                header=header,
                encoding='utf-8'
            )
            current_part_header_written = True
            current_part_rows += len(processed_chunk)
            total_processed += len(chunk)
            total_kept += len(processed_chunk)
            pbar.update(len(chunk))
    
    logger.info(f"\n处理完成!")
    logger.info(f"原始记录数: {total_processed:,}")
    logger.info(f"保留记录数: {total_kept:,}")
    logger.info(f"保留比例: {total_kept/total_processed*100:.2f}%")
    logger.info(f"分段输出目录: {parts_dir}")
    logger.info("提示: 如需合并为单文件，可在去重步骤中自动合并")


if __name__ == '__main__':
    main()
