#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主入口：数据清洗流水线
功能：
    1. Step 1: 正则提取任职要求 -> all_in_one2.csv
    2. Step 2: SimHash/MinHash 去重 -> all_in_one2_dedup.csv

运行方式：
    python run_cleaning_pipeline.py
    
或者单独运行：
    python step1_extract_requirements.py
    python step2_deduplicate.py
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 获取当前脚本目录
SCRIPT_DIR = Path(__file__).parent


def run_step(script_name: str, description: str) -> bool:
    """运行单个步骤"""
    script_path = SCRIPT_DIR / script_name
    
    logger.info(f"\n{'='*60}")
    logger.info(f"开始: {description}")
    logger.info(f"脚本: {script_path}")
    logger.info(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        logger.info(f"\n✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ {description} 失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("数据清洗流水线")
    logger.info("="*60)
    
    steps = [
        ("step1_extract_requirements.py", "Step 1: 正则提取任职要求"),
        ("step2_deduplicate.py", "Step 2: SimHash/MinHash 去重"),
    ]
    
    for script_name, description in steps:
        success = run_step(script_name, description)
        if not success:
            logger.error("流水线中断!")
            sys.exit(1)
    
    logger.info("\n" + "="*60)
    logger.info("🎉 全部清洗步骤完成!")
    logger.info("="*60)
    
    # 显示输出文件
    output_files = [
        SCRIPT_DIR / "all_in_one2.csv",
        SCRIPT_DIR / "all_in_one2_dedup.csv",
    ]
    
    logger.info("\n输出文件:")
    for f in output_files:
        if f.exists():
            size_mb = f.stat().st_size / (1024*1024)
            logger.info(f"  ✓ {f.name} ({size_mb:.1f} MB)")
        else:
            logger.info(f"  ✗ {f.name} (未生成)")


if __name__ == '__main__':
    main()
