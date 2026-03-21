#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动：一键运行完整分析流程

执行顺序：
1. 主题对齐（如果尚未完成）
2. 计算LDA综合化指数
3. 计算SBERT技能指纹
4. 综合化指标构建
5. 稳健性检验
6. 生成可视化报告
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def run_script(script_path: Path, description: str):
    """运行单个Python脚本"""
    logger.info("\n" + "="*70)
    logger.info(f"📌 {description}")
    logger.info("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        logger.info(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} 失败: {e}")
        return False


def main():
    print("\n" + "🚀"*35)
    print(" "*20 + "TY项目 - 全流程自动化分析")
    print("🚀"*35 + "\n")
    
    # Step 1: 主题对齐
    logger.info("Step 1/5: 主题对齐（如果已完成可跳过）")
    align_script = SRC_DIR / "lda" / "align_topics.py"
    
    alignment_result_path = PROJECT_ROOT / "output" / "lda" / "alignment" / "alignment_matrix.csv"
    
    if alignment_result_path.exists():
        logger.info("  ℹ️ 主题对齐已完成，跳过此步骤")
    else:
        if not run_script(align_script, "主题对齐"):
            logger.error("主题对齐失败，请检查是否所有窗口都已训练LDA模型")
            return
    
    # Step 2: SBERT技能指纹
    logger.info("\nStep 2/5: SBERT技能指纹计算")
    sbert_script = SRC_DIR / "sbert" / "compute_skill_fingerprint.py"
    
    if not run_script(sbert_script, "SBERT技能指纹"):
        logger.warning("⚠️ SBERT计算失败，将仅使用LDA方法")
    
    # Step 3: 综合化指标
    logger.info("\nStep 3/5: 综合化指标构建")
    index_script = SRC_DIR / "analysis" / "build_comprehensiveness_index.py"
    
    if not run_script(index_script, "综合化指标"):
        logger.error("综合化指标计算失败")
        return
    
    # Step 4: 稳健性检验
    logger.info("\nStep 4/5: 稳健性检验（LDA vs SBERT）")
    logger.info("  ℹ️ 需要手动编辑 build_comprehensiveness_index.py 中的validate_with_sbert()调用")
    logger.info("  ℹ️ 或在下一步骤中生成对比图表")
    
    # Step 5: 可视化
    logger.info("\nStep 5/5: 生成可视化报告")
    viz_script = SRC_DIR / "analysis" / "visualize_results.py"
    
    if viz_script.exists():
        run_script(viz_script, "可视化报告")
    else:
        logger.info("  ℹ️ 可视化脚本尚未创建，请稍后运行")
    
    # 完成
    print("\n" + "="*70)
    print("🎉 全流程完成！请查看以下输出目录：")
    print(f"  - LDA对齐结果: {PROJECT_ROOT}/output/lda/alignment/")
    print(f"  - SBERT指纹: {PROJECT_ROOT}/output/sbert/")
    print(f"  - 综合化指标: {PROJECT_ROOT}/output/lda/analysis/")
    print("="*70)


if __name__ == "__main__":
    main()
