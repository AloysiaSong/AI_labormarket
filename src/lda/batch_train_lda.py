#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量训练LDA：自动处理所有已分词的时间窗口

使用场景：
- 重新划分窗口后，批量训练所有LDA模型
- 自动检测已分词数据，跳过未准备好的窗口
"""

import sys
from pathlib import Path
import subprocess

# 路径配置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import TOKENIZED_DIR, LDA_OUTPUT_DIR

TRAIN_SCRIPT = Path(__file__).parent / "train_lda.py"
MODEL_DIR = LDA_OUTPUT_DIR / "models"


def find_tokenized_windows():
    """查找所有已分词的窗口"""
    corpus_files = sorted(TOKENIZED_DIR.glob("window_*_corpus.pkl"))
    windows = [f.stem.replace("_corpus", "") for f in corpus_files]
    return windows


def check_model_exists(window_name: str) -> bool:
    """检查模型是否已存在"""
    model_path = MODEL_DIR / f"{window_name}_lda.model"
    return model_path.exists()


def train_window(window_name: str, force: bool = False):
    """训练单个窗口的LDA"""
    if not force and check_model_exists(window_name):
        print(f"  ⏭️  模型已存在，跳过 {window_name}")
        return True
    
    print(f"\n{'='*70}")
    print(f"🔄 开始训练: {window_name}")
    print(f"{'='*70}")
    
    try:
        # 调用train_lda.py的train_window函数
        # 为了简化，直接导入并调用
        from train_lda import train_window as do_train
        result = do_train(window_name)
        
        if result is not None:
            print(f"✅ {window_name} 训练完成")
            return True
        else:
            print(f"❌ {window_name} 训练失败")
            return False
            
    except Exception as e:
        print(f"❌ {window_name} 训练出错: {e}")
        return False


def main(force_retrain: bool = False):
    """
    主流程
    
    Args:
        force_retrain: 是否强制重新训练已有模型
    """
    print("="*70)
    print("🚀 批量训练LDA模型")
    print("="*70)
    
    # 1. 查找已分词窗口
    windows = find_tokenized_windows()
    print(f"\n📁 发现 {len(windows)} 个已分词窗口:")
    for w in windows:
        status = "✅" if check_model_exists(w) else "⏳"
        print(f"  {status} {w}")
    
    if not windows:
        print("\n⚠️ 没有找到已分词的窗口数据！")
        print("   请先运行: python src/cleaning/tokenize_with_esco.py")
        return
    
    # 2. 逐个训练
    print("\n" + "="*70)
    print("开始训练...")
    print("="*70)
    
    results = {}
    for window in windows:
        success = train_window(window, force=force_retrain)
        results[window] = success
    
    # 3. 汇总结果
    print("\n" + "="*70)
    print("📊 训练结果汇总")
    print("="*70)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n成功: {success_count}/{total_count}")
    
    for window, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {window}")
    
    if success_count == total_count:
        print("\n🎉 所有窗口训练完成！")
        print(f"\n下一步: python src/lda/align_topics.py")
    else:
        print(f"\n⚠️ 有 {total_count - success_count} 个窗口训练失败")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量训练LDA模型")
    parser.add_argument("--force", action="store_true", 
                        help="强制重新训练已有模型")
    
    args = parser.parse_args()
    main(force_retrain=args.force)
