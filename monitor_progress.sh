#!/bin/bash
# 监控分词和LDA训练进度

echo "📊 TY项目进度监控"
echo "===================="

echo -e "\n【1】分词进度："
echo "-------------------"
ls -lh /Users/yu/code/code2601/TY/data/processed/tokenized/window_*.pkl 2>/dev/null | wc -l | xargs -I {} echo "已完成分词的窗口: {}个 / 预期5个"

echo -e "\n详细："
for window in 2016_2017 2018_2019 2020_2021 2022_2023 2024_2025; do
    if [ -f "/Users/yu/code/code2601/TY/data/processed/tokenized/window_${window}_corpus.pkl" ]; then
        echo "  ✅ window_${window}"
    else
        echo "  ⏳ window_${window} (处理中...)"
    fi
done

echo -e "\n【2】LDA模型训练进度："
echo "-------------------"
ls -1 /Users/yu/code/code2601/TY/output/lda/models/window_*_lda.model 2>/dev/null | wc -l | xargs -I {} echo "已训练的模型: {}个"

echo -e "\n详细："
for window in 2016_2017 2018_2019 2020_2021 2022_2023 2024_2025; do
    if [ -f "/Users/yu/code/code2601/TY/output/lda/models/window_${window}_lda.model" ]; then
        echo "  ✅ window_${window}"
    else
        echo "  ⏳ window_${window}"
    fi
done

echo -e "\n【3】当前运行的进程："
echo "-------------------"
ps aux | grep -E "tokenize|train_lda" | grep -v grep | awk '{print $11, $12, $13}' || echo "  无活动进程"

echo -e "\n【4】最近日志（分词）："
echo "-------------------"
tail -3 /tmp/tokenize_log.txt 2>/dev/null || echo "  无日志"

echo -e "\n===================="
echo "刷新间隔: watch -n 10 ./monitor_progress.sh"
