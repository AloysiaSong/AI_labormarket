#!/usr/bin/env python3
"""
任务对数据CSV导出工具
将智能过滤后的任务对数据导出为CSV格式
"""

import pickle
import csv
from pathlib import Path

def export_task_pairs_to_csv(data, output_dir):
    """导出任务对数据到CSV"""
    output_file = output_dir / "task_pairs.csv"

    print(f"导出任务对数据到: {output_file}")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头
        writer.writerow(['verb', 'noun', 'frequency'])

        # 按频率降序排序
        sorted_pairs = sorted(data['task_pairs'].items(), key=lambda x: x[1], reverse=True)

        # 写入数据
        for (verb, noun), frequency in sorted_pairs:
            writer.writerow([verb, noun, frequency])

    print(f"成功导出 {len(sorted_pairs)} 个任务对")

def export_verbs_to_csv(data, output_dir):
    """导出动词频率数据到CSV"""
    output_file = output_dir / "verbs.csv"

    print(f"导出动词数据到: {output_file}")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头
        writer.writerow(['verb', 'frequency'])

        # 按频率降序排序
        sorted_verbs = sorted(data['verb_freq'].items(), key=lambda x: x[1], reverse=True)

        # 写入数据
        for verb, frequency in sorted_verbs:
            writer.writerow([verb, frequency])

    print(f"成功导出 {len(sorted_verbs)} 个动词")

def export_nouns_to_csv(data, output_dir):
    """导出名词频率数据到CSV"""
    output_file = output_dir / "nouns.csv"

    print(f"导出名词数据到: {output_file}")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头
        writer.writerow(['noun', 'frequency'])

        # 按频率降序排序
        sorted_nouns = sorted(data['noun_freq'].items(), key=lambda x: x[1], reverse=True)

        # 写入数据
        for noun, frequency in sorted_nouns:
            writer.writerow([noun, frequency])

    print(f"成功导出 {len(sorted_nouns)} 个名词")

def export_metadata_to_csv(data, output_dir):
    """导出元数据到CSV"""
    output_file = output_dir / "metadata.csv"

    print(f"导出元数据到: {output_file}")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头
        writer.writerow(['key', 'value'])

        # 写入元数据
        for key, value in data['metadata'].items():
            if isinstance(value, dict):
                # 对于嵌套字典，转换为字符串
                value = str(value)
            writer.writerow([key, value])

    print("成功导出元数据")

def create_summary_report(data, output_dir):
    """创建汇总报告"""
    report_file = output_dir / "export_summary.txt"

    print(f"创建汇总报告: {report_file}")

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("任务对数据导出汇总报告\n")
        f.write("=" * 50 + "\n\n")

        f.write("数据概览:\n")
        f.write(f"- 任务对总数: {len(data['task_pairs'])}\n")
        f.write(f"- 唯一动词数: {len(data['verb_freq'])}\n")
        f.write(f"- 唯一名词数: {len(data['noun_freq'])}\n")
        f.write(f"- 过滤类型: {data['metadata']['filter_type']}\n\n")

        f.write("Top 10 任务对:\n")
        sorted_pairs = sorted(data['task_pairs'].items(), key=lambda x: x[1], reverse=True)
        for i, ((verb, noun), count) in enumerate(sorted_pairs[:10]):
            f.write(f"{i+1:2d}. {verb}-{noun}: {count}\n")

        f.write("\nTop 10 动词:\n")
        sorted_verbs = sorted(data['verb_freq'].items(), key=lambda x: x[1], reverse=True)
        for i, (verb, count) in enumerate(sorted_verbs[:10]):
            f.write(f"{i+1:2d}. {verb}: {count}\n")

        f.write("\nTop 10 名词:\n")
        sorted_nouns = sorted(data['noun_freq'].items(), key=lambda x: x[1], reverse=True)
        for i, (noun, count) in enumerate(sorted_nouns[:10]):
            f.write(f"{i+1:2d}. {noun}: {count}\n")

        f.write("\n导出文件:\n")
        f.write("- task_pairs.csv: 所有任务对数据\n")
        f.write("- verbs.csv: 动词频率统计\n")
        f.write("- nouns.csv: 名词频率统计\n")
        f.write("- metadata.csv: 元数据信息\n")
        f.write("- export_summary.txt: 本汇总报告\n")

    print("汇总报告创建完成")

def main():
    """主函数：导出所有数据到CSV"""
    # 设置路径
    data_file = '/Users/yu/code/code2601/TY/output/taskpair/task_pairs_dict_smart_filtered.pkl'
    output_dir = Path('/Users/yu/code/code2601/TY/output/taskpair')

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始导出任务对数据到CSV...")
    print(f"数据文件: {data_file}")
    print(f"输出目录: {output_dir}")

    # 加载数据
    print("加载数据...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    print(f"数据加载完成: {len(data['task_pairs'])} 个任务对")

    # 导出各种数据
    export_task_pairs_to_csv(data, output_dir)
    export_verbs_to_csv(data, output_dir)
    export_nouns_to_csv(data, output_dir)
    export_metadata_to_csv(data, output_dir)
    create_summary_report(data, output_dir)

    print("\n✅ 所有数据导出完成！")
    print(f"📁 输出目录: {output_dir}")
    print("📄 生成文件:")
    print("  - task_pairs.csv")
    print("  - verbs.csv")
    print("  - nouns.csv")
    print("  - metadata.csv")
    print("  - export_summary.txt")

if __name__ == "__main__":
    main()