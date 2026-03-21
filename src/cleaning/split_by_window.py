#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按时间窗口分割数据
将去重后的招聘数据按2年一个窗口划分为五个时间窗口

时间窗口设计（均匀分割，改善LDA对齐质量）:
- Window 1 (2016-2017): 早期阶段
- Window 2 (2018-2019): 过渡阶段
- Window 3 (2020-2021): 数字化深化
- Window 4 (2022-2023): 转型关键期
- Window 5 (2024-2025): AI 2.0 爆发期
"""
import pandas as pd
import sys
from pathlib import Path
import re
import argparse

# 使用集中路径配置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.paths import DEDUPED_DATA, PROCESSED_DIR

# 输出目录
WINDOWS_DIR = PROCESSED_DIR / "windows"
WINDOWS_DIR.mkdir(parents=True, exist_ok=True)

# 中国行政区划白名单（省级 + 直辖市/自治区/特别行政区 + 台湾）
CHINA_ADMIN_WHITELIST = [
    # 直辖市
    "北京","上海","天津","重庆",
    # 省
    "河北","山西","辽宁","吉林","黑龙江","江苏","浙江","安徽","福建","江西","山东",
    "河南","湖北","湖南","广东","海南","四川","贵州","云南","陕西","甘肃","青海",
    # 自治区
    "内蒙古","广西","西藏","宁夏","新疆",
    # 特别行政区/台湾
    "香港","澳门","台湾",
]

# 常见海外国家/地区关键词（用于硬排除）
FOREIGN_KEYWORDS = [
    "美国","英国","法国","德国","意大利","西班牙","葡萄牙","荷兰","比利时","瑞士","奥地利",
    "瑞典","挪威","芬兰","丹麦","爱尔兰","波兰","捷克","匈牙利","希腊","土耳其","俄罗斯",
    "加拿大","墨西哥","巴西","阿根廷","智利","秘鲁",
    "日本","韩国","朝鲜","新加坡","马来西亚","泰国","越南","菲律宾","印度","印尼","澳大利亚","新西兰",
    "阿联酋","沙特","卡塔尔","以色列","伊朗","伊拉克","埃及","南非","肯尼亚","尼日利亚",
    "usa","u.s.","uk","england","france","germany","italy","spain","portugal","netherlands","switzerland",
    "austria","sweden","norway","finland","denmark","russia","canada","mexico","brazil","argentina","chile",
    "japan","korea","singapore","malaysia","thailand","vietnam","philippines","india","indonesia","australia","new zealand",
    "uae","saudi","qatar","israel","iran","iraq","egypt","south africa",
]

# 常见海外城市中文名（用于硬排除）
FOREIGN_CITY_KEYWORDS = [
    "纽约","洛杉矶","旧金山","西雅图","芝加哥","波士顿","华盛顿","迈阿密","休斯顿","达拉斯",
    "波特兰","拉斯维加斯","圣何塞","圣地亚哥","费城","亚特兰大","底特律","明尼阿波利斯",
    "伦敦","巴黎","柏林","法兰克福","慕尼黑","苏黎世","日内瓦","阿姆斯特丹","布鲁塞尔",
    "马德里","巴塞罗那","罗马","米兰","维也纳","布拉格","布达佩斯","华沙",
    "斯德哥尔摩","奥斯陆","赫尔辛基","哥本哈根","雅典","伊斯坦布尔",
    "东京","大阪","名古屋","首尔","釜山","新加坡","吉隆坡","曼谷","河内","胡志明",
    "马尼拉","雅加达","悉尼","墨尔本","奥克兰",
    "多伦多","温哥华","蒙特利尔","卡尔加里",
    "迪拜","阿布扎比","多哈","利雅得",
    "开罗","内罗毕","约翰内斯堡","拉各斯",
    "圣保罗","里约","布宜诺斯艾利斯","墨西哥城","利马","圣地亚哥",
]

# 额外白名单（从非中国城市样本中人工/半自动确认）
CHINA_CITY_WHITELIST = set()

def load_city_whitelist(path: Path):
    """加载额外城市白名单（每行一个城市名）"""
    global CHINA_CITY_WHITELIST
    if path.exists():
        cities = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        CHINA_CITY_WHITELIST.update(cities)
        print(f"加载城市白名单: {path} ({len(cities)}条)")

def is_china_city(value: str) -> bool:
    """判断工作城市是否为中国境内（含港澳台）"""
    if pd.isna(value):
        return False
    s = str(value).strip().lower()
    if not s:
        return False
    # 命中人工/半自动白名单
    if s in CHINA_CITY_WHITELIST:
        return True
    # 硬排除海外国家关键词
    if any(k in s for k in FOREIGN_KEYWORDS):
        return False
    # 硬排除海外城市关键词
    if any(k in s for k in FOREIGN_CITY_KEYWORDS):
        return False
    # 命中省级行政区或港澳台直接保留
    if any(k in s for k in CHINA_ADMIN_WHITELIST):
        return True
    # 中文行政区划后缀（严格白名单近似）
    if re.search(r'(省|市|自治区|自治州|地区|盟|县|区|旗)', s) and re.search(r'[\u4e00-\u9fa5]', s):
        return True
    # 其余一律剔除
    return False

# 时间窗口定义（5个2年窗口）
TIME_WINDOWS = {
    "window_2016_2017": {
        "years": [2016, 2017],
        "label": "Window 1: 早期阶段 (2016-2017)",
        "output": WINDOWS_DIR / "window_2016_2017.csv"
    },
    "window_2018_2019": {
        "years": [2018, 2019],
        "label": "Window 2: 过渡阶段 (2018-2019)",
        "output": WINDOWS_DIR / "window_2018_2019.csv"
    },
    "window_2020_2021": {
        "years": [2020, 2021],
        "label": "Window 3: 数字化深化 (2020-2021)",
        "output": WINDOWS_DIR / "window_2020_2021.csv"
    },
    "window_2022_2023": {
        "years": [2022, 2023],
        "label": "Window 4: 转型关键期 (2022-2023)",
        "output": WINDOWS_DIR / "window_2022_2023.csv"
    },
    "window_2024_2025": {
        "years": [2024, 2025],
        "label": "Window 5: AI 2.0 爆发期 (2024-2025)",
        "output": WINDOWS_DIR / "window_2024_2025.csv"
    }
}


def run(input_csv: Path):
    print("=" * 60)
    print("按时间窗口分割数据")
    print("=" * 60)

    print(f"\n输入文件: {input_csv}")
    print(f"输出目录: {WINDOWS_DIR}")

    # 加载额外城市白名单（如果存在）
    whitelist_path = WINDOWS_DIR / "china_city_whitelist.txt"
    load_city_whitelist(whitelist_path)

    # 读取去重后的数据
    print("\n正在读取数据...")
    try:
        df = pd.read_csv(input_csv, low_memory=False)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    total_records = len(df)
    print(f"总记录数: {total_records:,}")

    # 检查年份列
    year_col = '招聘发布年份'
    if year_col not in df.columns:
        print(f"错误: 找不到列 '{year_col}'")
        print(f"可用列: {df.columns.tolist()}")
        return

    # 过滤非中国境内工作城市（含港澳台）
    city_col = '工作城市'
    if city_col in df.columns:
        before = len(df)
        mask_china = df[city_col].apply(is_china_city)
        non_china_df = df[~mask_china].copy()
        df = df[mask_china].copy()
        after = len(df)
        print(f"\n过滤非中国工作城市: 删除 {before - after:,} 条，保留 {after:,} 条")
        # 保存被剔除样本用于审查
        if len(non_china_df) > 0:
            excluded_path = WINDOWS_DIR / "non_china_cities.csv"
            non_china_df.to_csv(excluded_path, index=False, encoding='utf-8-sig')
            print(f"非中国城市样本已保存: {excluded_path}")
    else:
        print(f"\n警告: 找不到列 '{city_col}'，未执行境内城市过滤")

    # 转换年份为整数
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')

    # 显示年份分布
    print("\n年份分布:")
    year_counts = df[year_col].value_counts().sort_index()
    for year, count in year_counts.items():
        if pd.notna(year):
            print(f"  {int(year)}: {count:,} 条")

    # 按窗口分割并保存
    print("\n" + "=" * 60)
    print("开始分割...")
    print("=" * 60)

    stats = []

    for window_name, config in TIME_WINDOWS.items():
        years = config["years"]
        label = config["label"]
        output_path = config["output"]

        # 筛选数据
        mask = df[year_col].isin(years)
        window_df = df[mask].copy()
        count = len(window_df)
        pct = count / total_records * 100

        print(f"\n{label}")
        print(f"  年份: {years}")
        print(f"  记录数: {count:,} ({pct:.1f}%)")

        # 保存
        window_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  已保存: {output_path}")

        stats.append({
            "window": window_name,
            "years": str(years),
            "records": count,
            "percentage": f"{pct:.1f}%"
        })

    # 统计未纳入的数据
    all_window_years = []
    for config in TIME_WINDOWS.values():
        all_window_years.extend(config["years"])

    excluded_mask = ~df[year_col].isin(all_window_years)
    excluded_count = excluded_mask.sum()

    print("\n" + "=" * 60)
    print("分割完成！")
    print("=" * 60)

    # 汇总统计
    print("\n汇总统计:")
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))

    # 保存窗口切分统计，便于复现实验设置
    stats_file = WINDOWS_DIR / "window_split_stats.csv"
    stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"\n窗口统计已保存: {stats_file}")

    print(f"\n未纳入分析的数据: {excluded_count:,} 条")
    if excluded_count > 0:
        excluded_years = df.loc[excluded_mask, year_col].value_counts().sort_index()
        for year, count in excluded_years.items():
            if pd.notna(year):
                print(f"  {int(year)}: {count:,} 条")

    print(f"\n输出文件位置: {WINDOWS_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="按时间窗口切分去重后的招聘数据")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEDUPED_DATA,
        help="输入CSV路径（默认: config.paths.DEDUPED_DATA）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input_csv)
