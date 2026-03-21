#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗脚本 - 分批处理13G数据
去除虚假、重复、不相关数据
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import hashlib


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, input_file: str, output_file: str, chunk_size: int = 10000):
        """
        初始化清洗器
        
        Args:
            input_file: 输入CSV文件路径
            output_file: 输出CSV文件路径
            chunk_size: 每批处理行数
        """
        self.input_file = input_file
        self.output_file = output_file
        self.chunk_size = chunk_size
        
        # 被过滤数据的输出文件
        import os
        output_dir = os.path.dirname(output_file)
        self.fake_file = os.path.join(output_dir, 'removed_fake.csv')
        self.duplicate_file = os.path.join(output_dir, 'removed_duplicates.csv')
        self.irrelevant_file = os.path.join(output_dir, 'removed_irrelevant.csv')
        
        # 用于去重的哈希集合
        self.seen_hashes = set()
        
        # 统计信息
        self.stats = {
            'total': 0,
            'duplicates': 0,
            'fake': 0,
            'irrelevant': 0,
            'valid': 0
        }
        
        print(f"数据清洗器初始化")
        print(f"输入: {input_file}")
        print(f"输出: {output_file}")
        print(f"虚假数据记录: {self.fake_file}")
        print(f"重复数据记录: {self.duplicate_file}")
        print(f"不相关数据记录: {self.irrelevant_file}")
        print(f"批大小: {chunk_size:,} 行/批")
    
    def is_fake_data(self, row) -> bool:
        """
        检测虚假数据
        
        Args:
            row: 数据行
            
        Returns:
            是否为虚假数据
        """
        # 1. 检查企业名称
        if pd.isna(row['企业名称']) or row['企业名称'].strip() == '':
            return True
        
        # 2. 检查招聘岗位
        if pd.isna(row['招聘岗位']) or row['招聘岗位'].strip() == '':
            return True
        
        # 3. 检查工作城市
        if pd.isna(row['工作城市']) or row['工作城市'].strip() == '':
            return True
        
        # 4. 检查薪资异常（过高或过低）
        try:
            min_salary = float(row['最低月薪']) if pd.notna(row['最低月薪']) else 0
            max_salary = float(row['最高月薪']) if pd.notna(row['最高月薪']) else 0
            
            # 月薪低于500或高于100万视为异常
            if min_salary > 0 and (min_salary < 500 or min_salary > 1000000):
                return True
            if max_salary > 0 and (max_salary < 500 or max_salary > 1000000):
                return True
            
            # 最低薪资大于最高薪资
            if min_salary > 0 and max_salary > 0 and min_salary > max_salary:
                return True
        except:
            pass
        
        # 5. 检查职位描述长度（太短可能是虚假信息）
        if pd.notna(row['职位描述']):
            desc = str(row['职位描述']).strip()
            if len(desc) < 10:  # 职位描述少于10个字符
                return True
        else:
            return True
        
        # 6. 检查日期异常
        try:
            year = int(row['招聘发布年份']) if pd.notna(row['招聘发布年份']) else 0
            if year > 0 and (year < 2010 or year > 2026):
                return True
        except:
            pass
        
        return False
    
    def is_irrelevant_data(self, row) -> bool:
        """
        检测不相关数据
        
        Args:
            row: 数据行
            
        Returns:
            是否为不相关数据
        """
        # 1. 检查非中国大陆地区（如果有）
        city = str(row['工作城市']).strip() if pd.notna(row['工作城市']) else ''
        
        # 排除港澳台及国外城市（根据需要调整）
        exclude_cities = ['香港', '澳门', '台湾', '台北', '高雄']
        for exc in exclude_cities:
            if exc in city:
                return True
        
        # 2. 检查招聘类别是否相关
        category = str(row['招聘类别']).strip() if pd.notna(row['招聘类别']) else ''
        irrelevant_categories = ['兼职', '实习', '志愿者']
        if category in irrelevant_categories:
            return True
        
        # 3. 检查职位描述中的垃圾信息
        if pd.notna(row['职位描述']):
            desc = str(row['职位描述']).lower()
            spam_keywords = [
                '微商', '代理', '刷单', '打字员', '兼职', 
                '无需经验', '日赚', '轻松', '在家', '手机',
                '加微信', 'qq', '联系电话'
            ]
            spam_count = sum(1 for kw in spam_keywords if kw in desc)
            if spam_count >= 2:  # 包含2个以上垃圾关键词
                return True
        
        return False
    
    def create_row_hash(self, row) -> str:
        """
        为行创建唯一哈希值（用于去重）
        
        Args:
            row: 数据行
            
        Returns:
            哈希值
        """
        # 使用关键字段组合生成哈希
        key_fields = [
            str(row.get('企业名称', '')),
            str(row.get('招聘岗位', '')),
            str(row.get('工作城市', '')),
            str(row.get('职位描述', ''))[:100],  # 职位描述前100字符
            str(row.get('招聘发布日期', ''))
        ]
        
        combined = '|'.join(key_fields)
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def clean_data(self):
        """执行数据清洗"""
        print("\n" + "="*80)
        print("开始数据清洗...")
        print("="*80)
        
        # 判断输入文件行数
        print("\n正在统计总行数...")
        total_lines = sum(1 for _ in open(self.input_file, 'r', encoding='utf-8'))
        print(f"总计: {total_lines:,} 行")
        
        # 标记是否第一次写入（需要写入表头）
        first_write = True
        first_write_fake = True
        first_write_duplicate = True
        first_write_irrelevant = True
        
        # 分批读取和处理
        print(f"\n开始分批处理（每批 {self.chunk_size:,} 行）...")
        
        chunk_iter = pd.read_csv(
            self.input_file,
            chunksize=self.chunk_size,
            encoding='utf-8',
            low_memory=False
        )
        
        for chunk_num, chunk in enumerate(tqdm(chunk_iter, total=total_lines//self.chunk_size)):
            self.stats['total'] += len(chunk)
            
            # 存储清洗后的数据
            clean_rows = []
            fake_rows = []
            duplicate_rows = []
            irrelevant_rows = []
            
            for idx, row in chunk.iterrows():
                # 1. 检查虚假数据
                if self.is_fake_data(row):
                    self.stats['fake'] += 1
                    fake_rows.append(row)
                    continue
                
                # 2. 检查不相关数据
                if self.is_irrelevant_data(row):
                    self.stats['irrelevant'] += 1
                    irrelevant_rows.append(row)
                    continue
                
                # 3. 检查重复数据
                row_hash = self.create_row_hash(row)
                if row_hash in self.seen_hashes:
                    self.stats['duplicates'] += 1
                    duplicate_rows.append(row)
                    continue
                
                # 添加到去重集合
                self.seen_hashes.add(row_hash)
                
                # 4. 数据有效，保留
                clean_rows.append(row)
                self.stats['valid'] += 1
            
            # 将清洗后的数据写入输出文件
            if clean_rows:
                clean_df = pd.DataFrame(clean_rows)
                
                if first_write:
                    clean_df.to_csv(
                        self.output_file,
                        index=False,
                        encoding='utf-8-sig',
                        mode='w'
                    )
                    first_write = False
                else:
                    clean_df.to_csv(
                        self.output_file,
                        index=False,
                        encoding='utf-8-sig',
                        mode='a',
                        header=False
                    )
            
            # 保存虚假数据
            if fake_rows:
                fake_df = pd.DataFrame(fake_rows)
                if first_write_fake:
                    fake_df.to_csv(self.fake_file, index=False, encoding='utf-8-sig', mode='w')
                    first_write_fake = False
                else:
                    fake_df.to_csv(self.fake_file, index=False, encoding='utf-8-sig', mode='a', header=False)
            
            # 保存重复数据
            if duplicate_rows:
                dup_df = pd.DataFrame(duplicate_rows)
                if first_write_duplicate:
                    dup_df.to_csv(self.duplicate_file, index=False, encoding='utf-8-sig', mode='w')
                    first_write_duplicate = False
                else:
                    dup_df.to_csv(self.duplicate_file, index=False, encoding='utf-8-sig', mode='a', header=False)
            
            # 保存不相关数据
            if irrelevant_rows:
                irr_df = pd.DataFrame(irrelevant_rows)
                if first_write_irrelevant:
                    irr_df.to_csv(self.irrelevant_file, index=False, encoding='utf-8-sig', mode='w')
                    first_write_irrelevant = False
                else:
                    irr_df.to_csv(self.irrelevant_file, index=False, encoding='utf-8-sig', mode='a', header=False)
            
            # 定期输出进度
            if (chunk_num + 1) % 10 == 0:
                print(f"\n进度更新 (批次 {chunk_num + 1}):")
                print(f"  已处理: {self.stats['total']:,} 行")
                print(f"  有效数据: {self.stats['valid']:,} 行 ({self.stats['valid']/self.stats['total']*100:.1f}%)")
                print(f"  重复: {self.stats['duplicates']:,} 行")
                print(f"  虚假: {self.stats['fake']:,} 行")
                print(f"  不相关: {self.stats['irrelevant']:,} 行")
        
        # 最终统计
        print("\n" + "="*80)
        print("数据清洗完成！")
        print("="*80)
        print(f"总计处理: {self.stats['total']:,} 行")
        print(f"✓ 有效数据: {self.stats['valid']:,} 行 ({self.stats['valid']/self.stats['total']*100:.1f}%)")
        print(f"✗ 重复数据: {self.stats['duplicates']:,} 行 ({self.stats['duplicates']/self.stats['total']*100:.1f}%)")
        print(f"✗ 虚假数据: {self.stats['fake']:,} 行 ({self.stats['fake']/self.stats['total']*100:.1f}%)")
        print(f"✗ 不相关数据: {self.stats['irrelevant']:,} 行 ({self.stats['irrelevant']/self.stats['total']*100:.1f}%)")
        print(f"\n输出文件: {self.output_file}")
        
        # 输出文件大小
        import os
        if os.path.exists(self.output_file):
            size_mb = os.path.getsize(self.output_file) / 1024 / 1024
            print(f"输出大小: {size_mb:.1f} MB")
        
        print(f"\n被过滤数据记录:")
        if os.path.exists(self.fake_file):
            size_mb = os.path.getsize(self.fake_file) / 1024 / 1024
            print(f"  虚假数据: {self.fake_file} ({size_mb:.1f} MB)")
        if os.path.exists(self.duplicate_file):
            size_mb = os.path.getsize(self.duplicate_file) / 1024 / 1024
            print(f"  重复数据: {self.duplicate_file} ({size_mb:.1f} MB)")
        if os.path.exists(self.irrelevant_file):
            size_mb = os.path.getsize(self.irrelevant_file) / 1024 / 1024
            print(f"  不相关数据: {self.irrelevant_file} ({size_mb:.1f} MB)")


def main():
    """主函数"""
    print("="*80)
    print("招聘数据清洗工具")
    print("="*80)

    # 使用集中路径配置
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.paths import RAW_ALL_IN_ONE, CLEANED_DATA, CLEANED_DIR

    # 确保输出目录存在
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    input_file = str(RAW_ALL_IN_ONE)
    output_file = str(CLEANED_DATA)
    
    print(f"\n配置:")
    print(f"  输入文件: {input_file}")
    print(f"  输出文件: {output_file}")
    print(f"  批处理大小: 10,000 行/批")
    
    print(f"\n清洗规则:")
    print(f"  1. 去除虚假数据（缺失关键字段、异常薪资、职位描述过短）")
    print(f"  2. 去除重复数据（基于企业+岗位+描述的哈希）")
    print(f"  3. 去除不相关数据（兼职、港澳台、垃圾信息）")
    
    print(f"\n预计:")
    print(f"  总数据: 13,492,246 行 (13GB)")
    print(f"  处理时间: 约2-3小时")
    print(f"  预期保留率: 60-80%")
    
    confirm = input("\n确认开始清洗? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        cleaner = DataCleaner(input_file, output_file, chunk_size=10000)
        cleaner.clean_data()
    else:
        print("已取消")


if __name__ == "__main__":
    main()
