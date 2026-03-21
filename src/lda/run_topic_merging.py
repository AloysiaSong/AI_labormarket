# 文件: task3_modeling/t3_4_topic_merging/run_topic_merging.py

"""
主题合并主执行脚本
根据项目具体情况实现相似主题合并功能
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from gensim.models import LdaModel
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TopicMerger:
    def __init__(self, base_path=None):
        if base_path is None:
            # 使用脚本所在目录的相对路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(script_dir, "../../")
        self.base_path = base_path
        self.windows = ['window_2016_2017', 'window_2018_2019', 'window_2020_2021', 'window_2022_2023', 'window_2024_2025']
        self.similarity_threshold = 0.75  # 相似度阈值
        self.min_topics_per_window = 20   # 每个窗口最小主题数
        self.max_cluster_size = 5         # 最大簇大小

    def load_topic_vectors(self, window):
        """加载主题向量"""
        vector_path = f"{self.base_path}output/lda/topic_vectors/{window}_topic_vectors.npy"
        return np.load(vector_path)

    def load_lda_model(self, window):
        """加载LDA模型"""
        model_path = f"{self.base_path}output/lda/models/{window}_lda.model"
        return LdaModel.load(model_path)

    def compute_similarity_matrix(self, topic_vectors):
        """计算主题相似度矩阵"""
        return cosine_similarity(topic_vectors)

    def determine_optimal_clusters(self, similarity_matrix):
        """基于相似度阈值确定最佳聚类数量"""
        n_topics = similarity_matrix.shape[0]

        # 计算需要合并的主题对
        merge_pairs = []
        for i in range(n_topics):
            for j in range(i+1, n_topics):
                if similarity_matrix[i,j] > self.similarity_threshold:
                    merge_pairs.append((i, j, similarity_matrix[i,j]))

        # 使用贪心算法确定合并组
        clusters = [[i] for i in range(n_topics)]

        # 按相似度降序排序合并
        merge_pairs.sort(key=lambda x: x[2], reverse=True)

        for topic_i, topic_j, sim in merge_pairs:
            # 找到包含这两个主题的簇
            cluster_i = None
            cluster_j = None
            for idx, cluster in enumerate(clusters):
                if topic_i in cluster:
                    cluster_i = idx
                if topic_j in cluster:
                    cluster_j = idx

            # 如果在不同簇中且簇大小不超过限制，则合并
            if cluster_i != cluster_j and len(clusters[cluster_i]) + len(clusters[cluster_j]) <= self.max_cluster_size:
                clusters[cluster_i].extend(clusters[cluster_j])
                clusters.pop(cluster_j)

        # 确保不低于最小主题数
        while len(clusters) < self.min_topics_per_window and len(merge_pairs) > 0:
            # 如果主题数太少，降低阈值继续合并
            self.similarity_threshold *= 0.95
            logger.info(f"降低相似度阈值至 {self.similarity_threshold:.3f} 以达到最小主题数要求")

            # 重新计算合并对
            merge_pairs = [(i,j,sim) for i,j,sim in merge_pairs if sim > self.similarity_threshold]
            # 重新聚类 (简化版)
            clusters = self._recluster_with_lower_threshold(similarity_matrix, merge_pairs)

        return clusters

    def _recluster_with_lower_threshold(self, similarity_matrix, merge_pairs):
        """使用更低阈值重新聚类"""
        n_topics = similarity_matrix.shape[0]
        clusters = [[i] for i in range(n_topics)]

        for topic_i, topic_j, sim in merge_pairs:
            cluster_i = None
            cluster_j = None
            for idx, cluster in enumerate(clusters):
                if topic_i in cluster:
                    cluster_i = idx
                if topic_j in cluster:
                    cluster_j = idx

            if cluster_i != cluster_j:
                clusters[cluster_i].extend(clusters[cluster_j])
                clusters.pop(cluster_j)

        return clusters

    def merge_topic_vectors(self, topic_vectors, clusters):
        """合并主题向量"""
        merged_vectors = []
        cluster_mapping = {}

        for new_topic_id, cluster in enumerate(clusters):
            # 计算簇内主题向量的平均值
            cluster_vectors = topic_vectors[cluster]
            merged_vector = np.mean(cluster_vectors, axis=0)

            # L2归一化
            merged_vector = merged_vector / np.linalg.norm(merged_vector)

            merged_vectors.append(merged_vector)

            # 记录映射关系
            for old_topic_id in cluster:
                cluster_mapping[old_topic_id] = new_topic_id

        return np.array(merged_vectors), cluster_mapping

    def merge_topic_labels(self, model, clusters):
        """合并主题标签"""
        merged_labels = {}

        for new_topic_id, cluster in enumerate(clusters):
            # 收集簇内所有主题的关键词
            all_keywords = []
            keyword_weights = {}

            for old_topic_id in cluster:
                top_words = model.show_topic(old_topic_id, topn=5)
                for word, weight in top_words:
                    if word not in keyword_weights:
                        keyword_weights[word] = 0
                    keyword_weights[word] += weight

            # 选择权重最高的关键词
            sorted_keywords = sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True)
            top_keywords = [word for word, _ in sorted_keywords[:3]]

            merged_labels[new_topic_id] = f"Merged_{new_topic_id}: {' '.join(top_keywords)}"

        return merged_labels

    def process_window(self, window):
        """处理单个时间窗口"""
        logger.info(f"处理窗口: {window}")

        # 加载数据
        topic_vectors = self.load_topic_vectors(window)
        lda_model = self.load_lda_model(window)

        # 计算相似度矩阵
        similarity_matrix = self.compute_similarity_matrix(topic_vectors)

        # 确定聚类
        clusters = self.determine_optimal_clusters(similarity_matrix)

        # 合并主题向量
        merged_vectors, cluster_mapping = self.merge_topic_vectors(topic_vectors, clusters)

        # 合并主题标签
        merged_labels = self.merge_topic_labels(lda_model, clusters)

        # 保存结果
        self.save_window_results(window, merged_vectors, cluster_mapping, merged_labels, clusters)

        logger.info(f"{window}: {len(topic_vectors)} → {len(merged_vectors)} 主题")

        return {
            'original_count': len(topic_vectors),
            'merged_count': len(merged_vectors),
            'clusters': clusters,
            'mapping': cluster_mapping
        }

    def save_window_results(self, window, merged_vectors, cluster_mapping, merged_labels, clusters):
        """保存窗口处理结果"""
        # 创建输出目录
        output_dir = f"{self.base_path}output/lda/merged_topic_vectors"
        os.makedirs(output_dir, exist_ok=True)

        # 保存合并后的向量
        np.save(f"{output_dir}/{window}_merged_vectors.npy", merged_vectors)

        # 保存标签
        with open(f"{output_dir}/{window}_merged_labels.pkl", 'wb') as f:
            pickle.dump(merged_labels, f)

        # 保存簇信息
        cluster_info = {
            'clusters': clusters,
            'mapping': cluster_mapping,
            'labels': merged_labels
        }

        with open(f"{output_dir}/{window}_cluster_info.pkl", 'wb') as f:
            pickle.dump(cluster_info, f)

    def generate_merge_report(self, results):
        """生成合并报告"""
        report_data = []
        total_original = 0
        total_merged = 0

        for window, result in results.items():
            reduction_rate = (result['original_count'] - result['merged_count']) / result['original_count'] * 100

            report_data.append({
                'window': window,
                'original_topics': result['original_count'],
                'merged_topics': result['merged_count'],
                'reduction_rate': reduction_rate,
                'clusters': len(result['clusters'])
            })

            total_original += result['original_count']
            total_merged += result['merged_count']

        # 保存详细报告
        df = pd.DataFrame(report_data)
        df.to_csv(f"{self.base_path}output/lda/topic_merge_report.csv", index=False)

        # 总体统计
        overall_reduction = (total_original - total_merged) / total_original * 100

        logger.info("=== 主题合并总体报告 ===")
        logger.info(f"总原始主题数: {total_original}")
        logger.info(f"总合并主题数: {total_merged}")
        logger.info(f"总体减少率: {overall_reduction:.1f}%")

        return df

    def run(self):
        """主执行函数"""
        logger.info("🎯 开始主题合并处理")
        logger.info(f"相似度阈值: {self.similarity_threshold}")
        logger.info(f"最小主题数: {self.min_topics_per_window}")

        results = {}

        for window in self.windows:
            try:
                result = self.process_window(window)
                results[window] = result
            except Exception as e:
                logger.error(f"处理 {window} 失败: {e}")
                continue

        # 生成合并报告
        self.generate_merge_report(results)

        # 保存全局映射
        self.save_global_mapping(results)

        logger.info("✅ 主题合并处理完成")

    def save_global_mapping(self, results):
        """保存全局主题映射"""
        global_mapping = {}
        for window, result in results.items():
            global_mapping[window] = result['mapping']

        with open(f"{self.base_path}output/lda/topic_merge_mapping.pkl", 'wb') as f:
            pickle.dump(global_mapping, f)

        logger.info("💾 全局主题映射已保存")

if __name__ == "__main__":
    merger = TopicMerger()
    merger.run()