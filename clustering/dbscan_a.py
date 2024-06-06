from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix
import numpy as np

class DBSCANClusterer:
    def __init__(self, eps=0.3, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
    
    def fit_predict(self, data):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return dbscan.fit_predict(data)
    
    # def group_data_within_clusters(self, data, clusters, max_group_size=5):
    #     cluster_labels = np.unique(clusters)
    #     index_mapping = np.full(len(data), -1, dtype=int)
    #     group_id = 0

    #     for cluster_label in cluster_labels:
    #         if cluster_label == -1:
    #             continue
            
    #         cluster_indices = np.where(clusters == cluster_label)[0]
    #         if len(cluster_indices) == 0:
    #             continue

    #         distances = euclidean_distances(data[cluster_indices], data[cluster_indices])
    #         np.fill_diagonal(distances, np.inf)

    #         assigned = np.array([False] * len(cluster_indices))  # 그룹에 할당된 포인트를 추적하기 위한 배열

    #         for i in range(len(cluster_indices)):
    #             if assigned[i]:
    #                 continue  # 이미 그룹에 할당된 경우 건너뜁니다.
                
    #             group_indices = [i]  # 현재 포인트로 새 그룹을 시작합니다.
    #             assigned[i] = True  # 현재 포인트를 할당으로 표시합니다.

    #             while len(group_indices) < max_group_size:
    #                 if assigned.all():  # 모든 포인트가 할당된 경우
    #                     break
                    
    #                 # 할당되지 않은 포인트에 대한 거리만 고려합니다.
    #                 dists_to_group = np.min(distances[np.ix_(group_indices, ~assigned)], axis=0)
    #                 if len(dists_to_group) == 0:
    #                     break  # 더 이상 할당할 수 있는 포인트가 없습니다.

    #                 closest_point_idx = np.argmin(dists_to_group)
    #                 closest_unassigned_idx = np.where(~assigned)[0][closest_point_idx]

    #                 group_indices.append(closest_unassigned_idx)
    #                 assigned[closest_unassigned_idx] = True

    #             for idx in group_indices:
    #                 index_mapping[cluster_indices[idx]] = group_id
    #             group_id += 1

    #     return index_mapping
    
    
    def group_data_within_clusters_2(self, data, clusters, max_group_size=5):
        cluster_labels = np.unique(clusters)  # DBSCAN에서 생성된 고유한 클러스터 라벨을 추출합니다.
        index_mapping = np.full(len(data), -1, dtype=int)  # 각 데이터 포인트의 소그룹 ID를 저장할 배열을 초기화합니다.
        group_id = 0  # 소그룹 ID를 초기화합니다.

        for cluster_label in cluster_labels:
            if cluster_label == -1: continue  # 노이즈 데이터(-1 라벨)는 처리하지 않습니다.
            cluster_indices = np.where(clusters == cluster_label)[0]  # 특정 클러스터에 속하는 데이터 포인트의 인덱스를 찾습니다.
            cluster_data = data[cluster_indices]  # 해당 클러스터의 데이터를 추출합니다.
            
            if len(cluster_data) < max_group_size:
                continue  # 그룹 크기가 충분하지 않으면 소그룹 형성을 건너뜁니다.
            
            # 클러스터 중심에서 가장 먼 점을 찾습니다.
            cluster_center = np.mean(cluster_data, axis=0)
            distances_to_center = np.linalg.norm(cluster_data - cluster_center, axis=1)
            edge_point_index = np.argmax(distances_to_center)
            
            # 유클리디안 거리를 계산하여 그 점에서 가장 가까운 점들을 그룹화합니다.
            distances = np.linalg.norm(cluster_data - cluster_data[edge_point_index], axis=1)
            sorted_indices = np.argsort(distances)
            
            assigned_group = np.full(len(cluster_data), -1)  # 클러스터 내 데이터 포인트의 그룹 할당 상태를 초기화합니다.

            i = 0
            while i < len(sorted_indices):
                if assigned_group[sorted_indices[i]] == -1:  # 아직 그룹이 할당되지 않은 경우
                    current_group_members = sorted_indices[i:i + max_group_size]
                    assigned_group[current_group_members] = group_id
                    group_id += 1  # 새 소그룹을 시작하기 위해 소그룹 ID를 증가시킵니다.
                i += max_group_size
            
            # 최종적으로, 각 데이터 포인트의 소그룹 ID를 전체 데이터셋에 대한 인덱스 매핑으로 업데이트합니다.
            for i, idx in enumerate(cluster_indices):
                index_mapping[idx] = assigned_group[i]

        return index_mapping


    # def apply_algorithm_to_clusters(self, data):
    #     cluster_labels = self.fit_predict(data)
    #     unique_labels = np.unique(cluster_labels)
    #     results = {}
    #     for label in unique_labels:
    #         if label == -1:
    #             continue  # Skip noise
    #         cluster_data = data[cluster_labels == label]
    #         cluster_size = max(5, len(cluster_data) // 5)  # Example cluster size calculation
    #         results[label] = self.genetic_algorithm(cluster_data, cluster_size)
    #     return results

