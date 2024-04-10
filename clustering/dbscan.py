from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

class DBSCANClusterer:
    def __init__(self, eps=0.3, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
    
    def fit_predict(self, data):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return dbscan.fit_predict(data)
    
    def group_data_within_clusters(self, data, clusters, max_group_size=5):
        cluster_labels = np.unique(clusters)
        index_mapping = np.full(len(data), -1, dtype=int)
        group_id = 0

        for cluster_label in cluster_labels:
            if cluster_label == -1:
                continue
            
            cluster_indices = np.where(clusters == cluster_label)[0]
            if len(cluster_indices) == 0:
                continue

            distances = euclidean_distances(data[cluster_indices], data[cluster_indices])
            np.fill_diagonal(distances, np.inf)

            assigned = np.array([False] * len(cluster_indices))  # 그룹에 할당된 포인트를 추적하기 위한 배열

            for i in range(len(cluster_indices)):
                if assigned[i]:
                    continue  # 이미 그룹에 할당된 경우 건너뜁니다.
                
                group_indices = [i]  # 현재 포인트로 새 그룹을 시작합니다.
                assigned[i] = True  # 현재 포인트를 할당으로 표시합니다.

                while len(group_indices) < max_group_size:
                    if assigned.all():  # 모든 포인트가 할당된 경우
                        break
                    
                    # 할당되지 않은 포인트에 대한 거리만 고려합니다.
                    dists_to_group = np.min(distances[np.ix_(group_indices, ~assigned)], axis=0)
                    if len(dists_to_group) == 0:
                        break  # 더 이상 할당할 수 있는 포인트가 없습니다.

                    closest_point_idx = np.argmin(dists_to_group)
                    closest_unassigned_idx = np.where(~assigned)[0][closest_point_idx]

                    group_indices.append(closest_unassigned_idx)
                    assigned[closest_unassigned_idx] = True

                for idx in group_indices:
                    index_mapping[cluster_indices[idx]] = group_id
                group_id += 1

        return index_mapping
