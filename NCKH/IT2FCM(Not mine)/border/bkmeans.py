import time
import numpy as np
from f1.clustering.kmeans import Dkmeans


class DbKmeans(Dkmeans):

    def __init__(self, n_clusters: int):
        self.data = None
        super().__init__(n_clusters=n_clusters)

    # Giai đoạn 1
    def state1_initial_centroids(self, data: np.ndarray) -> tuple:
        self.data = data

        # Sắp xếp các điểm dữ liệu theo thứ tự tăng dần
        data = np.sort(data.reshape(-1))
        n = len(data)
        data = data.reshape(n, 1)

        # Gán một giá trị tâm cho mỗi cụm C bằng cách chọn k điểm dữ liệu được phân bố đều trên tập dữ liệu.
        Vs = data[np.floor(np.arange(self._n_clusters) * (n - 1) / (self._n_clusters - 1)).astype(int)]
        Vs = Vs.reshape(self._n_clusters, 1)
        _labels = self._update_labels(data, Vs)
        centroids = self._update_centroids(data, _labels)
        clusters = np.array([np.array(data[_labels == i].reshape(-1)) for i in range(self._n_clusters)], dtype=object)
        return centroids, clusters

    def __move_points_borders(self, vs: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        borders = (vs[:-1] + vs[1:]) / 2
        borders = borders[:self._n_clusters-1]

        # Duyệt qua các cluster và sắp xếp lại các điểm theo các giá trị 'borders'
        for i in range(1, self._n_clusters):
            left_cluster, right_cluster = clusters[i - 1], clusters[i]

            # Chuyển các phần tử từ cluster bên trái sang cluster bên phải nếu cần
            mask_right = left_cluster >= borders[i - 1]
            clusters[i] = np.concatenate((left_cluster[mask_right], right_cluster))
            clusters[i - 1] = left_cluster[~mask_right]

            # Chuyển các phần tử từ cluster bên phải sang cluster bên trái nếu cần
            mask_left = clusters[i] < borders[i - 1]
            clusters[i - 1] = np.concatenate((clusters[i - 1], clusters[i][mask_left]))
            clusters[i] = clusters[i][~mask_left]

        # Tính giá trị trung bình của mỗi cluster và đưa kết quả vào mảng _v
        return np.array([cluster.mean() if len(cluster) > 0 else 0 for cluster in clusters]).reshape(self._n_clusters, 1)

    # Giai đoạn 2: Tính toán các biên và cập nhật các cụm mờ
    def state2_final_centroids(self, centroids: np.ndarray, clusters: np.ndarray):
        vs = centroids.reshape(-1)
        _v = self.__move_points_borders(vs, clusters)
        # --------------------------------------
        self.labels = self._update_labels(self.data, _v)
        self.centroids = self._update_centroids(self.data, self.labels)

    def fit(self, data: np.ndarray) -> tuple:
        self.local_data = data
        _start = time.time()
        _centroids, clusters = self.state1_initial_centroids(data=data)
        self.state2_final_centroids(centroids=_centroids, clusters=clusters)
        self.process_time = time.time() - _start
        self.step = 1
        return self.labels, self.centroids


if __name__ == '__main__':
    data = np.array([[1], [3], [7], [5], [2], [4], [9], [1], [4]])
    bf = DbKmeans(n_clusters=3)
    l, c = bf.fit(data=data)
    print('KQ l', l.shape, l)
    print('KQ c', c.shape, c)
    
# python nckh/f1/clustering/border/bkmeans.py
