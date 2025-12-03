import time
import numpy as np
from IT2FCM.border.f1.clustering.utility import extract_labels, distance_cdist
from IT2FCM.border.f1.clustering.cmeans.fcm import Dfcm


class DbFcm(Dfcm):
    def __init__(self, n_clusters: int, m: float = 2, epsilon: float = 1e-5, max_iter: int = 10000, index: int = 0, metric: str = 'euclidean'):
        self.local_data = None
        super().__init__(n_clusters=n_clusters, m=m, epsilon=epsilon, max_iter=max_iter, index=index, metric=metric)

    # Giai đoạn 1
    def state1_initial_centroids(self, data: np.ndarray) -> np.ndarray:
        self.local_data = data
        # Sắp xếp các điểm dữ liệu theo thứ tự tăng dần
        data = np.sort(data.reshape(-1))
        n = len(data)
        data = data.reshape(n, 1)
        # Gán một giá trị tâm cho mỗi cụm C bằng cách chọn k điểm dữ liệu được phân bố đều trên tập dữ liệu.
        Vs = data[np.floor(np.arange(self._n_clusters) * (n - 1) / (self._n_clusters - 1)).astype(int)]
        Vs = Vs.reshape(self._n_clusters, 1)
        # Áp dụng FCM để cập nhật ma trận thành viên và các centroid
        distances = distance_cdist(data, Vs)  # Khoảng cách Euclidean giữa data và centroids
        self.membership = self.calculate_membership(distances)
        self.centroids = self._update_centroids(data, self.membership)
        _labels = extract_labels(self.membership)
        # clusters = [list(data[_labels == i].reshape(-1)) for i in range(self._n_clusters)]
        return np.array([np.array(data[_labels == i].reshape(-1)) for i in range(self._n_clusters)], dtype=object)

    def __move_points_borders(self, vs: np.ndarray, clusters: np.ndarray, testid: int = 0) -> np.ndarray:
        if testid == 0:
            borders = (vs[:-1] + vs[1:]) / 2
        else:
            _ub = self.membership.sum(axis=0)
            if testid == 1:
                borders = (vs[:-1] * _ub[:-1] + vs[1:] * _ub[1:]) / (_ub[:-1] + _ub[1:])
            else:
                _ublm = _ub[:-1] ** self._m
                _ubrm = _ub[1:] ** self._m
                borders = (vs[:-1] * _ublm + vs[1:] * _ubrm) / (_ublm + _ubrm)
        # -----------------------------------
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

    # def __move_points_borders(self, vs: np.ndarray, membership: np.ndarray, clusters: np.ndarray, testid: int = 0) -> np.ndarray:
    #     borders = np.zeros(self._n_clusters - 1)
    #     for i in range(1, self._n_clusters):
    #         # Tính biên mờ dựa trên độ thành viên mờ (fuzzy membership), dùng trung bình trọng số của các centroids
    #         u_left = membership[:, i - 1]
    #         u_right = membership[:, i]
    #         # ulm, urm = u_left ** self._m, u_right ** self._m
    #         # fuzzy_border = (vs[i - 1] * ulm + vs[i] * urm) / (ulm + urm)
    #         fuzzy_border = (vs[i - 1] * u_left + vs[i] * u_right) / (u_left + u_right)
    #         borders[i - 1] = np.mean(fuzzy_border)
    #     # --------------------------------------
    #     lbs = list(borders)
    #     for i in range(1, self._n_clusters):
    #         left_cluster = clusters[i - 1]
    #         while left_cluster and left_cluster[-1] >= lbs[i - 1]:
    #             _point = left_cluster.pop()
    #             clusters[i].insert(0, _point)
    #         # -----------------
    #         right_cluster = clusters[i]
    #         while right_cluster and right_cluster[0] < lbs[i - 1]:
    #             _point = right_cluster.pop(0)
    #             clusters[i - 1].append(_point)
    #     # --------------------------------------
    #     return np.array([np.array(cluster).mean() if len(cluster) > 0 else 0 for cluster in clusters]).reshape(self._n_clusters, 1)

    # Giai đoạn 2: Tính toán các biên và cập nhật các cụm 
    def state2_final_centroids(self, clusters: np.ndarray, testid: int = 0) -> int:
        for step in range(self._max_iter):
            old_u = self.membership
            vs = self.centroids.reshape(-1)
            _v = self.__move_points_borders(vs, clusters, testid=testid)
            old_u = self.membership
            # --------------------------------------S
            distances = distance_cdist(self.local_data, _v)
            self.membership = self.calculate_membership(distances=distances)
            self.centroids = self._update_centroids(self.local_data, self.membership)

            # self.step = 1
            # return self.step
            if self.check_exit_by_membership(old_u):
                break
        self.step = step + 1
        return self.step

    def fit(self, data: np.ndarray, testid: int = 0) -> tuple:
        _start = time.time()
        _clusters = self.state1_initial_centroids(data=data)
        self.state2_final_centroids(clusters=_clusters, testid=testid)
        self.process_time = time.time() - _start
        return self.membership, self.centroids, self.step


if __name__ == '__main__':
    data = np.array([[1], [3], [7], [5], [2], [4], [9], [1], [4]])
    bf = DbFcm(n_clusters=3)
    bf.fit(data=data)
    print('step', bf.step)
    print('U', bf.membership.shape, bf.membership)
    print('V', bf.centroids.shape, bf.centroids)
# python nckh/f1/clustering/border/bfcm.py
