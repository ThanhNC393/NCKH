import numpy as np


class DbKmv1:
    def __init__(self, n_clusters: int):
        self._n_clusters = n_clusters

    # State 1
    def state1_initial_centroids(self, data: np.ndarray) -> tuple:
        # Sắp xếp các điểm dữ liệu theo thứ tự tăng dần
        data = np.sort(data.reshape(-1)).reshape(-1, 1)
        n = data.shape[0]

        # Gán một giá trị tâm cho mỗi cụm C bằng cách chọn k điểm dữ liệu được phân bố đều trên tập dữ liệu.
        Vs = data[(np.floor(np.arange(self._n_clusters) * (n - 1) / (self._n_clusters - 1))).astype(int)].reshape(self._n_clusters, 1)

        # Gán các điểm dữ liệu với cụm gần nhất
        distances = np.abs(data - Vs.T)
        closest_clusters = np.argmin(distances, axis=1)

        # Tính S là tổng các điểm dữ liệu trong một cụm và N là số lượng điểm dữ liệu trong một cụm
        S = np.array([data[closest_clusters == i].sum() for i in range(self._n_clusters)]).reshape(-1, 1)
        N = np.array([(closest_clusters == i).sum() for i in range(self._n_clusters)]).reshape(-1, 1)

        # Cập nhật tâm cụm dựa trên giá trị trung bình của các điểm dữ liệu trong một cụm
        for i in range(self._n_clusters):
            if N[i] > 0:
                Vs[i] = S[i] / N[i]

        clusters = [list(data[closest_clusters == i].reshape(-1)) for i in range(self._n_clusters)]

        # Vs: tâm các cụm
        # S: Tổng các điểm dữ liệu trong một cụm
        # N: Số điểm dữ liệu trong một cụm
        # clusters: Các điểm dữ liệu trong một cụm
        return Vs, S, N, clusters

    # State 2
    def state2_final_centroids(self, C: np.ndarray, S: np.ndarray, N: np.ndarray, clusters: list) -> tuple:
        # Tính toán các biên
        borders = (C[:-1] + C[1:]) / 2

        moved = False
        for i in range(1, self._n_clusters):
            left_cluster = clusters[i - 1]
            right_cluster = clusters[i]

            # Di chuyển các điểm từ left_cluster sang right_cluster
            while left_cluster and left_cluster[-1] >= borders[i - 1]:
                point = left_cluster.pop()
                right_cluster.insert(0, point)
                S[i - 1] -= point
                N[i - 1] -= 1
                S[i] += point
                N[i] += 1
                moved = True

            # Di chuyển các điểm từ right_cluster sang left_cluster
            while right_cluster and right_cluster[0] < borders[i - 1]:
                point = right_cluster.pop(0)
                left_cluster.append(point)
                S[i] -= point
                N[i] -= 1
                S[i - 1] += point
                N[i - 1] += 1
                moved = True

            # Cập nhật lại clusters sau khi di chuyển
            clusters[i - 1] = left_cluster
            clusters[i] = right_cluster

            # Nếu không có điểm nào di chuyển, thoát khỏi vòng lặp
            if not moved:
                break

        # Cập nhật lại centroids
        for i in range(self._n_clusters):
            if N[i] > 0:
                C[i] = S[i] / N[i]
        return C, clusters, borders.tolist()

    def fit(self, data: np.ndarray):
        C_intial, S, N, clusters_inital = self.state1_initial_centroids(data)
        C_final, clusters_final, borders = self.state2_final_centroids(C_intial, S, N, clusters_inital)
        self.centroids = C_final
        return C_final, clusters_final, borders
