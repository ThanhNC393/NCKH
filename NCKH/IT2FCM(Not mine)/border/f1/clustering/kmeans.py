import time
import numpy as np
from f1.clustering.utility import distance_cdist


class Dkmeans():

    def __init__(self, n_clusters: int, epsilon: float = 1e-5, max_iter: int = 10000, metric: str = 'euclidean'):
        self._metric = metric
        self._n_clusters = n_clusters
        self._epsilon = epsilon
        self._max_iter = max_iter

        self.labels = None
        self.centroids = None

        self.process_time = 0
        self.step = 0

    def _update_labels(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        distances = distance_cdist(data, centroids, metric=self._metric)
        return np.argmin(distances, axis=1)  # Tim x khi f(x) min

    def _update_centroids(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.array([data[labels == k].mean(axis=0) if np.any(labels == k) else np.zeros(data.shape[1]) for k in range(self._n_clusters)])
        # return np.array([data[labels == k].mean(axis=0) for k in range(self._n_clusters)])

    def fit(self, data: np.ndarray, seed: int = None) -> tuple:
        self.local_data = data
        _start_tm = time.time()
        if seed is not None:
            np.random.seed(seed=seed)
        # Khởi tạo tâm cụm ngẫu nhiên
        self.centroids = data[np.random.choice(len(data), self._n_clusters, replace=False)]
        for step in range(self._max_iter):
            # Gán nhãn cho các điểm dữ liệu
            self.labels = self._update_labels(data, self.centroids)
            # Cập nhật tâm cụm
            new_centroids = self._update_centroids(data, self.labels)
            # Kiểm tra hội tụ
            if np.sum((new_centroids - self.centroids) ** 2) < self._epsilon:
                break
            self.centroids = new_centroids
        self.process_time = time.time() - _start_tm
        self.step = step + 1
        return self.labels, self.centroids, self.step


if __name__ == '__main__':
    import os
    from f1.clustering.dataset import fetch_data_from_local, TEST_CASES
    from f1.clustering.utility import round_float
    from f1.clustering.validity import dunn, davies_bouldin, calinski_harabasz, silhouette
    from sklearn.cluster import KMeans

    OUTPUT_FOLDER = '/home/PUBLIC'
    ROUND_FLOAT = 4
    MAX_ITER = 10000  # 000
    DATA_ID = 602  # 53: Iris, 109: Wine, 602: DryBean
    EPSILON = 1e-5
    SEED = 42
    SPLIT = '&'

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def print_info(title: str, X: np.ndarray, labels: np.ndarray, process_time: float, step: int = 0, split: str = SPLIT) -> str:
        # print("labels=", len(labels), labels)
        # print("Số lần xuất hiện các phần tử trong mỗi cụm=", count_data_array(labels))
        kqdg = [
            title,
            wdvl(process_time),
            str(step),
            wdvl(dunn(X, labels)),  # DI
            wdvl(davies_bouldin(X, labels)),  # DB
            wdvl(silhouette(X, labels)),  # SI
            wdvl(calinski_harabasz(X, labels)),  # CH
        ]
        result = split.join(kqdg)
        print(result)
        return result

    if DATA_ID in TEST_CASES:
        _TEST = TEST_CASES[DATA_ID]
        _start_time = time.time()
        _dt = fetch_data_from_local(DATA_ID)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))

        titles = ['Alg', 'time', 'step', 'DI', 'DB', 'CH', 'SI']
        print(SPLIT.join(titles))
        print('---------------------')

        n_cluster = _TEST['n_cluster']  # Số lượng cụm
        clustering_report = []
        # ============================================
        dkm = Dkmeans(n_clusters=n_cluster, epsilon=EPSILON, max_iter=MAX_ITER)
        dkm.fit(data=_dt['X'], seed=SEED)
        danh_gia = print_info(title='Dkmeans', X=_dt['X'], labels=dkm.labels, process_time=dkm.process_time, step=dkm.step)
        clustering_report.append(danh_gia)
        # ============================================
        _start_time = time.time()
        # km = KMeans(n_clusters=n_cluster, init='k-means++', n_init=10, max_iter=MAX_ITER, random_state=SEED)
        km = KMeans(n_clusters=n_cluster, max_iter=MAX_ITER, random_state=SEED)
        km.fit(_dt['X'])
        _process_time = time.time() - _start_time
        danh_gia = print_info(title='sklearn', X=_dt['X'], labels=km.labels_, process_time=_process_time)
        clustering_report.append(danh_gia)
        # ============================================
        with open(os.path.join(OUTPUT_FOLDER, f'kmeans_report.txt'), 'w') as crw:
            crw.write('\\\\ \n'.join(clustering_report))


# /home/d312env/bin/python /home/dll/ncs/clustering/kmeans.py
