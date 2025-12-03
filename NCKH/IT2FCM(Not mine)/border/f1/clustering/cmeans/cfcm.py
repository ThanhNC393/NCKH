# HoangNX update 05/10/2024
import time
import numpy as np
from numpy import ndarray
from f1.clustering.cmeans.fcm import Dfcm
from f1.clustering.utility import distance_cdist

class Dcfcm:
    def __init__(self, n_clusters: int, n_sites: int, epsilon: float = 1e-5, max_iter: int = 10000, metric: str = 'euclidean'):
        self._metric = metric
        self._n_clusters = n_clusters
        self._m = 2
        self._n_sites = n_sites
        self._epsilon = epsilon
        self._max_iter = max_iter
        self._ds = []

    @property
    def datasites(self) -> list:
        return self._ds

    # Giai đoạn 1: Phân cụm cục bộ trên từng data site
    def phase1(self, datas: list, seed: int = None) -> list:
        self._ds = []
        for i, data in enumerate(datas):
            ds = Dfcm(n_clusters=self._n_clusters, epsilon=self._epsilon, max_iter=self._max_iter, index=i, metric=self._metric)
            ds.fit(data=data, seed=seed)
            ds.exited = False
            self._ds.append(ds)

    def _v_tilde_mean(self) -> np.ndarray:
        all_centroids = np.array([ds.centroids for ds in self._ds])
        return np.mean(all_centroids, axis=0)

    # Giai đoạn 2: Cộng tác giữa các data site
    def phase2(self, beta: float = 0) -> int:
        _start_time = time.time()
        for step in range(self._max_iter):
            v_tilde = self._v_tilde_mean()
            for ds in self._ds:
                if not ds.exited:
                    old_u = ds.membership.copy()
                    _beta = beta if beta > 0 else self._compute_beta(dii=ds)
                    self.update_membership(ds=ds, v_tilde=v_tilde, beta=_beta)
                    self.update_centroids(ds=ds, v_tilde=v_tilde, beta=_beta)
                    if ds.check_exit_by_membership(old_u):
                        ds.process_time += time.time() - _start_time
                        ds.step += step + 1
            # Thỏa mãn điều kiện dừng trên mọi data site
            if all(_ds.exited for _ds in self._ds):
                break
        return step + 1

    def fit(self, datas: list, beta: float = 0, seed: int = None) -> int:
        self.phase1(datas=datas, seed=seed)
        return self.phase2(beta=beta)

    # u ngã thể hiện sự cộng tác của datasite ii và jj
    def __compute_u_tilde(self, ds: Dfcm, centroids: np.ndarray) -> np.ndarray:
        distances = distance_cdist(ds.local_data, centroids, metric=self._metric)
        return ds.calculate_membership(distances)

    # tính giá trị hàm mục tiêu
    def __compute_j(self, data: np.ndarray, membership: np.ndarray, centroids: np.ndarray,) -> np.ndarray:
        distances = distance_cdist(data, centroids, metric=self._metric)
        return np.sum((membership ** 2) * (distances ** 2))

    def _compute_beta(self, dii: Dfcm) -> float:
        J = self.__compute_j(dii.local_data, dii.membership, dii.centroids)
        result = 0
        for jj, djj in enumerate(self._ds):
            if jj != dii.index:
                u_tilde = self.__compute_u_tilde(ds=dii, centroids=djj.centroids)
                j_title = self.__compute_j(dii.local_data, u_tilde, djj.centroids)
                result += min(1, J / j_title)
        return result / (self._n_sites - 1)

    def update_membership(self, ds: Dfcm, v_tilde: np.ndarray, beta: float):
        distances_m2 = distance_cdist(ds.local_data, ds.centroids, metric=self._metric) ** 2
        d_tilde_m2 = np.linalg.norm(ds.centroids - v_tilde, axis=1) ** 2
        tu = 1 / (distances_m2 + beta * d_tilde_m2)

        mau = np.sum((1 / (distances_m2 + beta * d_tilde_m2)) ** (1/(self._m - 1)), axis=1)
        mau = np.repeat(mau, self._n_clusters).reshape(tu.shape[0], self._n_clusters)
        ds.membership = tu / mau

    def _calc_um4v(self, ds: Dfcm):
        return ds.membership ** self._m  # (N, C)

    def update_centroids(self, ds: Dfcm, v_tilde: ndarray, beta: float):
        _um = self._calc_um4v(ds=ds)
        _t1 = np.dot(_um.T, ds.local_data)  # (C, N) x (N, D) = (C, D)
        _t2 = beta * _um.sum(axis=0)[:, np.newaxis] * v_tilde
        tu = _t1 + _t2

        mau = (1 + beta) * _um.sum(axis=0)[:, np.newaxis]
        ds.centroids = tu / mau

    def predict(self, new_data: np.ndarray):
        predictions = [ds.predict(new_data=new_data) for ds in self._ds]
        return np.mean(predictions, axis=0).astype(int)


if __name__ == '__main__':
    import time
    from f1.clustering.utility import round_float, extract_labels
    from f1.clustering.dataset import fetch_data_from_local, TEST_CASES, LabelEncoder, split_data_balanced
    from f1.clustering.validity import dunn, davies_bouldin, partition_coefficient, partition_entropy, Xie_Benie

    ROUND_FLOAT = 3
    LABELED_RATIOS = 0.7
    NUM_SITES = 3  # Số lượng datasite
    SEED = 42
    SPLIT = '\t'
    # =======================================

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def write_report_fcm(alg: str, index: int, process_time: float, step: int, X: np.ndarray, V: np.ndarray, U: np.ndarray) -> str:
        labels = extract_labels(U)  # Giai mo
        kqdg = [
            alg,
            str(index),
            wdvl(process_time, n=2),
            str(step),
            wdvl(dunn(X, labels)),  # DI
            wdvl(davies_bouldin(X, labels)),  # DB
            wdvl(partition_coefficient(U)),  # PC
            wdvl(partition_entropy(U)),  # PE
            wdvl(Xie_Benie(X, V, U)),  # XB
        ]
        return SPLIT.join(kqdg)
    # =======================================

    clustering_report = []
    data_id = 602
    if data_id in TEST_CASES:
        _start_time = time.time()
        _TEST = TEST_CASES[data_id]
        _dt = fetch_data_from_local(data_id)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        X, Y = _dt['X'], _dt['Y']
        _size = f"{_dt['data']['num_instances']} x {_dt['data']['num_features']}"
        print(f'size={_size}')
        C = _TEST['n_cluster']
        # ===============================================
        dlec = LabelEncoder()
        X = _dt['X']
        datas, labels = split_data_balanced(data=X, labels=dlec.fit_transform(_dt['Y'].flatten()), n_sites=NUM_SITES, seed=SEED)

        # ===============================================
        print(f'VERSION 1.0 Num_site={NUM_SITES}, C={C} ===================================')
        cfcm = Dcfcm(n_clusters=C, n_sites=NUM_SITES, epsilon=1e-5, max_iter=1000)
        cfcm.phase1(datas=datas, seed=SEED)
        titles = ['Alg', 'DS', 'Time', 'Step', 'DI+', 'DB-', 'PC+', 'PE-', 'XB-']
        print(SPLIT.join(titles))
        for i, ds in enumerate(cfcm.datasites):
            danh_gia = write_report_fcm('FCM', i, ds.process_time, ds.step, X=ds.local_data, V=ds.centroids, U=ds.membership)
            print(danh_gia)
        # ---------------------
        step = cfcm.phase2()
        print(f'Collaborative step={step} ------------------------------------------')
        for i, ds in enumerate(cfcm.datasites):
            danh_gia = write_report_fcm('CFCM', i, ds.process_time, ds.step, X=ds.local_data, V=ds.centroids, U=ds.membership)
            print(danh_gia)

# python nckh/f1/clustering/cmeans/cfcm.py
