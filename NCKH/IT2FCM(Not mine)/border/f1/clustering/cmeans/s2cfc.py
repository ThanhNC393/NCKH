# https://drive.google.com/file/d/1Tp3FOt9EG_cdPUnFA2dnUNzdzZdrIgNr/view?usp=drive_link
# 2021-S2CFC-Semi‑supervised collaborative fuzzy clustering method

import time
import numpy as np

from scipy.optimize import linear_sum_assignment
from f1.clustering.cmeans.fcm import Dfcm
from f1.clustering.cmeans.ssfcm import Dssfcm
from f1.clustering.utility import distance_cdist, division_by_zero, reorder_centroids


class Ds2CFC:
    def __init__(self, n_clusters: int, n_sites: int, gamma: float = 0.5, delta: float = 0.5, alpha: float = 0.5,
                 m: float = 2, epsilon: float = 1e-6, max_iter: int = 10000, metric: str = 'euclidean', no_label: int = -1):
        self._n_clusters = n_clusters
        self._n_sites = n_sites
        self._gamma = gamma
        self._delta = delta
        self._alpha = alpha

        self._m = m
        self._epsilon = epsilon
        self._metric = metric
        self._max_iter = max_iter
        self._no_label = no_label

        self._ds = None
        self._u_star = None
        self._beta = None

    def __safe_log(self, data: np.ndarray) -> np.ndarray:
        data[data == 0] = np.finfo(float).eps
        return np.log(data)

    def __safe_exp(self, data: np.ndarray) -> np.ndarray:
        data[data == 0] = np.finfo(float).eps
        return np.exp(data)

    @property
    def datasites(self) -> list:
        return self._ds

    # Giai đoạn 1: Phân cụm cục bộ tại các datasites
    def phase1(self, datas: list[np.ndarray], labeled: list[np.ndarray], seed: int = 42):
        self._ds = [Dssfcm(n_clusters=self._n_clusters,
                           m=self._m, epsilon=self._epsilon, max_iter=self._max_iter,
                           no_label=self._no_label,
                           index=i, metric=self._metric) for i in range(self._n_sites)]
        for i, ds in enumerate(self._ds):
            ds.fit(data=datas[i], labeled=labeled[i], seed=seed)

    def check_exit(self, val: np.ndarray) -> bool:
        return np.abs(val).max(axis=(0, 1)) < self._epsilon

    # Giai đoạn 2: Cộng tác giữa các datasites
    def phase2(self):
        _start_time = time.time()
        # Trao đổi và tính toán các IGL (ma trận phân vùng cộng tác)
        _ustar = self.__compute_membership_star()
        # Sử dụng GH để sắp xếp các IGL
        _, self._col_ind = self.__gregson_hungarian(_ustar)
        # old_memberships = np.zeros((self._P, self._ds[0].local_data.shape[0], self._n_clusters))
        for _s1 in range(self._max_iter):
            # Trao đổi và tính toán các IGL (ma trận phân vùng cộng tác)
            _ustar = self.__compute_membership_star()
            # Sử dụng GH để sắp xếp các IGL
            self._u_star = self.__reorder_u_star_with_col_ind(_ustar, self._col_ind)
            # Tính toán ma trận beta
            self._beta = self.__compute_beta_matrix()

            _ck_olds = {}
            # Cập nhật ma trận phân vùng U và tâm cụm tại từng datasite
            for i, ds in enumerate(self._ds):
                for _s2 in range(self._max_iter):
                    _ck_olds[i] = ds.centroids.copy()
                    ds.centroids = self._update_centroids(idx=i)
                    ds.membership = self._update_membership(idx=i)
                    ds.step += 1
                    if self.check_exit(ds.centroids - _ck_olds[i]):
                        break
            # Thỏa mãn điều kiện dừng trên mọi data site
            if all(self.check_exit(ds.centroids - _ck_olds[i]) for i, ds in enumerate(self._ds)):
                for ds in self._ds:
                    ds.process_time += time.time() - _start_time
                break

    # Cập nhật ma trận phân vùng tại datasite i
    def _update_membership(self, idx: int) -> np.ndarray:
        _zeta = self.__compute_zeta(idx)
        _tau = self.__compute_tau(idx)
        _numerator = self.__safe_exp(_tau / _zeta)
        _denominator = np.sum(_numerator, axis=1)
        return _numerator / division_by_zero(_denominator)[:, None]

    # Cập nhật lại tâm cụm
    def _update_centroids(self, idx: int) -> np.ndarray:
        _numerator = np.dot(self._ds[idx].membership.T, self._ds[idx].local_data)
        _denominator = np.sum(self._ds[idx].membership, axis=0)
        return _numerator / division_by_zero(_denominator)[:, None]

    # Tính toán zeta tại datasite i => shape (N)
    def __compute_zeta(self, idx: int) -> float:
        _element_1 = self._gamma
        _element_2 = self._alpha * self._ds[idx].labeled_bar
        _element_3 = 0
        for j in range(self._n_sites):
            if j != idx:
                _element_3 += self._beta[idx, j] * (1 - self._ds[idx].labeled_bar)
        return _element_1 + _element_2 + _element_3

    # Tính toán tau tại datasite i => shape (NxC)
    def __compute_tau(self, idx: int) -> float:
        _element_1 = - self._gamma
        _element_2 = - distance_cdist(self._ds[idx].local_data, self._ds[idx].centroids)
        _element_3 = self._alpha * self._ds[idx].labeled_bar * (self.__safe_log(self._ds[idx].membership_bar) - 1)

        _element_4 = 0
        for j in range(self._n_sites):
            if j != idx:
                _element_4 += self._beta[idx, j] * (1 - self._ds[idx].labeled_bar) * (self.__safe_log(self._u_star[idx, j]) - 1)
        return _element_1 + _element_2 + _element_3 + _element_4

    # Tính toán ma trận beta ma trận 2 chiều shape: (PxP)
    def __compute_beta_matrix(self) -> np.ndarray:
        self._eta = self.__compute_eta_matrix()
        _d = self.__safe_exp(-self._delta * self._eta - 1)
        column_sums = np.sum(_d, axis=0)
        return _d / division_by_zero(column_sums)[None, :]

    # Tính ma trận eta 2 chiều shape (PxP)
    def __compute_eta_matrix(self) -> np.ndarray:
        result = np.zeros((self._n_sites, self._n_sites))
        for i in range(self._n_sites):
            for j in range(self._n_sites):
                if i != j:
                    result[i, j] = self.__compute_eta_element(idxi=i, idxj=j)
        return result

    def __compute_eta_element(self, idxi: int, idxj: int) -> float:
        _e1 = self._ds[idxi].labeled_bar * self._ds[idxi].membership_bar * \
            self.__safe_log(self._ds[idxi].membership_bar / self._u_star[idxi, idxj])
        _e2 = (1 - self._ds[idxi].labeled_bar) * self._ds[idxi].membership * \
            self.__safe_log(self._ds[idxi].membership / self._u_star[idxi, idxj])
        return np.sum(_e1 + _e2)

    def __gregson_hungarian(self, u_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col_ind = np.zeros((self._n_sites, self._n_sites, self._n_clusters))
        for i in range(self._n_sites):
            for j in range(self._n_sites):
                if i != j:
                    membership = u_star[i, j].copy()
                    u_star[i, j], col_ind[i, j] = self.__reorder_membership_with_hungarian(u_ref=self._ds[i].membership, u=membership)
        return u_star, col_ind

    # Tính toán ma trận phân vùng cộng tác 4 chiều shape (PxPxNxC)
    def __compute_membership_star(self) -> np.ndarray:
        result = np.zeros((self._n_sites, self._n_sites, self._ds[0].local_data.shape[0], self._n_clusters))
        for i in range(self._n_sites):
            for j in range(self._n_sites):
                if i != j:
                    result[i, j] = self.__compute_u_star_element(sitei=i, sitej=j)
        return result

    # Tính toán một phần tử của ma trận phân vùng cộng tác 4 chiều
    def __compute_u_star_element(self, sitei: int, sitej: int) -> np.ndarray:
        _d = distance_cdist(self._ds[sitei].local_data, self._ds[sitej].centroids)
        _dk = (_d[:, :, None] / _d[:, None, :]) ** 2
        _dk = np.sum(_dk, axis=2)
        return 1 / _dk

    # membership_ref_col là một cột của ma trận độ thuộc lấy mà mốc của datasite chọn làm mốc
    # membership_col là một cột của ma trận độ thuộc cần map của datasite khác
    def __compute_dissimilarity_element(self, u_ref_col: np.ndarray, u_col: np.ndarray) -> float:
        numerator = np.sum(np.minimum(u_ref_col, u_col))
        denominator = np.sum(np.maximum(u_ref_col, u_col))
        if denominator == 0:
            denominator = np.finfo(float).eps
        return 1 - (numerator / denominator)

    # Tính toán ma trận không tương đồng shape: (CxC)
    def __compute_dissimilarity(self, u_ref: np.ndarray, u: np.ndarray) -> np.ndarray:
        result = np.zeros((self._n_clusters, self._n_clusters))
        for i in range(self._n_clusters):
            for j in range(self._n_clusters):
                result[i, j] = self.__compute_dissimilarity_element(u_ref[:, i], u[:, j])
        return result

    # Sắp xếp lại ma trận phân vùng U theo thứ tự mới
    def __reorder_membership_with_hungarian(self, u_ref: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Bước 1: Xây dựng ma trận không tương đồng
        D = self.__compute_dissimilarity(u_ref=u_ref, u=u)

        # Bước 2: Áp dụng thuật toán Hungarian để tìm cách ánh xạ tối ưu
        _row_ind, col_ind = linear_sum_assignment(D)

        # Bước 3: Sắp xếp lại ma trận phân vùng U theo thứ tự mới
        reordered_membership = u[:, col_ind]
        return reordered_membership, col_ind

    def __reorder_u_star_with_col_ind(self, u_star: np.ndarray, col_ind: np.ndarray) -> np.ndarray:
        col_ind = col_ind.astype(int)
        for i in range(self._n_sites):
            for j in range(self._n_sites):
                if i != j:
                    u_star[i, j] = u_star[i, j][:, col_ind[i, j]]
        return u_star

    # Kết nối các V cộng tác ----------------
    # Cộng rồn X, V, tính V theo FCM, tính U
    def compute_centroids_fcm(self, x_global: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _vcs = [ds.centroids for ds in self._ds]
        _vs = np.concatenate(_vcs, axis=0)
        _fcm = Dfcm(n_clusters=self._n_clusters, m=self._m, epsilon=self._epsilon, max_iter=self._max_iter, metric=self._metric)
        _fcm.fit(data=_vs)
        u_global = Dssfcm.calculate_membership_by_distances(x_global, _fcm.centroids)
        return u_global, _fcm.centroids

    # Cộng rồn X, tính V trung bình, tính U theo FCM
    def compute_centroids_mean(self, x_global: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _vs = reorder_centroids(list_centroids=[ds.centroids for ds in self._ds])
        # ----------------------------------------
        v_global = np.mean(_vs, axis=0)
        _distances = distance_cdist(x_global, v_global)
        u_global = Dssfcm.calculate_membership_by_distances(_distances, m=self._m)
        return u_global, v_global

    # Tính U, V tổng hợp
    def fit(self, datas: list[np.ndarray], labeled: list[np.ndarray], by_vmean: bool = True) -> tuple:
        _start = time.time()
        self.phase1(datas=datas, labeled=labeled)
        self.phase2()

        x_global = np.concatenate(datas, axis=0)
        u_global, v_global = self.compute_centroids_mean(x_global) if by_vmean else self.compute_centroids_fcm(x_global)
        return u_global, v_global, x_global, time.time() - _start


if __name__ == "__main__":
    from f1.clustering.utility import round_float, extract_labels, count_data_array
    from f1.clustering.dataset import fetch_data_from_local, TEST_CASES, LabelEncoder, split_data_remove_label_for_collaborative
    from f1.clustering.validity import davies_bouldin, partition_coefficient, dunn, classification_entropy, silhouette, hypervolume, cs, separation, calinski_harabasz
    from f1.clustering.validity import mean_distance_cluster, mean_distance_point_cluster, f1_score, accuracy_score

    # info ================================
    ALGORITHM = '2021-S2CFC-Semi‑supervised collaborative fuzzy clustering method'
    VERSION = '1.3'
    AUTHOR = 'ManhNV & HoangNX'
    DATE = '10/11/24'

    # params ====================================
    ROUND_FLOAT = 3
    MAX_ITER = 10000
    DATA_ID = 53
    EPSILON = 1e-6
    LABELED_RATIOS = 0.7
    SEED = 42
    M = 2
    GAMMA = 0.005
    ALPHA = 1
    DELTA = 1e-3
    P = 3
    # ============================================
    SPLIT = '\t'

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def print_info(title: str, X: np.ndarray, U: np.ndarray, V: np.ndarray, y_true: np.ndarray, process_time: float, step: int = 0, split: str = SPLIT) -> str:
        labels = extract_labels(U)  # Giai mo
        kqdg = [
            title,
            wdvl(process_time),
            str(step),
            wdvl(dunn(X, labels)),  # DI
            wdvl(davies_bouldin(X, labels)),  # DB
            wdvl(partition_coefficient(U)),  # PC
            wdvl(classification_entropy(U)),  # CE
            wdvl(silhouette(X, labels)),  # SI
            wdvl(hypervolume(U, M)),  # FHV
            wdvl(cs(X, U, V, M)),  # CS
            wdvl(separation(X, U, V, M), n=0),  # S
            wdvl(calinski_harabasz(X, labels)),  # CH
            wdvl(mean_distance_cluster(V)),  # VTB
            wdvl(mean_distance_point_cluster(X, V)),  # XVT
            wdvl(f1_score(y_true, labels, average='macro')),  # F1
            wdvl(accuracy_score(y_true, labels))
        ]
        result = split.join(kqdg)
        print(result)
        return result

    # Chuẩn hóa dữ liệu về khoảng [0, 1]
    def normalize_data(data: np.ndarray) -> np.ndarray:
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        return normalized_data

    # ============================================
    if DATA_ID in TEST_CASES:
        _TEST = TEST_CASES[DATA_ID]
        _start_time = time.time()
        _dt = fetch_data_from_local(DATA_ID)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        n_cluster = _TEST['n_cluster']
        # ----------------------------------------
        dlec = LabelEncoder()
        X = _dt['X']
        Y = dlec.fit_transform(_dt['Y'].flatten())
        datas, labeled, labels = split_data_remove_label_for_collaborative(data=X,
                                                                           labels=Y,
                                                                           n_sites=P,
                                                                           labeled_percentage=LABELED_RATIOS,
                                                                           seed=SEED)
        # ----------------------------------------
        # Chuẩn hóa dữ liệu cho từng datasite
        # datas = [normalize_data(data) for data in datas]
        # ----------------------------------------
        print(f'Alg {ALGORITHM}')
        print(f'V{VERSION}, {AUTHOR} - Update {DATE}')
        print(f'Dataset', _TEST['name'])
        print(f'LABELED_RATIOS= {LABELED_RATIOS}')

        titles = ['SITE', 'time', 'step', 'DI+', 'DB-', 'PC+', 'CE-', 'CH+', 'SI+', 'FHV+', 'CS-', 'S-', 'VTB+', 'XVT+', 'F1+', 'AC+']
        print(SPLIT.join(titles))

        s2cfc = Ds2CFC(n_clusters=n_cluster, n_sites=P, m=M, epsilon=EPSILON, max_iter=MAX_ITER, gamma=GAMMA, delta=DELTA, alpha=ALPHA)
        s2cfc.phase1(datas=datas, labeled=labeled)
        print('#SSFCM ----------------------------------------')
        for i, dsi in enumerate(s2cfc.datasites):
            print_info(title=str(i),
                       X=datas[i],
                       U=dsi.membership,
                       V=dsi.centroids,
                       process_time=dsi.process_time,
                       y_true=labels[i],
                       step=dsi.step)

        print('#S2CFC ----------------------------------------')
        s2cfc.phase2()
        for i, dsi in enumerate(s2cfc.datasites):
            print_info(title=str(i),
                       X=datas[i],
                       U=dsi.membership,
                       V=dsi.centroids,
                       process_time=dsi.process_time,
                       y_true=labels[i],
                       step=dsi.step)

        # =====================================
        # # Test --------------------------------
        # u_global1, v_global1 = s2cfc.compute_centroids_mean(x_global=X)
        # print('X', X.shape)
        # print('u_global1', u_global1.shape)
        # print('v_global1', v_global1.shape)
        # l_global1 = extract_labels(u_global1)
        # print('l_global1', l_global1)
        # print('l_global1', count_data_array(l_global1))
        # # exit()
        # # print_info(title='centroids mean',
        # #            X=X,
        # #            U=u_global1,
        # #            V=v_global1,
        # #            process_time=0,
        # #            y_true=Y,
        # #            step=0)
        # # exit()
        # # Test --------------------------------
        # u_global2, v_global2 = s2cfc.compute_centroids_fcm(x_global=X)
        # print('u_global2', u_global2.shape)
        # print('v_global2', v_global2.shape)
        # l_global2 = extract_labels(u_global2)
        # print('l_global2', l_global2)
        # print('l_global2', count_data_array(l_global2))
        # # exit()
        # # print_info(title='centroids fcm',
        # #            X=X,
        # #            U=u_global2,
        # #            V=v_global2,
        # #            process_time=0,
        # #            y_true=Y,
        # #            step=0)

# python nckh/f1/clustering/cmeans/s2cfc.py
