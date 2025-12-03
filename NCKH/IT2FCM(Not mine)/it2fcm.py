import numpy as np
from f1.clustering.cmeans.fcm import Dfcm
from CoreFCM.nckh.f1.clustering.dataset import fetch_data_from_local
from f1.clustering.utility import distance_cdist
import time


class It2fcm(Dfcm):
    def __init__(self, n_clusters: int, m1: float, m2: float, epsilon: float, max_iter: int):
        self._n_clusters = n_clusters
        self._m1 = m1
        self._m2 = m2
        self._epsilon = epsilon
        self._max_iter = max_iter
        self.step = 0
        self.process_time = 0

    # Tính U mờ loại 2: U trên và U dưới     
    def compute_u(self): 
        distance = distance_cdist(self.local_data, self.centroids)
        u1 = self.calculate_membership_by_distances(distance, self._m1)
        u2 = self.calculate_membership_by_distances(distance, self._m2)
        u1_u2 = np.stack([u1, u2], axis=0)
        moc_so_sanh = 1 / np.sum((distance.T[:, :, None] / distance), axis=2).T
        
        # Đổi chỗ U trên và U dưới nếu mốc so sánh nhỏ hơn 1/n_clusters
        index_swap = np.where(moc_so_sanh < 1 / self._n_clusters)
        u1_u2_copy = u1_u2.copy()
        u1_u2[0][index_swap] = u1_u2[1][index_swap]
        u1_u2[1][index_swap] = u1_u2_copy[0][index_swap]

        self.membership = u1_u2 # 2xNxC


    def compute_v(self):
        # Tính trung bình U trên và U dưới
        u_trung_binh = np.sum(self.membership, axis=0) / 2 # NxC
        
        return np.sum(u_trung_binh[:, None, :] * self.local_data[:, :, None], axis=0).T / np.sum(u_trung_binh, axis=0)[:, None] # CxD / Cx1

    def compute_c(self, lower_index: np.ndarray, u_lower: np.ndarray, upper_index: np.ndarray, u_upper: np.ndarray) -> np.ndarray:
        c2 = np.sum(self.local_data[lower_index] * u_lower[:, None], axis=0) + np.sum(self.local_data[upper_index] * u_upper[:, None], axis=0) 
        return c2 / (np.sum(u_upper) + np.sum(u_lower))

    def karnik_algo(self, clus: int, mode: int):
        c = self.compute_v()
        data_T = self.local_data.T # DxN

        # Sắp xếp dữ liệu theo từng cột
        list_index = []
        for i in range(self._n_feature):
            index_sort = np.argsort(data_T[i])
            list_index.append(index_sort)
            data_T[i] = data_T[i][index_sort]
        list_index = np.stack(list_index, axis=0) # DxN

        for z in range(self._max_iter):
            list_k = []
            for i in range(self._n_feature):
                for j in range(self._n_sample):
                    if data_T[i][j] >= c[clus][i] or j==self._n_sample-1:
                        list_k.append(j)
                        break
            for i in range(self._n_feature):
                if list_k[i] == 0:
                    list_k[i] += 1
                elif list_k[i] == self._n_sample - 1:
                    list_k[i] -= 1
            c_new = []
            u_2 = []
            for i in range(self._n_feature):
                lower_index = list_index[i][: list_k[i] + 1]
                upper_index = list_index[i][list_k[i] + 1 :]
                if mode == 1:
                    u_lower = self.membership[0, lower_index, clus]
                    u_upper = self.membership[1, upper_index, clus]
                else:
                    u_lower = self.membership[1, lower_index, clus]
                    u_upper = self.membership[0, upper_index, clus]
                c_new.append(self.compute_c(lower_index=lower_index, u_lower=u_lower, upper_index=upper_index, u_upper=u_upper))
                u_2.append(np.concatenate([u_lower, u_upper])[list_index[i]])
            
            c_new=np.sum(np.array(c_new), axis=0) / self._n_feature
            if np.linalg.norm(c_new-c[clus]) < self._epsilon:
                self.step += z + 1
                return c_new, np.sum(np.stack(u_2, axis=0).T, axis=1) / self._n_feature
            c[clus]=c_new
        self.step += self._max_iter
        return c_new, np.sum(np.stack(u_2, axis=0).T, axis=1) / self._n_feature
        

    def fit(self, data: np.ndarray, seed: int = 42):
        start_time=time.time()
        self.local_data = data
        self._n_sample = self.local_data.shape[0]
        self._n_feature = self.local_data.shape[1]
        self.centroids = self._init_centroid_random(seed=seed)
        self.membership = np.zeros(shape=[2, self._n_sample, self._n_clusters])
        for i in range(self._max_iter):
            self.compute_u()
            upper_centroid = []
            lower_centroid = []
            real_u = []
            for j in range(self._n_clusters): 
                km_for_upper = self.karnik_algo(j, 1)
                upper_centroid.append(km_for_upper[0])
                km_for_lower = self.karnik_algo(j, 0)
                lower_centroid.append(km_for_lower[0])
                real_u.append((km_for_upper[1]+km_for_lower[1])/2)
            upper_centroid = np.array(upper_centroid)
            lower_centroid = np.array(lower_centroid)

            hard_partioning = (upper_centroid + lower_centroid) / 2
            if np.linalg.norm(self.centroids - hard_partioning) < self._epsilon:
                self.centroids = hard_partioning
                self.process_time = time.time() - start_time
                self.membership = np.stack(real_u, axis=0).T
                break
            self.centroids = hard_partioning
        
if __name__ == '__main__':










    import pandas as pd
    from CoreFCM.nckh.f1.clustering.dataset import TEST_CASES, LabelEncoder
    # from f1.clustering.dataset import TEST_CASES, LabelEncoder
    from f1.clustering.utility import round_float, extract_labels
    from f1.clustering.validity import davies_bouldin, partition_coefficient, partition_entropy, Xie_Benie, accuracy_score, f1_score
    from f1.clustering.validity import separation, calinski_harabasz, silhouette

    # info =================================================
    ALG = 'IT2FCM'
    PAPER = 'Uncertain Fuzzy Clustering: Interval Type-2 Fuzzy Approach to C-Means'
    AUTHOR = 'ThanhNC & ManhNV'
    DATE = '2025-02-22 10:00'

    # params ===============================================
    DATA_ID = 53
    ROUND_FLOAT = 3
    N_SPACE = 10
    
    M1 = 2
    M2 = 3
    EPSILON = 1e-5
    MAX_ITER = 10000
    SEED = 42
    ALPHA = 0.01

    # ============================
    def print_info(dataset: str):
        print(f'PAPER: {PAPER}')
        print(f'ALG: {ALG}')
        print(f'DATASET: {dataset}')
        print(f'AUTHOR: {AUTHOR}')
        print(f'DATE: {DATE}')
    # ============================
    def write_report(X: np.ndarray, V: np.ndarray, U: np.ndarray, y_true: np.ndarray, time_process: float, step: int, title: str, site: int, labeled: np.ndarray = None, no_label: int = -1) -> str:
        labels = extract_labels(U)  # Giai mo
        kqdg = {
            'DB': round_float(davies_bouldin(X, labels), n=ROUND_FLOAT),
            'PC': round_float(partition_coefficient(U), n=ROUND_FLOAT),
            'PE': round_float(partition_entropy(U), n=ROUND_FLOAT),
            'XB': round_float(Xie_Benie(X, V, U), n=ROUND_FLOAT),
            'SE': round_float(separation(X, U, V), n=ROUND_FLOAT),
            'CH': round_float(calinski_harabasz(X, labels), n=ROUND_FLOAT),
            'SI': round_float(silhouette(X, labels), n=ROUND_FLOAT),
            'F1': round_float(f1_score(y_true, labels), n=ROUND_FLOAT),
            'AC': round_float(accuracy_score(y_true, labels), n=ROUND_FLOAT)
        }
        return '{site:<{n_space}} {title:<{n_space}} {time_process:<{n_space}} {step:<{n_space}} {DB:<{n_space}} {PC:<{n_space}} {PE:<{n_space}} {XB:<{n_space}} {SE:<{n_space}} {CH:<{n_space}} {SI:<{n_space}} {F1:<{n_space}} {AC:<{n_space}}'.format(site=site, title=title, time_process=round_float(time_process, n=ROUND_FLOAT), step=step, **kqdg, n_space=N_SPACE)

    # ============================
    if DATA_ID in TEST_CASES:
        _TEST = TEST_CASES[DATA_ID]
        _start_time = time.time()
        _dt = fetch_data_from_local(DATA_ID, folder='E:\\Python\\NCKH\\IT2FCM\\CoreFCM\\dataset')
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        n_cluster = _TEST['n_cluster']
        print_info(dataset=_TEST['name'])
        # ===============================================

        data = _dt['X']
        Y = _dt['Y']
        # Mã hóa nhãn đầu vào
        dlec = LabelEncoder()
        labels = dlec.fit_transform(Y.flatten())
        # --------------------------------------------
        print(f'{"SITE":<{N_SPACE}} {"ALG":<{N_SPACE}} {"TIME":<{N_SPACE}} {"STEP":<{N_SPACE}} {"DB-":<{N_SPACE}} {"PC+":<{N_SPACE}} {"PE-":<{N_SPACE}} {"XB-":<{N_SPACE}} {"S+":<{N_SPACE}} {"CH+":<{N_SPACE}} {"SI+":<{N_SPACE}} {"F1+":<{N_SPACE}} {"AC+":<{N_SPACE}}')
        print('-'*N_SPACE*13)
        # --------------------------------------------
        fcm = Dfcm(n_clusters=n_cluster, m=M1, epsilon=EPSILON, max_iter=MAX_ITER)
        fcm.fit(data, seed=SEED)
        print(write_report(X=data, V=fcm.centroids, U=fcm.membership, y_true=labels, time_process=fcm.process_time, step=fcm.step, title='FCM', site=0))
        
        # --------------------------------------------
        it2fcm=It2fcm(n_clusters=n_cluster, m1=M1, m2=M2, epsilon=EPSILON, max_iter=MAX_ITER)
        it2fcm.fit(data, seed=SEED)

        print(write_report(X=data, V=it2fcm.centroids, U=it2fcm.membership, y_true=labels, time_process=it2fcm.process_time, step=it2fcm.step, title='IT2FCM', site=0, labeled=labels, no_label=0))
