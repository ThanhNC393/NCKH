import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from Phan_cum.FCM import Dfcm

class DSSFCM(Dfcm):

    def __init__(self, k: int, C: int, m: float = 2, eps: float = 1e-5, maxiter: int = 10000):
        super().__init__(C, m, eps, maxiter)
        self._k = k

    # Khởi tạo ma trận U

    # Tạo ma trận U_ngang với nhãn cụm
    def _create_partial_U(self, labels) -> np.ndarray:
        labels_num, uniques = pd.factorize(labels)
        n_labels = len(labels_num)
        n_cluster = self._C
        U_ = np.eye(n_cluster)[labels_num]
        
        # Chọn ngẫu nhiên (100 * k)% số dòng để giữ lại
        n_labels_1 = int(self._k * n_labels)
        mask = np.zeros(n_labels, dtype=bool)
        mask[np.random.choice(n_labels, n_labels_1, replace=False)] = True
        U_[~mask] = 0 # Đặt 100 * (1 - k)% số dòng còn lại thành 0
        
        return U_, labels_num

    # Cập nhật ma trận tâm cụm (tính centroid)
    def _update_centroids(self, data: np.ndarray, membership: np.ndarray, U_: np.ndarray) -> np.ndarray:
        u_m = (membership - U_) ** self._m
        centroids = (u_m.T @ data) / u_m.sum(axis=0)[:, np.newaxis]
        return centroids

    # Cập nhật ma trận độ thuộc U cho SSFCM
    def _update_membership(self, data: np.ndarray, centroids: np.ndarray, U_: np.ndarray) -> np.ndarray:
        dall = cdist(data, centroids)
        dall[dall == 0] = np.finfo(float).eps
        membership = self._init_membership(data)
        numerator = 1 - np.sum(U_, axis=1)
        for k in range(self._C):
            membership[:, k] = U_[:, k] + numerator / np.sum((dall[:, k][:, np.newaxis] / dall) ** (2 / (self._m - 1)), axis=1)
        return membership

    # Chạy thuật toán SSFCM
    def SSFCM(self, data: np.ndarray, labels: np.ndarray, seed: int = 42) -> tuple:
        if seed > 0:
            np.random.seed(seed=seed)
        N, D = data.shape

        membership = self._init_membership(data)
        centroids = np.random.rand(self._C, D)
        U_, labels_num = self._create_partial_U(labels)

        for step in range(self._maxiter):

            old_membership = membership.copy()
            centroids = self._update_centroids(data, membership, U_)
            membership = self._update_membership(data, centroids, U_)

            if np.linalg.norm(membership - old_membership) < self._eps:
                break

        return membership, centroids, U_, labels_num, step + 1
