import numpy as np
from scipy.spatial.distance import cdist

class Dfcm:

    def __init__(self, C: int, m: float = 2, eps: float = 1e-5, maxiter: int = 10000):
        self._C = C
        self._m = m
        self._eps = eps
        self._maxiter = maxiter
        # self._P = P  # Số lượng datasite

        self.membership = None
        self.centroids = None

        self.__exited = False

    @property
    def exited(self) -> bool:
        return self.__exited

    @exited.setter
    def exited(self, value: bool):
        self.__exited = value

    def _init_centroid_random(self, seed: int = 0) -> np.ndarray:
        if seed > 0:
            np.random.seed(seed=seed)
        return self.local_data[np.random.choice(len(self.local_data), self._C, replace=False)]
    
    @staticmethod
    def _division_by_zero(data: np.ndarray) -> np.ndarray:
        data[data == 0] = np.finfo(float).eps
        return data
    
    def __max_abs_epsilon(self, val: np.ndarray) -> bool:
        if not self.__exited:
            self.__exited = (np.abs(val)).max(axis=(0, 1)) < self._eps
        return self.__exited

    def check_exit_by_membership(self, membership: np.ndarray) -> bool:
        return self.__max_abs_epsilon(self.membership - membership)

    def check_exit_by_centroids(self, centroids: np.ndarray) -> bool:
        return self.__max_abs_epsilon(self.centroids - centroids)

    # Khởi tạo ma trận thành viên
    def _init_membership(self, data: np.ndarray, seed: int = 0) -> np.ndarray:
        N = data.shape[0]
        if seed > 0:
            np.random.seed(seed=seed)
        _U = np.random.rand(N, self._C)
        U = _U / _U.sum(axis = 1)[:, None]
        return U

    # Cập nhật ma trận tâm cụm
    def _update_centroids(self, data: np.ndarray, membership: np.ndarray) -> np.ndarray:
        u_m = membership ** self._m
        V = (u_m.T @ data) / (u_m.sum(axis=0)[:, None])
        return V
    
    @staticmethod
    def calculate_membership_by_distances(distances: np.ndarray, m: float = 2) -> np.ndarray:
        _d = distances[:, :, None] * ((1 / Dfcm._division_by_zero(distances))[:, None, :])
        power = 2 / (m - 1)
        denom = (_d ** power).sum(axis=2)
        return 1 / denom

    # Cập nhật ma trận độ thuộc
    def _update_membership(self, data: np.ndarray, centroids: np.ndarray):
        dall = cdist(data, centroids, metric='euclidean')
        return self.calculate_membership_by_distances(dall)

    def fcm(self, data: np.ndarray, seed: int = 42) -> list:

        membership = self._init_membership(data, seed=seed)
        for step in range(self._maxiter):
            old_u = membership.copy()
            centroids = self._update_centroids(data, old_u)
            membership = self._update_membership(data, centroids)
            if self.__max_abs_epsilon()
                break
        return membership, centroids, step + 1