import numpy as np
from scipy.spatial.distance import cdist
from Phan_cum.FCM import Dfcm

class DPCM(Dfcm):
    def __init__(self, C: int, K:float = 1, m: float = 2, eps: float = 0.00001, maxiter: int = 10000):
        self._k = K
        self.typicality = None
        super().__init__(C, m, eps, maxiter)

    def _init_typicality(self, data: np.ndarray, seed: int = 0):
        N = data.shape[0]
        if seed > 0:
            seed = seed
        return np.random.rand(N, self._C)
    
    def _compute_gammas(self, centroids: np.ndarray) -> float:
        _d = cdist(self.local_data, centroids)
        numerator = np.sum((self.typicality ** self._m) * (_d ** 2), axis=0)
        denominator = np.sum((self.typicality ** self._m), axis=0)
        return self._k * (numerator / denominator)

    def _update_typicality(self, centroids: np.ndarray) -> float:
        _gammas = self._compute_gammas(centroids=centroids)
        _d = cdist(self.local_data, centroids)

        denominator = 1 + ((1 / _gammas) * (_d ** 2)) ** (1 / (self._m - 1))
        return 1 / denominator

    def _update_centroid_from_typicality(self, data: np.ndarray, tm: np.ndarray) -> np.ndarray:
        denominator = np.sum(tm, axis=0)
        return np.dot(tm.T, data) / denominator[:, None]

    def _update_centroids(self, data: np.ndarray) -> np.ndarray:
        _tm = self.typicality ** self._m
        return self._update_centroid_from_typicality(data, _tm)

    def fit(self, data: np.ndarray, init_v: np.ndarray = None, seed: int = 42) -> tuple:
        self.local_data = data
        self.centroids = self._init_centroid_random(seed) if init_v is None else init_v
        self.typicality = self._init_typicality(data, seed)
        for step in range(self._maxiter):
            old_v = self.centroids.copy()
            self.typicality = self._update_typicality(centroids=old_v)
            self.centroids = self._update_centroids(data=self.local_data)
            if self.check_exit_by_centroids(old_v):
                break
        return self.typicality, self.centroids, step + 1