import numpy as np
from scipy.spatial.distance import cdist
from Phan_cum.PCM import DPCM

class Dpfcm(DPCM):

    def __init__(self, C: int, a: float, b: float, eta: float, K:float = 1, m: float = 2, eps: float = 1e-5, maxiter: int = 10000):
        self._a = a
        self._b = b
        self._eta = eta
        self.typicality = None
        super().__init__(C, K, m, eps, maxiter)

    def _compute_gammas(self, centroids: np.ndarray) -> float:
        _d = cdist(self.local_data, centroids)
        numer = np.sum((self.membership ** self._m) * (_d ** 2), axis=0)
        denom = np.sum((self.membership ** self._m), axis=0)
        return self._k * (numer / denom)
    
    def _update_typicality(self, centroids: np.ndarray) -> np.ndarray:
        _gammas = self._compute_gammas(centroids=centroids)
        _d = cdist(self.local_data, centroids)
        denom = 1 + ((self._b / _gammas)[None, :] * _d**2) ** (1 / (self._eta - 1))
        return 1 / denom
    
    def _update_centroids(self, data: np.ndarray, membership: np.ndarray) -> np.ndarray:
        _w = self._a * (membership ** self._m) + self._b * (self.typicality ** self._eta)
        numer = np.sum(_w[:, :, None] * data[:, None, :], axis=0)
        denom = np.sum(_w, axis=0)[:, None]
        return numer / denom
    
    def fit(self, data: np.ndarray, init_v: np.ndarray, seed: int = 0):
        self.local_data = data
        self.centroids = self._init_centroid_random(data, seed) if init_v is None else init_v
        for step in range(self._maxiter):
            old_v = self.centroids.copy()
            self.membership = self._update_membership(data, old_v)
            self.typicality = self._update_typicality(old_v)
            print(np.sum(self.typicality, axis=0))
            return
            self.centroids = self._update_centroids(self.local_data, self.membership)
            if self.check_exit_by_centroids(old_v):
                break
        return self.membership, self.centroids, step + 1, self.typicality