import numpy as np

from scipy.spatial.distance import cdist
from Phan_cum.PCM import DPCM
from Phan_cum.FCM import Dfcm

class Dfpcm(DPCM):

    def __init__(self, C: int, eta: float, K: float = 1, m: float = 2, eps: float = 0.00001, maxiter: int = 10000):
        self._eta = eta
        self.typicality = None
        super().__init__(C, K, m, eps, maxiter)        
    
    @staticmethod
    def calculate_typicality_by_distances(distances: np.ndarray, eta: float) -> np.ndarray:
        _d = distances[:, None, :] * ((1 / Dfcm._division_by_zero(distances))[None, :, :])
        power = 2 / (eta - 1)
        denom = (_d ** power).sum(axis=0)
        return 1 / denom
    
    def _update_typicality(self, centroids: np.ndarray):
        dall = cdist(self.local_data, centroids, metric='euclidean')
        return self.calculate_typicality_by_distances(dall, self._eta)
    
    def _update_centroids(self, data: np.ndarray, membership: np.ndarray) -> np.ndarray:
        _w = (membership ** self._m) + (self.typicality ** self._eta)
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
            self.centroids = self._update_centroids(self.local_data, self.membership)
            if self.check_exit_by_centroids(old_v):
                break
        return self.membership, self.centroids, step + 1, self.typicality