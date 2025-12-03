import numpy as np
from scipy.spatial.distance import cdist
from Phan_cum.SSFCM import DSSFCM

class SSCFCM:

    def __init__(self, X: np.ndarray, y: np.ndarray, P: int, C:int, eps: float = 1e-5, m: int = 2, max_iter: int = 10000) -> None:
        self.__X = X
        self.__y = y
        self.__P = P
        self.__C = C
        self.__eps = eps
        self.__m = m
        self.__max_iter = max_iter
        self.exit = [False] * self.__P

    def split_data(self) -> tuple:
        X_splits = np.array_split(self.__X, self.__P)
        Y_splits = np.array_split(self.__y, self.__P)
        return X_splits, Y_splits

    # def _Urs(self, X_site: np.ndarray, V: np.ndarray, V_nga: np.ndarray, C: int, Beta: float = 0.5) -> np.ndarray:
    #     # Tính ma trận khoảng cách Euclidean bình phương từ X tới các tâm cụm V
    #     Drs_squared = cdist(X_site, V) ** 2
        
    #     # Khởi tạo ma trận tổng
    #     Urs_res = np.zeros((X_site.shape[0], C))
    #     denominator_sum = 0
    #     denominator_list = []
    #     for j in range(C):
    #         denominator = Drs_squared[:, j] + Beta * (np.linalg.norm(V[j] - V_nga[j]) ** 2)
    #         denominator_list.append(denominator)
    #         denominator_sum += 1 / denominator

    #     for j in range(C):
    #         Urs_res[:, j] = 1 / (denominator_list[j] * denominator_sum)

    #     return Urs_res

    def _Urs(self, X_site: np.ndarray, V: np.ndarray, V_nga: np.ndarray, C: int, Beta: float = 0.5) -> np.ndarray:
        Drs_squared = cdist(X_site, V) ** 2  # (N, C)
        
        V_squared = np.linalg.norm(V - V_nga, axis=1) ** 2  # (C,)
        V_squared = V_squared[np.newaxis, :]  # (1, C)
        
        denominator = Drs_squared + Beta * V_squared  # (N, C)
        denominator_sum = np.sum(1 / denominator, axis=1)  # (N,)
        
        Urs_res = 1 / denominator / denominator_sum[:, np.newaxis]  # (N, C)
        
        return Urs_res


    def _Vrt(self, X_site: np.ndarray, U_rs: np.ndarray, U_: np.ndarray, V_nga: np.ndarray, C: int, Beta: float = 0.5, m: float = 2.0) -> np.ndarray:
        weights = (U_rs - U_) ** m
        cum1 = np.sum(weights[..., np.newaxis] * X_site[:, np.newaxis, :], axis=0)
        cum2 = np.sum(weights, axis=0)[:, np.newaxis] * V_nga * Beta
        cum3 = np.sum(weights, axis=0)[:, np.newaxis] * (1 + Beta)
        Vrt = (cum1 + cum2) / cum3
        return Vrt

    def SSCFCM(self) -> list:
        
        # Chia dữ liệu thành P datasite
        X_splits, y_splits = self.split_data()

        N, D = self.__X.shape
        V_nga = np.zeros((self.__C, D))
        ssfcm_values = []
        
        # Chạy SSFCM trên từng datasite và thu thập các giá trị
        for i in range(self.__P):
            ssfcm = DSSFCM(0.3, self.__C)
            ssfcm_instance = ssfcm.SSFCM(X_splits[i], y_splits[i])
            ssfcm_values.append(ssfcm_instance)
            V_nga += ssfcm_instance[1]  # Cộng V
            
        V_nga /= self.__C

        for step in range(self.__max_iter):
            for i, datasite in enumerate(X_splits):
                old_centroid = ssfcm_values[i][1]
                # Cập nhật U và V
                ssfcm_values[i] = list(ssfcm_values[i])  # Chuyển tuple thành list để cập nhật
                ssfcm_values[i][0] = self._Urs(X_site = datasite, V = ssfcm_values[i][1], V_nga = V_nga, C = self.__C)
                ssfcm_values[i][1] = self._Vrt(X_site = datasite, U_rs = ssfcm_values[i][0], U_ = ssfcm_values[i][2], V_nga = V_nga, C = self.__C)

                # Kiểm tra điều kiện dừng sớm
                if np.linalg.norm(ssfcm_values[i][1] - old_centroid) < self.__eps:
                    self.exit[i] = True
            
            if all(self.exit):
                break

        return ssfcm_values
