from Data.data import TEST_CASES, fetch_data_from_uci, round_float
from Phan_cum.FCM import Dfcm
from validity import print_metrics
import numpy as np

if __name__ == "__main__":
    import time
    _start_time = time.time()
    MAX_ITER = 10000  # 000
    DATA_ID = 602  # 53: Iris, 109: Wine, 602: DryBean
    epsilon = 1e-5
    if DATA_ID in TEST_CASES:
        _TEST = TEST_CASES[DATA_ID]
        _dt = fetch_data_from_uci(DATA_ID)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        # --------------------------------
        _start_time = time.time()
        n_cluster = _TEST['n_cluster']  # Số lượng cụm
        dfcm = Dfcm(C= n_cluster, m=2, eps=epsilon, maxiter=MAX_ITER)
        test_points = _TEST['test_points']
        U1, V1, step = dfcm.fcm(data=_dt['X'])
        
        print('#DFCM ----------------------------------------------------')
        print("Số bước lặp:", step)
        print("V:", len(V1), V1[:1])
        print("U:", len(U1), U1[:1])
        print("Thời gian tính toán FCM", round_float(time.time() - _start_time))
        print_metrics(_dt['X'], _dt['y'], U1, V1)