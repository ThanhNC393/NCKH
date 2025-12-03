import numpy as np
import pandas as pd
from Data.data import TEST_CASES, fetch_data_from_uci, round_float
from Phan_cum.SSCFCM import SSCFCM
from validity import print_metrics

if __name__ == "__main__":
    import time
    _start_time = time.time()
    k = 0.3
    P = 3
    MAX_ITER = 10000
    DATA_ID = 602  # 53: Iris, 109: Wine, 602: DryBean
    epsilon = 1e-5

    if DATA_ID in TEST_CASES:
        _TEST = TEST_CASES[DATA_ID]
        _dt = fetch_data_from_uci(DATA_ID)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()

        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
    
    # Chạy SSCFCM trên dữ liệu gốc
    _start_time = time.time()
    n_cluster = _TEST['n_cluster']
    sscfcm = SSCFCM(_dt['X'], _dt['y'], P = P, C = n_cluster, eps = epsilon, m = 2, max_iter = MAX_ITER)
    X, y = sscfcm.split_data()
    sscfcm_lst = sscfcm.SSCFCM()
    for i in range(P):
        print(f'#DSSCFCM trên datasite thứ {i + 1}: -------------------------------')
        print("Số bước lặp:", sscfcm_lst[i][-1])
        print("V:", len(sscfcm_lst[i][1]), sscfcm_lst[i][1][:1])
        print("U:", len(sscfcm_lst[i][0]), sscfcm_lst[i][0][:1])
        print(f"Thời gian tính toán SSCFCM trên datasite thứ {i + 1}:", round_float(time.time() - _start_time))

        # Chỉ số 
        # print_metrics(X[i], sscfcm_lst[i][0], sscfcm_lst[i][1], sscfcm_lst[i][3])