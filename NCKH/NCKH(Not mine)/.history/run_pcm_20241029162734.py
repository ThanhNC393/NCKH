from Data.data import TEST_CASES, fetch_data_from_uci, round_float
from Phan_cum.PCM import DPCM
from validity import print_info, SPLIT

import numpy as np

if __name__ == "__main__":
    import time
    _start_time = time.time()
    MAX_ITER = 10000  # 000
    DATA_ID = 53  # 53: Iris, 109: Wine, 602: DryBean
    epsilon = 1e-5
    K = 1
    M = 2
    if DATA_ID in TEST_CASES:
        _TEST = TEST_CASES[DATA_ID]
        _dt = fetch_data_from_uci(DATA_ID)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        # --------------------------------
        init_V = np.array([[5.8, 2.7, 5.1, 1.9], [5.0, 2.0, 3.5, 1.0], [5.10, 3.50, 1.40, 0.30]])
        _start_time = time.time()
        n_cluster = _TEST['n_cluster']  # Số lượng cụm
        dpcm = DPCM(C= n_cluster, K=K, m=M, eps=epsilon, maxiter=MAX_ITER)
        test_points = _TEST['test_points']
        T1, V1, step = dpcm.fit(data=_dt['X'], init_v=init_V)
        
        print('#DPCM ----------------------------------------------------')
        # print("Số bước lặp:", step)
        print("V:", len(V1), V1)
        print("T:", len(T1), T1[:1])
        # print("Thời gian tính toán PCM:", round_float(time.time() - _start_time))
        titles = ['Alg', 'time', 'step', 'DI', 'DB', 'PC', 'PE', 'CE', 'CH', 'SI', 'FHV', 'CS', 'S']
        process_time = round_float(time.time() - _start_time)
        print(SPLIT.join(titles))
        print('-' * 100)
        print_info('PFCM', _dt['X'], T1, V1, process_time, M, step)

from ge