from Data.data import TEST_CASES, fetch_data_from_uci, round_float
from Phan_cum.FPCM import Dfpcm
from validity import print_info, SPLIT

import numpy as np

if __name__ == "__main__":
    import time
    _start_time = time.time()
    MAX_ITER = 10000  # 000
    DATA_ID = 53  # 53: Iris, 109: Wine, 602: DryBean
    epsilon = 1e-5
    SEED = 42
    K = 1
    M = 2
    A = 1
    B = 1
    ETA = 2
    if DATA_ID in TEST_CASES:
        _TEST = TEST_CASES[DATA_ID]
        _dt = fetch_data_from_uci(DATA_ID)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        # --------------------------------
        # init_V = np.array([[7.2, 3.0, 5.8, 1.6], [5.5, 2.4, 3.7, 1.0], [5.10, 3.3, 1.70, 0.50]])
        init_V = np.array([[5.8, 2.7, 5.1, 1.9], [5.0, 2.0, 3.5, 1.0], [5.10, 3.50, 1.40, 0.30]])
        _start_time = time.time()
        n_cluster = _TEST['n_cluster']  # Số lượng cụm
        dfpcm = Dfpcm(C= n_cluster, eta=ETA, K=K, m=M, eps=epsilon, maxiter=MAX_ITER)
        test_points = _TEST['test_points']  
        U1, V1, step, T1 = dfpcm.fit(data=_dt['X'], init_v=init_V, seed=SEED)
        
        print('#DFPCM ----------------------------------------------------')
        # print("Số bước lặp:", step)
        print("V:", len(V1), V1)
        print("U:", len(U1), U1[:1])
        print("T:", len(T1), np.sum(T1.T[:1]))
        # print("Thời gian tính toán PFCM:", round_float(time.time() - _start_time))
        # print_metrics(_dt['X'], _dt['y'], U1, V1)
        titles = ['Alg', 'time', 'step', 'DI', 'DB', 'PC', 'PE', 'CE', 'CH', 'SI', 'FHV', 'CS', 'S']
        process_time = round_float(time.time() - _start_time)
        print(SPLIT.join(titles))
        print('-' * 100)
        print_info('FPCM', _dt['X'], U1, V1, process_time, M, step)