import time
import numpy as np
from bssfcm import DbssFcm
from f1.clustering.cmeans.s2cfc import Ds2CFC
from bfcm import DbFcm


class Dbs2CFC(Ds2CFC):
    # Giai đoạn 1: Phân cụm cục bộ tại các datasites
    def phase1(self, datas: list[np.ndarray], labeled: list[np.ndarray]):
        self._ds = []
        for i, data in enumerate(datas):
            ds = DbssFcm(n_clusters=self._n_clusters,
                         m=self._m, epsilon=self._epsilon, max_iter=self._max_iter,
                         index=i, metric=self._metric, no_label=self._no_label)
            # ds = DbFcm(n_clusters=self._n_clusters,
            #            m=self._m, epsilon=self._epsilon, max_iter=self._max_iter,
            #            index=i, metric=self._metric)
            ds.fit(data=data, labeled=labeled[i])
            # ds.fit(data=data)
            ds.exited = False
            self._ds.append(ds)


if __name__ == '__main__':
    import os
    from f1.clustering.utility import round_float, extract_labels, distance_cdist
    from f1.clustering.validity import davies_bouldin, partition_coefficient, classification_entropy, hypervolume
    from f1.clustering.validity import cs, separation, calinski_harabasz, f1_score, accuracy_score, partition_entropy, Xie_Benie
    from f1.clustering.dataset import split_data_balanced, remove_label_for_all_datasites
    from f1.satellite.imgsegment import DimgSegment
    from f1.satellite.img2data import Dimg2data
    from f1.clustering.utility import name_slug


    # info ================================
    ALG = 'BS2CFCM'
    DATASET = 'NDVI-Hanoi-1024'
    VERSION = '1.1'
    AUTHOR = 'ManhNV & HoangNX'
    DATE = '15/11/24 20:00'

    # params ================================
    C = 6
    EPSILON = 1e-5 
    MAXITER = 100
    SEED = 42
    ROUND_FLOAT = 3
    LABELED_RATIOS = 0.7
    M = 2
    P = 4
    GAMMA = 0.005
    ALPHA = 1
    DELTA = 1e-3    
    NO_LABEL = -1
    METRIC = 'euclidean'
    n_space = 12

    OUTPUT_FOLDER = os.path.join(os.getcwd(),'satellite/')
    IMG_DATA = os.path.join(os.getcwd(),'data/images')
    IMG_PATH = os.path.join(os.getcwd(),'satellite\\semi_labels\\1024x1024_6labels.tif')

    IMG_5000_b3 = 'E:\\Python\\NCKH\\IT2FCM\\b1_1024x1024.tif'
    IMG_5000_b4 = 'E:\\Python\\NCKH\\IT2FCM\\b2_1024x1024.tif'

    IMG_1024_b3 = 'E:\\Python\\NCKH\\IT2FCM\\b3_1024x1024.tif'
    IMG_1024_b4 = 'E:\\Python\\NCKH\\IT2FCM\\b4_1024x1024.tif'

    # ===============================================

    SPLIT = '\t'
    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def print_info(title: str, X: np.ndarray, U: np.ndarray, V: np.ndarray, y_true: np.ndarray, process_time: float, step: int = 0, split: str = SPLIT) -> str:
        labels = extract_labels(U)  # Giai mo
        kqdg = [
            title,
            wdvl(process_time),
            str(step),
            wdvl(davies_bouldin(X, labels)),  # DB
            wdvl(partition_coefficient(U)),  # PC
            wdvl(classification_entropy(U)),  # CE
            # wdvl(silhouette(X, labels)),  # SI
            wdvl(hypervolume(U, M)),  # FHV
            wdvl(cs(X, U, V, M)),  # CS
            wdvl(separation(X, U, V, M), n=0),  # S
            wdvl(calinski_harabasz(X, labels)),  # CH
            wdvl(f1_score(y_true, labels, average='macro')),  # F1
            wdvl(accuracy_score(y_true, labels))
        ]
        result = split.join(kqdg)
        print(result)
        return result

    def write_report_fcm(X: np.ndarray, V: np.ndarray, U: np.ndarray, y_true: np.ndarray, time_process: float, step: int) -> str:
        labels = extract_labels(U)  # Giai mo
        kqdg = {
            'DB': round_float(davies_bouldin(X, labels), n=ROUND_FLOAT),
            'PC': round_float(partition_coefficient(U), n=ROUND_FLOAT),
            # 'PC_I': round_float(partition_coefficient_I(U), n=ROUND_FLOAT),
            'PE': round_float(partition_entropy(U), n=ROUND_FLOAT),
            'XB': round_float(Xie_Benie(X, V, U), n=ROUND_FLOAT),
            'F1': round_float(f1_score(y_true, labels), n=ROUND_FLOAT),
            'AC': round_float(accuracy_score(y_true, labels), n=ROUND_FLOAT)
        }
        return '{time_process:<{n_space}} {step:<{n_space}} {DB:<{n_space}} {PC:<{n_space}} {PE:<{n_space}} {XB:<{n_space}} {F1:<{n_space}} {AC:<{n_space}}'.format(time_process=round_float(time_process, n=ROUND_FLOAT), step=step, **kqdg, n_space=n_space)

    # =======================================
    def tinh_toan_ndvi() -> tuple:
        _dsi = Dimg2data()
        bimgs = _dsi.read_single_band_images(imgpaths=[os.path.join(IMG_DATA, IMG_1024_b3),
                                                       os.path.join(IMG_DATA, IMG_1024_b4)])
        height, width = bimgs['height'], bimgs['width']
        b4, b3 = bimgs['bands'][1], bimgs['bands'][0]
        result = np.where((b4.astype(float) + b3) != 0, (b4.astype(float) - b3) / (b4.astype(float) + b3), 0)
        # print('max', np.max(ndvi), 'min', np.min(ndvi))
        # print('ndvi', result.shape, result[0])
        # Vẽ ảnh xám ------------------
        data2draw = Dimg2data.map_to_0_255(data=result)
        # print('data2draw', data2draw.shape, data2draw[0])
        # Tạo ảnh NDVI từ dữ liệu đã chuẩn hóa
        _imgpath = os.path.join(OUTPUT_FOLDER, 'ndvi_image.jpg')
        DimgSegment().draw_gray_from_data(data=data2draw, height=height, width=width, fileio=_imgpath)
        return result, height, width
    # =======================================
    
    def fuzzy_draw_image(X: np.ndarray, V: np.ndarray, U: np.ndarray, height: int, width: int, title: str) -> str:
        labels = extract_labels(U)
        _img_fileio = os.path.join(OUTPUT_FOLDER, f'{name_slug(title)}.jpg')
        DimgSegment().save_label2image(labels=labels, height=height, width=width, fileio=_img_fileio)

    # =======================================
    _start = time.time()
    ndvi, height, width = tinh_toan_ndvi()
    X2 = Dimg2data.map_to_0_1(data=ndvi)
    prefix = 'NDVI-KM'
    print(f"Thời gian đọc ảnh {prefix}=", round_float(time.time() - _start))


    labels, height, width = DimgSegment().map_image_segmented_to_labels(img_path=IMG_PATH)
    labels = labels.flatten()

    X2 = X2.reshape(X2.shape[2] * X2.shape[3], 1)
    datas, labels = split_data_balanced(data=X2, labels=labels, n_sites=P, seed=42)
    labeleds = remove_label_for_all_datasites(labels=labels, labeled_percentage=LABELED_RATIOS, seed=42, no_label=NO_LABEL)
    
    # ============================================
    print(f'Alg {ALG}')
    print(f'V{VERSION}, H-{AUTHOR} - Update {DATE}')
    print(f'Dataset {DATASET}')
    print(f'LABELED_RATIOS= {LABELED_RATIOS}')

    titles = ['Alg', 'time', 'step', 'DB-', 'PC+', 'CE-', 'CH+', 'FHV-', 'CS+', 'S+', '', 'F1+', 'ACC+']
    print(SPLIT.join(titles))

    bs2cfcm = Dbs2CFC(n_clusters=C, n_sites=P, gamma=GAMMA, delta=DELTA, alpha=ALPHA,
                 m=M, epsilon=EPSILON, max_iter=MAXITER, metric=METRIC, no_label=NO_LABEL)
    
    bs2cfcm.phase1(datas=datas, labeled=labeleds)
    print('#BSSFCM ----------------------------------------')
    for i, dsi in enumerate(bs2cfcm.datasites):
        print_info(title=f'DS{i}',
                    X=datas[i],
                    U=dsi.membership,
                    V=dsi.centroids,
                    process_time=dsi.process_time,
                    y_true=labels[i],
                    step=dsi.step)

    print('#BS2CFCM ----------------------------------------')
    v_means = []
    bs2cfcm.phase2()
    for i, dsi in enumerate(bs2cfcm.datasites):
        print_info(title=f'DS{i}',
                    X=datas[i],
                    U=dsi.membership,
                    V=dsi.centroids,
                    process_time=dsi.process_time,
                    y_true=labels[i],
                    step=dsi.step)
        
# python nckh/f1/clustering/border/bs2cfc.py
