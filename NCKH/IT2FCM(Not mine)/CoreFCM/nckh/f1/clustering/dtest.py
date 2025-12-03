import os
import numpy as np
from itertools import permutations
from f1.clustering.utility import round_float, name_slug, numpy_save, map_to_0_1
from f1.clustering.dataset import remove_label_for_semi_supervised
from f1.satellite.imgsegment import DimgSegment
from f1.satellite.img2data import Dimg2data
from f1.clustering.validity import davies_bouldin, partition_coefficient, classification_entropy, hypervolume, mean_distance_cluster
from f1.clustering.validity import mean_distance_cluster, mean_distance_point_cluster, accuracy_score


class Dtest():
    def __init__(self, dataset_path: str = '/home/dll/ncs/clustering/dataset/images', output_folder: str = '/home/PUBLIC', m: float = 2, seed: int = None, field_split: str = '&', thumb_width: int = 500):
        self.m = m
        self.seed = seed
        self.dataset_path = dataset_path
        self.output_folder = output_folder
        self.export_folder = self.output_folder
        self.thumb_width = thumb_width
        self.field_split = field_split

    @staticmethod
    def mkdir(folder: str = '') -> bool:
        if not os.path.isdir(folder):
            os.umask(0)
            os.makedirs(folder)
            return True
        return False

    def set_export_folder(self, folder: str):
        self.export_folder = os.path.join(self.output_folder, folder)
        self.mkdir(self.export_folder)

    def __viet_str(self, val: float, n: int = 3) -> str:
        return str(round_float(val, n=n))

    @staticmethod
    def sinh_hoan_vi_mau(colors: list) -> dict:
        result = {}
        _permutations = list(permutations([i for i, _c in enumerate(colors)]))
        for idx, perms in enumerate(_permutations, start=1):
            lbs, colors = [], []
            for perm in perms:
                lbs.append(str(perm))
                colors.append(colors[perm])
            _lbls = '_'.join(lbs)
            result[f'{idx}x{_lbls}'] = [colors[perm] for perm in perms]
        return result

    def ve_anh(self, title: str, labels: np.ndarray, height: int = 0, width: int = 0, colors: list = None, save_color: bool = True) -> np.ndarray:
        try:
            _fileio = os.path.join(self.export_folder, f'{name_slug(title)}_{width}x{height}')
            if self.thumb_width > 0:
                _wthumb = min(width, self.thumb_width)
                DimgSegment().save_label2image(labels=labels, height=height, width=width, format='PNG',
                                               output=f'{_fileio}_thumb_{_wthumb}.png', width_thumb=_wthumb)
            pcolors = DimgSegment().save_label2image(labels=labels, height=height, width=width, output=f'{_fileio}.tif', width_thumb=0, colors=colors)
            if save_color:
                numpy_save(f'{_fileio}_color.txt', pcolors, fmt='%.0f')
            return pcolors
        except:
            print('Ve anh loi')

    def chi_so_danh_gia(self, title: str, data: np.ndarray,
                        centroids: np.ndarray, labels: np.ndarray, membership: np.ndarray = None,
                        process_time: float = 0, step: int = 0,
                        y_true: np.ndarray = None, save_v: bool = True, save_u: bool = False) -> str:
        if save_u:
            _np_fname_u = os.path.join(self.export_folder, f'{name_slug(title)}_u.txt')
            numpy_save(_np_fname_u, membership)
        if save_v:
            _np_fname_v = os.path.join(self.export_folder, f'{name_slug(title)}_v.txt')
            numpy_save(_np_fname_v, centroids)
        # danh gia chat luong phan cum
        kqdg = [
            title,
            self.__viet_str(process_time),
            str(step),
            self.__viet_str(davies_bouldin(data, labels)),  # DB-
            self.__viet_str(partition_coefficient(membership)) if membership is not None else '-',  # PC+
            self.__viet_str(classification_entropy(membership)) if membership is not None else '-',  # CE-
            self.__viet_str(hypervolume(membership, m=self.m)) if membership is not None else '-',  # FHV-
            self.__viet_str(mean_distance_cluster(centroids=centroids)),  # ABC+
            self.__viet_str(mean_distance_point_cluster(data=data, centroids=centroids)),  # ADC-
            self.__viet_str(accuracy_score(y_true, labels)) if y_true is not None else '-'  # AC+
        ]
        return self.field_split.join(kqdg)

    def tinh_toan_anh_mau(self, dia_diem: str = 'Anh-mau/HaNoi.tif') -> tuple:
        _dsi = Dimg2data()
        img_path = os.path.join(self.dataset_path, dia_diem)
        rmbi = _dsi.read_multi_band_image(imgpath=img_path, normalize=True)
        height, width = rmbi['height'], rmbi['width']
        num_clusters = 6
        return num_clusters, rmbi['data'], height, width

    def tinh_toan_anh_da_pho(self, dia_diem: str = 'Landsat8/HaNoi-30') -> tuple:  # Anh-da-pho/HaNoi
        _dsi = Dimg2data()
        img_folder = os.path.join(self.dataset_path, dia_diem)
        mbd = _dsi.read_satellite_from_directory(folder=img_folder, normalize=True)
        num_clusters = 5
        return num_clusters, mbd['data'], mbd['height'], mbd['width']

    def tinh_toan_gray(self, dia_diem: str = 'Anh-mau/HaNoi.tif', ve_anh: bool = True) -> tuple:
        imgpath = os.path.join(self.dataset_path, dia_diem)
        _dsi = Dimg2data()
        rir = _dsi.read_image_rgb2gray(imgpath)
        # Vẽ ảnh xám ------------------
        if ve_anh:
            _N, _D = rir['data'].shape
            print('Ve anh GRAY xam', _N, _D)
            DimgSegment().draw_gray_from_data(data=rir['data'], height=rir['height'], width=rir['width'],
                                              output=os.path.join(self.export_folder, f'gray_image_{_N}_{_D}.tif'),
                                              width_thumb=self.thumb_width)
        num_clusters = 6
        return num_clusters, map_to_0_1(data=rir['data']), rir['height'], rir['width']

    def tinh_toan_ndvi(self, imgsize: int = 1024, ve_anh: bool = True) -> tuple:
        num_clusters = 6
        _dsi = Dimg2data()
        if imgsize == 298:
            bimgs = _dsi.read_ndvi_images(red_path=os.path.join(self.dataset_path, f'Landsat8/HaNoi-150/b4_150_{imgsize}x261.tif'),
                                          nir_path=os.path.join(self.dataset_path, f'Landsat8/HaNoi-150/b5_150_{imgsize}x261.tif'))
            num_clusters = 6
        elif imgsize == 1024:
            bimgs = _dsi.read_ndvi_images(red_path=os.path.join(self.dataset_path, f'Anh-da-pho/HaNoi/b3_{imgsize}x1024.tif'),
                                          nir_path=os.path.join(self.dataset_path, f'Anh-da-pho/HaNoi/b4_{imgsize}x1024.tif'))
            num_clusters = 6
        elif imgsize == 5000:
            bimgs = _dsi.read_ndvi_images(red_path=os.path.join(self.dataset_path, f'Landsat7/Bac-Bo-30-09-2009/b3_{imgsize}x5000.tif'),
                                          nir_path=os.path.join(self.dataset_path, f'Landsat7/Bac-Bo-30-09-2009/b4_{imgsize}x5000.tif'))
            num_clusters = 6
        elif imgsize == 5567:
            bimgs = _dsi.read_ndvi_images(red_path=os.path.join(self.dataset_path, f'Landsat8/HaNoi-20/b4_20_{imgsize}x5014.tif'),
                                          nir_path=os.path.join(self.dataset_path, f'Landsat8/HaNoi-20/b5_20_{imgsize}x5014.tif'))
            num_clusters = 6
        elif imgsize == 3711:
            bimgs = _dsi.read_ndvi_images(red_path=os.path.join(self.dataset_path, f'Landsat8/HaNoi-30/b4_30_{imgsize}x3343.tif'),
                                          nir_path=os.path.join(self.dataset_path, f'Landsat8/HaNoi-30/b5_30_{imgsize}x3343.tif'))
            num_clusters = 5
        elif imgsize == 10000:
            bimgs = _dsi.read_ndvi_images(red_path=os.path.join(self.dataset_path, f'Landsat7/Dong-Nam-Bo-09-10-2012/b3_{imgsize}x10000.tif'),
                                          nir_path=os.path.join(self.dataset_path, f'Landsat7/Dong-Nam-Bo-09-10-2012/b4_{imgsize}x10000.tif'))
            num_clusters = 6
        elif imgsize == 11132:
            bimgs = _dsi.read_ndvi_images(red_path=os.path.join(self.dataset_path, f'Landsat8/HaNoi-10/b4_10_{imgsize}x10028.tif'),
                                          nir_path=os.path.join(self.dataset_path, f'Landsat8/HaNoi-10/b5_10_{imgsize}x10028.tif'))
            num_clusters = 6
        # ------------------
        height, width = bimgs['height'], bimgs['width']
        # Vẽ ảnh xám ------------------
        if ve_anh:
            _N, _D = bimgs['data'].shape
            data2draw = Dimg2data.map_to_0_255(data=bimgs['data'])
            _path = os.path.join(self.export_folder, f'ndvi_image_{_N}_{_D}.tif')
            print('Ve anh NDVI xam', _N, _D, _path)
            # Tạo ảnh NDVI từ dữ liệu đã chuẩn hóa
            DimgSegment().draw_gray_from_data(data=data2draw, height=height, width=width,
                                              output=_path,
                                              width_thumb=self.thumb_width)
        return num_clusters, bimgs['data'], height, width

    def anh_co_nhan(self, imgsize: int = 1024) -> str:
        if imgsize in [1024, 5000, 10000]:
            imgpath = f'Labels/sub_hn6class_{imgsize}.tif'
        elif imgsize == 298:
            imgpath = f'Landsat8/HaNoi-150/Labeled_150_otsu_{imgsize}x261.tif'
        elif imgsize == 5567:
            imgpath = f'Landsat8/HaNoi-20/Labeled_20_otsu_{imgsize}x5014.tif'
        elif imgsize == 11132:
            imgpath = f'Landsat8/HaNoi-10/Labeled_10_otsu_{imgsize}x10028.tif'
        return os.path.join(self.dataset_path, imgpath)

    def anh_ban_giam_sat(self, imgsize: int = 1024, labeled_ratio: float = 0.7, no_labeled=-1) -> tuple:
        _img_path = self.anh_co_nhan(imgsize)
        lbdatas, height, width = DimgSegment().map_image_segmented_to_labels(img_path=_img_path)
        y_true = lbdatas.flatten()
        semi_labeled = remove_label_for_semi_supervised(labels=y_true,
                                                        labeled_percentage=labeled_ratio,
                                                        seed=self.seed,
                                                        no_label=no_labeled)
        return height, width, semi_labeled, y_true

    # ===========================================================
    def image_tif_path(self, region: str, name: str, zoom_ratio: int = 200):  # 200,8,15
        return os.path.join(self.dataset_path, f'Collab/{region}/z{zoom_ratio}_{name}.tif')

    def load_ndvi(self, region: str) -> tuple:
        bimgs = Dimg2data().read_ndvi_images(red_path=self.image_tif_path(region, name='B4'),
                                             nir_path=self.image_tif_path(region, name='B5'))
        height, width = bimgs['height'], bimgs['width']
        return bimgs['data'], height, width

    def image_labeled(self, region: str, labeled_ratio: float = 0.3, no_label: int = -1, seed: int = None):
        path = self.image_tif_path(region, name='OTSU_6class')
        lbdatas, height, width = DimgSegment.map_image_segmented_to_labels(img_path=path)
        y_true = lbdatas.flatten()
        semi_labeled = remove_label_for_semi_supervised(labels=y_true,
                                                        labeled_percentage=labeled_ratio,
                                                        seed=seed,
                                                        no_label=no_label)
        return height, width, semi_labeled, y_true
