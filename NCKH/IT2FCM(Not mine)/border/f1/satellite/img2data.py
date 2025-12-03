import numpy as np
import os
from rasterio import open as rasterio_open

class Dimg2data:
    @staticmethod
    def map_to_0_255(data: np.ndarray) -> np.ndarray:
        return ((data + 1) * 255 / 2).astype(np.uint8)

    @staticmethod
    def map_to_0_1(data: np.ndarray) -> np.ndarray:
        return data.astype(np.float32) / np.max(data)
    
    @staticmethod
    def __standard_scaler(bands: np.ndarray, normalize: bool = False) -> np.ndarray:
        if isinstance(bands, list):
            bands = np.array(bands)
        
        if bands.ndim == 3 and bands.shape[0] <= 3:
            bands = np.moveaxis(bands, 0, -1)
        
        if normalize:
            return Dimg2data.map_to_0_1(bands)
        return bands

    @staticmethod
    def read_multi_band_image(imgpath: str, normalize: bool = True) -> tuple:
        with rasterio_open(imgpath) as src:
            bands = src.read()
            height, width = src.height, src.width
            imd = Dimg2data.__standard_scaler(bands, normalize=normalize)
        return imd, height, width
    
    @staticmethod
    def rgb_to_grayscale(image_data: np.ndarray) -> np.ndarray:
        # ITU-R BT.601 weights for RGB to Grayscale conversion
        weights = np.array([0.2989, 0.5870, 0.1140])
        
        # Ensure the input is a 3D array (height, width, channels)
        if image_data.ndim != 3 or image_data.shape[2] != 3:
            raise ValueError("Input must be a 3D array with 3 channels (RGB)")
        
        # Apply the dot product along the last axis (color channels)
        grayscale_image = np.dot(image_data, weights)
        
        # Reshape to (x*x, 1)
        grayscale_image = grayscale_image.reshape(-1, 1)
        
        # Ensure the output is in the correct range and data type
        return np.clip(grayscale_image, 0, 255).astype(np.float32)

    def read_multi_band_directory(self, directory: str, normalize: bool = True) -> tuple:
        # Lấy danh sách tất cả các file ảnh trong thư mục
        image_files = [f for f in os.listdir(directory) if f.endswith(('.tif', '.jpg', '.png'))]
        image_files.sort()  # Sắp xếp để đảm bảo thứ tự nhất quán

        # Đọc từng ảnh và đưa vào ma trận dữ liệu
        _bands, height, width = [], 0, 0
        for image_file in image_files:
            imgpath = os.path.join(directory, image_file)
            with rasterio_open(imgpath) as src:
                _bands.append(src.read())
                if width == 0:
                    height, width = src.height, src.width
        # bands = np.concatenate(_bands, axis=0)
        imd = self.__standard_scaler(_bands, normalize=normalize)
        return imd, height, width

    def read_single_band_images(self, imgpaths: list, normalize: bool = True) -> dict:
        result = {'height': 0, 'width': 0, 'bands': []}
        for imgpath in imgpaths:
            with rasterio_open(imgpath) as src:
                _bands = src.read()
                if result['width'] == 0:
                    result.update({'height': src.height, 'width': src.width})
                result['bands'].append(self.__standard_scaler([_bands], normalize=normalize))
        return result
    
    # Lấy các bands từ những ảnh đơn phổ chỉ định
    @staticmethod
    def read_single_band_images(imgpaths: list, normalize: bool = True) -> dict:
        result = {'height': 0, 'width': 0, 'bands': []}
        for imgpath in imgpaths:
            with rasterio_open(imgpath) as src:
                _band = src.read()
                if result['width'] == 0:
                    result.update({'height': src.height, 'width': src.width})
                result['bands'].append(Dimg2data.__standard_scaler([_band], normalize=normalize))
        # ----------------------
        return result
    
    @staticmethod
    # Lấy ảnh NDVI từ 2 kênh phổ NIR & RED
    def read_ndvi_images(red_path: str, nir_path: str, empty: int = 0) -> dict:
        bimgs = Dimg2data.read_single_band_images(imgpaths=[red_path, nir_path], normalize=False)
        nir = bimgs['bands'][1].astype(float)
        red = bimgs['bands'][0].astype(float)

        # Tránh chia cho 0 bằng cách thêm epsilon, tránh trường hợp nir.astype(float) + red) rất nhỏ nhưng khác 0
        epsilon = 1e-10
        denominator = nir + red
        denominator = np.where(denominator < epsilon, epsilon, denominator)
        _data = (nir - red) / denominator

        print(_data.shape)

        if empty != 0:
            _data = np.array(_data[_data != empty])[:, None]
        bimgs['data'] = _data
        return bimgs