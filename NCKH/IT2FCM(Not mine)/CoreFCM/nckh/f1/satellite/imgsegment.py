import numpy as np
from skimage import io
from skimage.filters import threshold_multiotsu

COLORS = np.array([
    [0, 0, 255],        # Class 1: Rivers, lakes, ponds
    [128, 128, 128],    # Class 2: Vacant land, roads
    [0, 255, 0],        # Class 3: Field, grass
    [1, 192, 255],      # Class 4: Sparse forest, low trees
    [0, 128, 0],        # Class 5: Perennial Plants
    [0, 64, 0],         # Class 6: Dense forest, jungle
    # -----------------------
    # [255, 0, 0],    # Red
    # [255, 255, 0],  # Yellow
    # [255, 0, 255],  # Magenta
    # [0, 255, 255],  # Cyan
    # [128, 0, 0],    # Maroon
    # [0, 0, 128],    # Navy
    # [128, 128, 0],  # Olive
    # [0, 0, 0],      # Black
    # [255, 255, 255]  # White
], dtype=np.uint8)


class DimgSegment:

    @staticmethod
    def segment_image(img_path: str, num_classes: int = 6) -> tuple:
        # Xám hóa ảnh đầu vào
        _image = io.imread(img_path, as_gray=True)
        # Phân ngưỡng Otsu
        thresholds = threshold_multiotsu(_image, classes=num_classes)
        # Tạo ảnh phân đoạn (gán nhãn cho từng pixel dựa trên ngưỡng phân đoạn)
        segmented_image = np.digitize(_image, bins=thresholds)
        return segmented_image, thresholds

    @staticmethod
    def map_lables_to_colors(data: np.ndarray, color_palette: np.ndarray = None) -> tuple:
        if color_palette is None:
            color_palette = COLORS
        height, width = data.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        for i, color in enumerate(color_palette):
            rgb_image[data == i] = color  # labels của các đoạn với màu tương ứng
        return rgb_image, height, width

    @staticmethod
    def save_segmented_image(data: np.ndarray, output_path: str):
        io.imsave(output_path, data)

    @staticmethod
    def map_image_segmented_to_labels(img_path: str, color_palette: np.ndarray = None) -> tuple:
        if color_palette is None:
            color_palette = COLORS
        _segmented_image = io.imread(img_path)
        height, width = _segmented_image.shape[:2]
        labels = np.zeros((height, width), dtype=np.uint8)
        for i, color in enumerate(color_palette):
            labels[np.all(_segmented_image == color, axis=-1)] = i
        return labels, height, width

    @staticmethod
    def draw_image_from_data(data: np.ndarray, fileio: str, format: str = 'JPEG'):
        from PIL import Image
        image = Image.fromarray(data)
        image.save(fileio, format=format)
        return image

    @staticmethod
    def draw_gray_from_data(data: np.ndarray, height: int, width: int, fileio: str, format: str = 'JPEG'):
        out_shape = (height, width)
        segmented_image = data.reshape(out_shape)
        return DimgSegment.draw_image_from_data(segmented_image, fileio=fileio, format=format)

    @staticmethod
    def save_label2image(labels: np.ndarray, height: int, width: int, fileio: str, format: str = 'JPEG'):  # TIFF
        unique_lbs = np.unique(labels)
        C = len(unique_lbs)
        color_palette = COLORS
        # Nếu số cụm lớn hơn số màu trong bảng màu, lặp lại bảng màu
        if C > len(COLORS):
            # Lặp lại bảng màu cho đến khi đủ số cụm
            color_palette = np.tile(COLORS, (1 + C // len(COLORS), 1))
        color_palette = color_palette[:C]
        out_shape = (height, width)
        segmented_image = labels.reshape(out_shape)
        colored_segmented_image = color_palette[segmented_image]
        return DimgSegment.draw_image_from_data(data=colored_segmented_image, fileio=fileio, format=format)
