import re
import numpy as np
COLORS = ['Blue', 'Orange', 'Green', 'Red', 'Cyan', 'Yellow', 'Purple', 'Pink', 'Brown', 'Black', 'Gray', 'Beige', 'Turquoise', 'Silver', 'Gold']


# =======================================
def numpy_load(filepath: str) -> np.ndarray:
    return np.loadtxt(filepath)


def numpy_save(filepath: str, data: np.ndarray, fmt: str = '%.18e', delimiter: str = ' '):
    np.savetxt(filepath, data, fmt=fmt, delimiter=delimiter)


# Ánh xạ mã mầu cho vẽ ảnh sau phân cụm dựa trên tâm cụm và bảng mầu của ảnh đã vẽ chuẩn
def map_centroids_to_colors(new_centroids: np.ndarray, centroids_path: str, color_path: str) -> np.ndarray:
    _centroids_standard = numpy_load(centroids_path)
    # print('centroids_standard', _centroids_standard.shape)
    # print(_centroids_standard)

    # print('new_centroids', new_centroids.shape)
    # print(new_centroids)
    # Tính khoảng cách giữa các centroids mới với centroids chuẩn
    distances = distance_cdist(new_centroids, _centroids_standard)

    # Khởi tạo mảng lưu kết quả ánh xạ giữa color_palette và new_color_palette
    C = new_centroids.shape[0]

    # Tạo mảng lưu index kết quả ánh xạ
    _kqaxs = np.full(C, -1)
    for _ in range(C):
        # Lấy ra vị trí của khoảng cách nhỏ nhất (min_index)
        new_centroid_index, standard_centroid_index = np.unravel_index(distances.argmin(), distances.shape)
        _kqaxs[new_centroid_index] = standard_centroid_index

        # Gán lại khoảng cách tới new_centroid_index là vô cực -> đã được xét
        distances[new_centroid_index, :] = np.inf

        # Gán lại khoảng cách tới điểm standard_centroid_index là vô cực -> đã được xét
        distances[:, standard_centroid_index] = np.inf

    _color_palette = numpy_load(color_path)
    # print('color_palette', _color_palette)
    # print('kq', _color_palette[_kqaxs])
    return _color_palette[_kqaxs]


def name_slug(text: str, delim: str = '-') -> str:
    __punct_re = re.compile(r'[\t !’"“”#@$%&~\'()*\+:;\-/<=>?\[\\\]^_`{|},.]+')
    if text:
        from unidecode import unidecode
        result = [unidecode(word) for word in __punct_re.split(text.lower()) if word]
        result = [rs if rs != delim and rs.isalnum() else '' for rs in result]
        return re.sub(r'\s+', delim, delim.join(result).strip())


# Hàm ánh xạ 1 điểm trong khoảng [0,1]
def map_to_0_1(data: np.ndarray) -> np.ndarray:
    return data.astype(np.float32) / np.max(data)


# Làm tròn số
def round_float(number: float, n: int = 3) -> float:
    if n == 0:
        return int(number)
    return round(number, n)


# Ma trận độ thuộc ra nhãn (giải mờ)
def extract_labels(membership: np.ndarray) -> np.ndarray:
    return np.argmax(membership, axis=1)


# Ma trận chi phí giữa nhãn dự đoán và nhãn thực tế
def compute_confusion_matrix(labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(labels, pred_labels)
    # C = len(np.unique(labels))
    # result = np.zeros((C, C), dtype=int)
    # for i in range(len(labels)):
    #     predicted_cluster = pred_labels[i]          # Cụm dự đoán (C_i)
    #     true_label = labels[i]                      # Nhãn thực (L_j)
    #     result[predicted_cluster, true_label] += 1  # Tăng số đếm tại C_ij
    # return result.T


# Lấy ma trận nhãn tối ưu sau khi giải mờ để tính AC
def optimized_pred_labels(labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    # Tạo ma trận confusion matrix giữa nhãn dự đoán và nhãn thực tế
    cm = compute_confusion_matrix(labels, pred_labels)

    # Sử dụng thuật toán Hungarian để tìm ánh xạ tối ưu
    row_ind, col_ind = linear_sum_assignment(-cm)   # Đảo dấu để tối đa hóa tổng

    # Tạo nhãn mới cho dự đoán dựa trên ánh xạ tối ưu
    result = np.zeros_like(pred_labels)
    for i, j in zip(row_ind, col_ind):
        result[pred_labels == j] = i
    return result


# Sắp xếp tối ưu các ma trận tâm cụm để cùng tính chất
def reorder_centroids(list_centroids: list) -> list:
    from scipy.optimize import linear_sum_assignment
    _v_ref = list_centroids[0]
    newvs = [_v_ref]                                # lấy thước ngắm là tâm cụm 0
    for i in range(1, len(list_centroids)):         # lặp cho P-1 datasite còn lại
        _v = list_centroids[i]
        cost_matrix = distance_cdist(_v_ref, _v)    # ma trận chi phí là ma trận khoảng cách vref tới v
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # đảo thứ tự 2 tâm để đảm bảo đồng tính chất (tối ưu khoảng cách)
        _v_ot = np.zeros_like(_v)
        for i, j in zip(row_ind, col_ind):
            _v_ot[_v == j] = i
        newvs.append(_v_ot)
    return newvs


# Chia các điểm vào các cụm
def extract_clusters(data: np.ndarray, labels: np.ndarray, n_clusters: int = 0) -> list:
    if n_clusters == 0:
        n_clusters = np.unique(labels)
    return [data[labels == i] for i in range(n_clusters)]


# Chuẩn Euclidean của một vector đo lường độ dài của vector
# là căn bậc hai của tổng bình phương các phần tử của vector đó.
# d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
def norm_distances(A: np.ndarray, B: np.ndarray, axis: int = None) -> float:
    # np.sqrt(np.sum((np.asarray(A) - np.asarray(B)) ** 2))
    # np.sum(np.abs(np.array(A) - np.array(B)))
    return np.linalg.norm(A - B, axis=axis)


def minus_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X[:, np.newaxis, :] - Y[np.newaxis, :, :]


def distance_euclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    distance = minus_distances(X, Y)
    return np.sqrt(np.sum(distance ** 2, axis=2))


def distance_chebyshev(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    distance = minus_distances(X, Y)
    return np.max(np.abs(distance), axis=2)


def distance_l21(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    distance = minus_distances(X, Y)
    return np.sum(np.sqrt(np.abs(distance)), axis=2)


# Ma trận khoảng cách Euclide giữa các điểm trong 2 tập hợp dữ liệu
def distance_cdist(X: np.ndarray, Y: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    # return distance_euclidean(X,Y) if metric=='euclidean' else distance_chebyshev(X,Y)
    from scipy.spatial.distance import cdist
    return cdist(X, Y, metric=metric)


# Khoảng cách của 2 cặp điểm trong một ma trận
def distance_pdist(data: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    from scipy.spatial.distance import pdist
    return pdist(data, metric=metric)


# Chia hết cho 0
def division_by_zero(data: np.ndarray) -> np.ndarray:
    data[data == 0] = np.finfo(float).eps
    return data


# lấy giá trị lớn nhất để tránh lỗi chia cho 0
def not_division_by_zero(data: np.ndarray):
    return np.fmax(data, np.finfo(np.float64).eps)


# Chuẩn hóa mỗi hàng của ma trận sao cho tổng của mỗi hàng bằng 1.
# \mathbf{x}_{norm} = \frac{\mathbf{x}}{\sum_{i=1}^m \mathbf{x}_{i,:}}
def standardize_rows(data: np.ndarray) -> np.ndarray:
    # Ma trận tổng của mỗi cột (cùng số chiều)
    _sum = np.sum(data, axis=0, keepdims=1)
    # Chia từng phần tử của ma trận cho tổng tương ứng của cột đó.
    return data / _sum


# Đếm số lần xuất hiện của từng phần tử trong 1 mảng
def count_data_array(data: np.ndarray) -> dict:
    unique_elements, counts = np.unique(data, return_counts=True)
    return {int(element): int(count) for element, count in zip(unique_elements, counts)}


# Giảm chiều PCA
def pca(data: np.ndarray, k: int, sklearn: bool = True) -> np.ndarray:
    if sklearn:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=k)
        return pca.fit_transform(data)

    '''Thực hiện từng bước trong PCA produre'''
    # Tính mean từng feature
    data_mean = np.mean(data, axis=0)

    # Thực hiện chuẩn hóa bằng cách data - mean
    data_bar = data - data_mean

    # Tính ma trận hiệp phương sai
    # covariance_matrix = (1/(len(data_bar)))*(data_bar.T @ data_bar)
    covariance_matrix = np.cov(data_bar, rowvar=False)

    # Tính các giá trị riêng và vector riêng
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # Sắp xếp các giá trị riêng và vector riêng theo thứ tự giảm dần của vector riêng
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Giữ lại k thành phần chính (k là số chiều mới)
    principal_components = eigenvectors[:, :k]
    # Chiếu dữ liệu ban đầu vào không gian mới
    return data_bar @ principal_components


# Sử dụng KneeLocator để tìm điểm khuỷu tay
def choose_clusters_auto(wcss: list, min_cluster: int = 2, max_cluster: int = 10) -> int:
    from kneed import KneeLocator
    kneedle = KneeLocator(range(min_cluster, max_cluster), wcss, curve='convex', direction='decreasing')
    return int(kneedle.elbow)


def draw_matplot(title: str, C: int, data: np.ndarray, labels: np.ndarray, centroids: np.ndarray, x_label: str, y_label: str, save2img: str = ''):
    import matplotlib.pyplot as plt
    plt.subplots()
    for i in range(C):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], color=COLORS[i], label=f'Cum {i+1}')

    plt.scatter(centroids[0], centroids[1], color='red', marker='x', label='Tam cum')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    if save2img:
        plt.savefig(save2img)


def show_matplot():
    import matplotlib.pyplot as plt
    plt.show()

# Sắp xếp tối ưu các ma trận tâm cụm để cùng tính chất
def reorder_centroids(list_centroids: list) -> list:
    from scipy.optimize import linear_sum_assignment
    _v_ref = list_centroids[0]  # lấy thước ngắm là tâm cụm 0
    newvs = [_v_ref]
    for i in range(1, len(list_centroids)):  # lặp cho P-1 dataset còn lại
        _v = list_centroids[i]
        cost_matrix = distance_cdist(_v_ref, _v)  # ma trận chi phí là ma trận khoảng cách vref tới v
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # đảo thứ tự 2 tâm để đảm bảo đồng tính chất (tối ưu khoảng cách)
        _v_ot = np.zeros_like(_v)
        for i, j in zip(row_ind, col_ind):
            _v_ot[_v == j] = i
        newvs.append(_v_ot)
    return newvs


def compute_normalized_laplacian(data: np.ndarray, k: int = 10, sigma: float = 1.0):
    from scipy.sparse import csgraph
    from sklearn.neighbors import kneighbors_graph
    """
    Tính ma trận Laplace chuẩn hóa từ dữ liệu đầu vào.

    Parameters:
    - data: numpy.ndarray, dữ liệu đầu vào (mỗi hàng là một mẫu).
    - k: int, số k-láng giềng để xây dựng ma trận trọng số W.
    - sigma: float, tham số cho Gaussian kernel.

    Returns:
    - W: numpy.ndarray, ma trận trọng số.
    - L: numpy.ndarray, ma trận Laplacian chuẩn hóa.
    """
    # Tính ma trận trọng số W sử dụng k-láng giềng
    W = kneighbors_graph(data, k, mode='connectivity', include_self=False).toarray()
    
    # Thêm Gaussian kernel
    distances = distance_cdist(data, data)  # Khoảng cách giữa các mẫu
    W = np.exp(-distances ** 2 / (2 * sigma ** 2)) * W  # Chỉ giữ các k-láng giềng

    # Tính ma trận Laplacian chuẩn hóa
    L = csgraph.laplacian(W, normed=True)

    return L

