import os
import json
import numpy as np
import pandas as pd
from urllib import request, parse, error
import certifi
import ssl

TEST_CASES = {
    # 14: {
    #     'name': 'BreastCancer',
    #     'n_cluster': 2,
    #     'test_points': ['30-39', 'premeno', '30-34', '0-2', 'no', 3, 'left', 'left_low', 'no']
    # },
    53: {
        'name': 'Iris',
        'n_cluster': 3,
        'test_points': [5.1, 3.5, 1.4, 0.2]
    },
    # 80: {
    #     'name': 'Digits',
    #     'n_cluster': 10,
    #     'test_points': [0, 1, 6, 15, 12, 1, 0, 0, 0, 7, 16, 6, 6, 10, 0, 0, 0, 8, 16, 2, 0, 11, 2, 0, 0, 5, 16, 3, 0, 5, 7, 0, 0, 7, 13, 3, 0, 8, 7, 0, 0, 4, 12, 0, 1, 13, 5, 0, 0, 0, 14, 9, 15, 9, 0, 0, 0, 0, 6, 14, 7, 1, 0, 0]
    # },
    109: {
        'name': 'Wine',
        'n_cluster': 3,
        'test_points': [14.23, 1.71, 2.43, 15.6,
                        127, 2.80, 3.06, 0.28,
                        2.29, 5.64, 1.04, 3.92,
                        1065]
    },
    236: {
        'name': 'Seeds',
        'n_cluster': 3,
        'test_points': [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22]
    },
    602: {
        'name': 'DryBean',
        'n_cluster': 7,
        'test_points': [
            28395, 610.291, 208.178117, 173.888747,
            1.197191, 0.549812, 28715, 190.141097,
            0.763923, 0.988856, 0.958027, 0.913358,
            0.007332, 0.003147, 0.834222, 0.998724]
    }
}


# Mã hóa nhãn
class LabelEncoder:
    def __init__(self):
        self.index_to_label = {}
        self.unique_labels = None

    @property
    def classes_(self) -> np.ndarray:
        return self.unique_labels

    def fit_transform(self, labels) -> np.ndarray:
        self.unique_labels = np.unique(labels)
        label_to_index = {label: index for index, label in enumerate(self.unique_labels)}
        self.index_to_label = {index: label for label, index in label_to_index.items()}
        return np.array([label_to_index[label] for label in labels])

    def inverse_transform(self, indices) -> np.ndarray:
        return np.array([self.index_to_label[index] for index in indices])


def load_dataset(data: dict, file_csv: str = '', header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    # label_name = data['data']['target_col']
    print('DATASET UCI id=', data['data']['uci_id'], data['data']['name'], f"{data['data']['num_instances']} x {data['data']['num_features']}")  # Mã + Tên bộ dữ liệu
    # print('data abstract=', data['data']['abstract'])  # Tóm tắt bộ dữ liệu
    # print('feature types=', data['data']['feature_types'])  # Kiểu nhãn
    # print('num instances=', data['data']['num_instances'])  # Số lượng điểm dữ liệu
    # print('num features=', data['data']['num_features'])  # Số lượng đặc trưng
    metadata = data['data']
    # colnames = ['Area', 'Perimeter']
    df = pd.read_csv(file_csv if file_csv != '' else metadata['data_url'], header=header, index_col=index_col, usecols=usecols, nrows=nrows)
    # print('data top', df.head())  # Hiển thị một số dòng dữ liệu
    # Trích xuất ma trận đặc trưng X (loại trừ nhãn lớp)
    return {'data': data['data'], 'ALL': df.iloc[:, :].values, 'X': df.iloc[:, :-1].values, 'Y': df.iloc[:, -1:].values}


# Lấy dữ liệu từ ổ cứng
def fetch_data_from_local(name_or_id=53, folder: str = '/home/manh/Documents/gdrive/core/CoreFCM/dataset', header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    if isinstance(name_or_id, str):
        name = name_or_id
    else:
        name = TEST_CASES[name_or_id]['name']
    _folder = os.path.join(folder, name)
    fileio = os.path.join(_folder, 'api.json')
    if not os.path.isfile(fileio):
        print(f'File {fileio} not found!')
    with open(fileio, 'r') as cr:
        response = cr.read()
    return load_dataset(json.loads(response),
                        file_csv=os.path.join(_folder, 'data.csv'),
                        header=header, index_col=index_col, usecols=usecols, nrows=nrows)


# Lấy dữ liệu từ ISC UCI (53: Iris, 602: DryBean, 109: Wine)
def fetch_data_from_uci(name_or_id=53, header: int = 0, index_col: list = None, usecols: list = None, nrows: int = None) -> dict:
    api_url = 'https://archive.ics.uci.edu/api/dataset'
    if isinstance(name_or_id, str):
        api_url += '?name=' + parse.quote(name_or_id)
    else:
        api_url += '?id=' + str(name_or_id)
    try:
        _rs = request.urlopen(api_url, context=ssl.create_default_context(cafile=certifi.where()))
        response = _rs.read()
        _rs.close()
        return load_dataset(json.loads(response),
                            header=header, index_col=index_col, usecols=usecols, nrows=nrows)
    except (error.URLError, error.HTTPError):
        raise ConnectionError('Error connecting to server')


# Chia data, labels thành P datasite sao cho mỗi datasite có cùng số điểm dữ liệu và mỗi nhãn phải có ít nhất 1 điểm dữ liệu
# def split_data_balanced(data: np.ndarray, labels: np.ndarray, n_sites: int, seed: int = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
#     if seed is not None:
#         np.random.seed(seed)

#     # Kiểm tra số lượng điểm dữ liệu cho từng nhãn
#     _, label_counts = np.unique(labels, return_counts=True)

#     # Kiểm tra điều kiện mỗi nhãn phải có ít nhất 1 điểm ở mỗi datasite
#     if any(count < n_sites for count in label_counts):
#         raise ValueError("Không thể thỏa mãn điều kiện: Số điểm dữ liệu của ít nhất một nhãn nhỏ hơn số datasite")

#     # Kiểm tra và điều chỉnh số lượng điểm dữ liệu sao cho chia đều cho P
#     n_samples = len(data)
#     samples_to_remove = n_samples % n_sites
#     if samples_to_remove != 0:
#         n_samples -= samples_to_remove
#         data = data[:n_samples]
#         labels = labels[:n_samples]
#         print(f"Đã xóa {samples_to_remove} điểm dữ liệu để đảm bảo có thể chia đều cho {n_sites} datasite.")

#     # Nhóm lại các chỉ số dữ liệu theo nhãn và hoán vị ngẫu nhiên các chỉ số
#     all_indices = np.arange(n_samples)
#     np.random.shuffle(all_indices)

#     # Khởi tạo danh sách chứa dữ liệu cho mỗi site
#     samples_per_site = n_samples // n_sites
#     _datas, _labels = [], []

#     # Phân bổ dữ liệu vào các site một cách đều đặn
#     for i in range(n_sites):
#         start_idx = i * samples_per_site
#         end_idx = start_idx + samples_per_site

#         site_indices = all_indices[start_idx:end_idx]
#         _datas.append(data[site_indices])
#         _labels.append(labels[site_indices])

#     return _datas, _labels

def split_data_balanced(data: np.ndarray, labels: np.ndarray, n_sites: int, seed: int = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if seed is not None:
        np.random.seed(seed)

    # Kiểm tra và điều chỉnh số lượng điểm dữ liệu sao cho chia đều cho P
    n_samples = len(data)
    samples_to_remove = n_samples % n_sites
    if samples_to_remove != 0:
        n_samples -= samples_to_remove
        data = data[:n_samples]
        labels = labels[:n_samples]
        print(f"Đã xóa {samples_to_remove} điểm dữ liệu để đảm bảo có thể chia đều cho {n_sites} datasite => Còn lại {n_samples} điểm dữ liệu")

    # Nhóm các chỉ số dữ liệu theo nhãn
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_to_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    # Kiểm tra điều kiện mỗi nhãn phải có ít nhất 1 điểm ở mỗi datasite
    if any(count < n_sites for count in label_counts):
        raise ValueError("Không thể thỏa mãn điều kiện: Số điểm dữ liệu của ít nhất một nhãn nhỏ hơn số datasite")

    datas = [[] for _ in range(n_sites)]
    labeled = [[] for _ in range(n_sites)]

    # Phân phối điểm dữ liệu ban đầu để đảm bảo mỗi datasite có ít nhất 1 điểm của mỗi nhãn
    for label, indices in label_to_indices.items():
        np.random.shuffle(indices)
        for i in range(n_sites):
            datas[i].append(data[indices[i]])
            labeled[i].append(labels[indices[i]])

        # Remove the assigned indices
        label_to_indices[label] = indices[n_sites:]

    # Phân bổ các điểm còn lại một cách đều đặn cho các datasite
    remaining_indices = np.concatenate(list(label_to_indices.values()))
    np.random.shuffle(remaining_indices)
    samples_per_site = len(remaining_indices) // n_sites

    for i in range(n_sites):
        start_idx = i * samples_per_site
        end_idx = start_idx + samples_per_site
        site_indices = remaining_indices[start_idx:end_idx]
        datas[i].extend(data[site_indices])
        labeled[i].extend(labels[site_indices])

    datas = [np.array(site_data) for site_data in datas]
    labeled = [np.array(site_labels) for site_labels in labeled]

    return datas, labeled



# Xóa nhãn giữ lại labeled_percentage % số điểm dữ liệu có nhãn
def remove_label_for_semi_supervised(labels: np.ndarray, labeled_percentage: float, seed: int = None, no_label: int = -1) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed=seed)
    n_samples = len(labels)
    n_labeled = int(n_samples * labeled_percentage)

    mask = np.zeros(n_samples, dtype=bool)
    mask[:n_labeled] = True
    np.random.shuffle(mask)
    return np.where(mask, labels.astype(np.int8), no_label)  # -1 đại diện cho nhãn bị xóa


# Xóa nhãn cho tất cả các datasite có thể truyền vào site_unlabeled để xóa toàn bộ nhãn cho các site (cộng tác thường)
def remove_label_for_all_datasites(labels: list[np.ndarray], labeled_percentage: float, site_unlabeled: list[int] = None, seed: int = None, no_label: int = -1) -> list[np.ndarray]:
    result = []
    if site_unlabeled is None:
        for i in range(len(labels)):
            result.append(remove_label_for_semi_supervised(labels=labels[i], labeled_percentage=labeled_percentage, seed=seed, no_label=no_label))
    else:
        for i in range(len(labels)):
            if i not in site_unlabeled:
                result.append(remove_label_for_semi_supervised(labels=labels[i], labeled_percentage=labeled_percentage, seed=seed, no_label=no_label))
            else:
                result.append(np.full(labels[i].shape, no_label))
    return result


# Chia data, labels thành
def split_data_remove_label_for_collaborative(data: np.ndarray, labels: np.ndarray, n_sites: int, labeled_percentage: float, seed: int = None) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    datas, _labels = split_data_balanced(data=data, labels=labels, n_sites=n_sites, seed=seed)

    # Xóa bớt nhãn tại các datasites dựa vào labels theo tỉ lệ phần trăm
    labeled = [remove_label_for_semi_supervised(labels=label, labeled_percentage=labeled_percentage, seed=seed) for label in _labels]
    return datas, labeled, _labels
        
def read_BMI_data(file_csv: str = '/home/manhnv/Documents/gdrive/nckh/data/csv/bmi.csv') -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_csv)
    data = df.iloc[:, 0].values
    labels = df.iloc[:, 1].values
    return data, labels



if __name__ == '__main__':
    import time
    from f1.clustering.utility import round_float

    DATA_ID = 53
    P = 3
    LABELED_PERCENTAGE = 0.3
    NO_LABEL = -1
    SEED = 42
    SITE_UNLABELED = [0]

    if DATA_ID in TEST_CASES:
        _TEST = TEST_CASES[DATA_ID]
        _start_time = time.time()
        _dt = fetch_data_from_local(DATA_ID)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        n_cluster = _TEST['n_cluster']

        dlec = LabelEncoder()
        labels = dlec.fit_transform(_dt['Y'].flatten())
        X = _dt['X']
        datas, labels = split_data_balanced(data=X, labels=labels, n_sites=P)

        # Print shapes for each site
        for i in range(P):
            print(f"Site {i+1}:")
            print("Data shape:", datas[i].shape)
            print("Label shape:", labels[i].shape, ' - ', np.unique(labels[i]))
        labeled = remove_label_for_all_datasites(datas=datas, labels=labels, labeled_percentage=LABELED_PERCENTAGE, site_unlabeled=SITE_UNLABELED, no_label=NO_LABEL)
        for i in range(P):
            print(f"Site {i+1}:")
            print("Label shape:", labeled[i].shape, ' - ', np.unique(labeled[i]))