import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, frame, window_size):
        self.windows = []
        self.labels = []
        self.timestamps = []
        self.electrodes = []
        for row in frame:
            data = row[0]
            label = row[1]
            timestamps = row[2]
            electrodes = row[3]
            for i in range(len(data) - window_size + 1):
                window = data[i:i + window_size]
                self.windows.append(window)
                self.labels.append(label)
                self.timestamps.append(timestamps)
                self.electrodes.append(electrodes)

        self.windows = torch.tensor(self.windows, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.timestamps = torch.tensor(self.timestamps, dtype=torch.float32)
        self.electrodes = torch.tensor(self.electrodes, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx], self.timestamps[idx], self.electrodes[idx]

def get_window_size_of_frame(frame):
    return len(frame[0][0])
"""
frame = np.array([
    [[1, 2, 3],[0, 0, 0],[0, 0.1, 0.2],0],
    [[4, 5, 6],[0, 0, 1],[0.3, 0.4, 0.5],0],
    [[7, 8, 9],[0, 1, 1],[0.6, 0.7, 0.8],0]
], dtype=object)

window_size_of_frame = get_window_size_of_frame(frame)
dataset = TimeSeriesDataset(frame, window_size=window_size_of_frame)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for batch in dataloader:
    windows, labels, timestamps, electrodes = batch
    print('Windows:', windows)
    print('Labels:', labels)
    print('timestamps:', timestamps)
    print('Electrodes:', electrodes)
"""

def generate_demo_frame(n_windows=30, a=10, b=5):
    windows = np.empty((n_windows,), dtype=[
        ('arr1', np.float64, (a,)),
        ('arr2', np.int32, (a,)),
        ('arr3', np.float32, (a,)),
        ('int_val', np.int64),
        ('bool_val', np.bool_),
        ('arr4', np.float64, (b,))
    ])
    for i in range(n_windows):
        windows[i]['arr1'] = np.random.rand(a)
        windows[i]['arr2'] = np.random.randint(0, 10, size=a)
        windows[i]['arr3'] = np.random.rand(a).astype(np.float32)
        windows[i]['int_val'] = np.random.randint(0, 100)
        windows[i]['bool_val'] = np.random.choice([True, False])
        windows[i]['arr4'] = np.random.rand(b)
    return windows

def splitting_data(data, labels):
    from sklearn.model_selection import train_test_split
    X = data
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.2)
    return x_train, y_train, x_test, y_test, x_val, y_val

#frame = generate_demo_frame()
#data=frame['arr1']
#labels=frame['bool_val']
#x_train, y_train, x_test, y_test, x_val, y_val = splitting_data(data, labels)


def create_directory_structure(path):
    """
    Checks if a certain directory structure exists at the given path, and creates the structure if it doesn't.
    :param path: path to working directory.
    :return: None
    """
    import os
    # Define the directory structure you want to create
    directory_structure = [
        "data/raw/test",
        "data/raw/test",
        "data/raw/test",
        "data/save/after_normalization",
        "data/save/before_normalization"
    ]

    # Check if each directory in the structure exists, and create it if it doesn't
    for directory in directory_structure:
        directory_path = os.path.join(path, directory)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
        else:
            print(f"Directory already exists: {directory_path}")

#working_directory = "data"
#create_directory_structure(working_directory)

print("finish")