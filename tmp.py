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

print("finish")