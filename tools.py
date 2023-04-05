
def import_recording_h5(path):
    """
    Import recording h5 file from MEArec.
    :param path: path to file
    :return: signal_raw, timestamps, ground_truth, channel_positions, template_locations
    """
    import h5py  # hdf5
    import numpy as np
    h5 = h5py.File(path, 'r')
    signal_raw = np.array(h5["recordings"])
    timestamps = np.array(h5["timestamps"])
    ground_truth = []
    for i in range(len(h5["spiketrains"].keys())):
        ground_truth.append(np.array(h5["spiketrains"][str(i)]["times"]))
    channel_positions = np.array(h5["channel_positions"]) #indexes of columns x: 1 y: 2 z: 0
    template_locations = np.array(h5["template_locations"]) #indexes of columns x: 1 y: 2 z: 0
    return signal_raw, timestamps, ground_truth, channel_positions, template_locations

def create_labels_for_spiketrain(timestamps, times):
    """
    Assign ground truth label of times to the nearest value of timestamps.
    :param timestamps: all timestamps
    :param times: ground truth timestamps of occurring spikes
    :return: labels: Returns list of length of timestamps with 1s at positions of times and 0s at the other positions.
    """
    import bisect
    import numpy as np
    labels = np.zeros(len(timestamps), dtype=int)
    times_sorted = np.sort(timestamps)
    for i, t in enumerate(times):
        index = bisect.bisect_left(times_sorted, t)
        if index == 0:
            nearest_timestamp = times_sorted[0]
        elif index == len(times_sorted):
            nearest_timestamp = times_sorted[-1]
        else:
            left_timestamp = times_sorted[index - 1]
            right_timestamp = times_sorted[index]
            if t - left_timestamp < right_timestamp - t:
                nearest_timestamp = left_timestamp
            else:
                nearest_timestamp = right_timestamp
        nearest_index = np.searchsorted(timestamps, nearest_timestamp)
        labels[nearest_index] = 1
    return labels

def create_labels_of_all_spiketrains(ground_truth, timestamps):
    """
    Create labels for all ground_truth spiketrains using create_labels_for_spiketrain()
    :param ground_truth:
    :param timestamps:
    :return: labels_of_all_spiketrains: Returns numpy array of all ground_truth spiketrains with 1s for a spike and
        0s otherwise.
    """
    import numpy as np
    labels_of_all_spiketrains = []
    for i in range(len(ground_truth)):
        labels = create_labels_for_spiketrain(timestamps, ground_truth[i])
        labels_of_all_spiketrains.append(labels)
    return np.array(labels_of_all_spiketrains)

def assign_neuron_locations_to_electrode_locations(electrode_locations, neuron_locations, threshold):
    """
    Assigns the index of a neuron location to the index of an electrode location if
    the distance between them is less than or equal to the threshold value.
    :param electrode_locations:
    :param neuron_locations:
    :param threshold: The maximum distance between an electrode location and a neuron location for them
        to be considered a match.
    :return:
    """
    import pandas as pd
    import numpy as np

    electrode_locations_df = pd.DataFrame(electrode_locations)
    neuron_locations_df = pd.DataFrame(neuron_locations)

    # Compute the distance between each electrode location and each neuron location
    distances = np.sqrt(((electrode_locations_df.values[:, np.newaxis, :] - neuron_locations_df.values)**2).sum(axis=2))

    # Create an empty DataFrame to store the results
    assignments = pd.DataFrame(index=electrode_locations_df.index, columns=neuron_locations_df.index, dtype=bool)

    # Assign each channel position to its closest neuron_locations (if within the threshold distance)
    for i, point_idx in enumerate(neuron_locations_df.index):
        mask = distances[:, i] <= threshold
        assignments.iloc[:, i] = mask

    return assignments

def merge_data_to_location_assignments(assignments, signal_raw, labels_of_all_spiketrains, timestamps):
    """
    Assigns the label vectors to the raw data. For the merging of multiple spiketrains to one electrode the
    np.logical_or() is used. For electrodes without an assignment to spiketrains empty spiketrains are generated.
    Additionally, timestamps are added.
    :param assignments: A DataFrame representing the local assignment between neurons and electrodes.
        With rows corresponding to electrodes and columns corresponding to neurons. Each cell in the
        DataFrame is True if the corresponding channel position is within the threshold distance of the
        corresponding neuron, and False otherwise. If a channel position is not assigned to any neuron position,
        the corresponding cells are False.
    :param signal_raw: A numpy array representing the recorded signal, with rows
        corresponding to electrodes of the MEA and columns corresponding to timestamps.
    :param labels_of_all_spiketrains: A numpy array representing the labels, with rows
        corresponding to spiketrains of the different neurons and columns corresponding to timestamps.
    :param timestamps:
    :return: merged_data: A numpy array representing the merged data. It's build like nested lists. The structure:
        [[[raw_data of the first electrode],[labels of the first electrode],[timestamps of the first electrode]],
        [[raw_data of the second electrode],[labels of the second electrode],[timestamps of the second electrode]], etc.]
    """
    import numpy as np

    assignments2 = np.array(assignments, dtype=bool)
    merged_data = []

    for i in range(assignments2.shape[0]):  # iterate over rows in assignments
        row = assignments2[i]  # equals electrode
        merged = np.zeros(len(labels_of_all_spiketrains[0]))  # generating empty spiketrains
        for j, value in enumerate(row):  # iterate over "columns" in rows
            if value:
                merged = np.logical_or(merged, labels_of_all_spiketrains[j, :])
        merged_data.append([signal_raw[i], merged.astype(int), timestamps])
    return np.array(merged_data)

def count_indexes_up_to_value(arr, value):
    import numpy as np
    # Find the indexes where the array values are less than or equal to the specified value
    indexes = np.where(arr <= value)[0]
    # Count the number of indexes
    count = len(indexes)
    return count
def get_window_size_in_index_count(timestamps, window_size_in_sec):
    """
    calculate window size in index counts from defined windowsize (in sec)
    :param timestamps: all timestamps (used for calculation)
    :param window_size_in_sec: windowsize in seconds
    :return: window_size_in_count
    """
    window_size_in_count = count_indexes_up_to_value(timestamps, window_size_in_sec)
    return window_size_in_count - 1


def devide_3_vectors_into_equal_windows_with_step(x1, x2, x3, window_size, step_size=None):
    """
    Devides vectors x1, x2, x3 into windows with one window_size. step_size is used to generate more windows with overlap.
    :param x1: Input list to be devided.
    :param x2: Input list to be devided.
    :param x3: Input list to be devided.
    :param window_size: Size of each window.
    :param step_size: If the step_size is not provided, it defaults to the window_size.
        If the step_size is set to True, it is set to half of the window_size.
        If the step_size is set to any other value, it is used directly as the step_size.
    :return: Returns for every input a list of lists. Each included list represents a window.
    """
    if step_size is None:
        step_size = window_size
    elif step_size is True:
        step_size = window_size // 2
    x1_windows = []
    x2_windows = []
    x3_windows = []
    for i in range(0, len(x1) - window_size + 1, step_size):
        x1_windows.append(x1[i:i + window_size])
        x2_windows.append(x2[i:i + window_size])
        x3_windows.append(x3[i:i + window_size])
    return x1_windows, x2_windows, x3_windows

def application_of_windowing(merged_data, window_size, step_size=None):
    """
    Application of windowing
    :param merged_data:
    :param window_size:
    :param step_size:
    :return: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw, labels, timestamps and electrode number.
    """
    import numpy as np

    frame = []
    for i in range(len(merged_data)):
        win1, win2, win3 = devide_3_vectors_into_equal_windows_with_step(merged_data[i][0], merged_data[i][1], merged_data[i][2], window_size, step_size)
        for l in range(len(win1)):
            frame.append(np.array([win1[l], win2[l], win3[l], i], dtype=object))
    return np.array(frame)

def preprocessing_for_one_recording(path, window_size_in_sec=0.002):
    """
    preprocessing pipeline for one recording (without normalization)
    :param path: path to recording file
    :return: frame: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw, labels, timestamps and electrode number.
    """
    signal_raw, timestamps, ground_truth, electrode_locations, neuron_locations = import_recording_h5(path)
    labels_of_all_spiketrains = create_labels_of_all_spiketrains(ground_truth, timestamps)
    assignments = assign_neuron_locations_to_electrode_locations(electrode_locations, neuron_locations, 20)
    merged_data = merge_data_to_location_assignments(assignments, signal_raw.transpose(), labels_of_all_spiketrains, timestamps)
    window_size_in_counts = get_window_size_in_index_count(timestamps, window_size_in_sec)
    frame = application_of_windowing(merged_data, window_size=window_size_in_counts, step_size=None)
    print('preprocessing finished for:', path)
    return frame

def preprocessing_for_multiple_recordings(path):
    """
    preprocessing pipeline for multiple recordings (without normalization)
    :param path: path to recording files. Only MEArec generated h5. recording files may be located here.
    :return: frame_of_multiple_recordings: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw, labels, timestamps and electrode number. No assignment to the recording!
    """
    from pathlib import Path
    import numpy as np
    recordings = [p for p in Path(path).iterdir()]
    frame_of_multiple_recordings = None
    print('preprocessing started for:', path)
    for rec in recordings:
        frame_of_one_recording = preprocessing_for_one_recording(rec)
        if frame_of_multiple_recordings is None:
            frame_of_multiple_recordings = frame_of_one_recording.copy()
        else:
            frame_of_multiple_recordings = np.vstack((frame_of_multiple_recordings, frame_of_one_recording))
    print('preprocessing finished for:', path)
    return frame_of_multiple_recordings

def normalize_frame(frame, scaler_type='minmax'):
    """
    Normalizes the raw data in the input array using the specified scaler type
    :param frame: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw, labels, timestamps and electrode number. No assignment to the recording!
    :param scaler_type: possible Scalers from sklearn.preprocessing: StandardScaler, MinMaxScaler, RobustScaler
    :return: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw (normalized), labels, timestamps and electrode number. No assignment to the recording!
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Scaler type {scaler_type} not supported. Please choose 'standard', 'minmax', or 'robust'")

    for i in frame:
        data_raw = i[0]
        data_norm = scaler.fit_transform(data_raw.reshape(-1,1))
        i[0] = data_norm.flatten()
    print(f"Normalization with scaler type '{scaler_type}' finished")
    return frame

def normalize_frame_whole(frame, scaler_type='minmax'):
    """
    Normalizes the raw data in the input array using the specified scaler type. Optimized for long input array,
    but uses the entire dataset to compute the scaler parameters and normalizes in one "batch".
    :param frame: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw, labels, timestamps and electrode number. No assignment to the recording!
    :param scaler_type: possible Scalers from sklearn.preprocessing: StandardScaler, MinMaxScaler, RobustScaler
    :return: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw (normalized), labels, timestamps and electrode number. No assignment to the recording!
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Scaler type {scaler_type} not supported. Please choose 'standard', 'minmax', or 'robust'")

    # Concatenate all raw data into a single 2D array
    data_raw_all = np.concatenate([i[0] for i in frame])
    # Compute the scaler parameters on the concatenated raw data
    scaler.fit(data_raw_all.reshape(-1, 1))

    for i in frame:
        data_raw = i[0]
        data_norm = scaler.transform(data_raw.reshape(-1,1))
        i[0] = data_norm.flatten()
    print(f"Normalization with scaler type '{scaler_type}' finished")
    return frame

def sum_and_count(arrays):
    """
    sums up arrays and counts the appearance of each sum
    :param arrays: Input array with nested arrays. Eg.: arrays = [[1, 2, 3], [4, 5, 6], [1, 2, 3]]
    :return: results (defaultdict) which can be printed out like this:
        results = sum_and_count(arrays)
        print(results)
    """
    from collections import defaultdict
    results = defaultdict(int)
    for array in arrays:
        results[sum(array)] += 1
    return results

## Data Loader
#https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html
#@TODO: data loader works but is not applicaple

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
                window_data = data[i:i + window_size]
                window_label = label[i:i + window_size]
                window_timestamp = timestamps[i:i + window_size]
                self.windows.append(window_data)
                self.labels.append(window_label)
                self.timestamps.append(window_timestamp)
                self.electrodes.append(electrodes)

        self.windows = torch.tensor(self.windows, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.timestamps = torch.tensor(self.timestamps, dtype=torch.float32)
        self.electrodes = torch.tensor(self.electrodes, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    """def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx], self.timestamps[idx], self.electrodes[idx]"""

    def __getitem__(self, idx):
        # Get the current index
        window = self.windows[idx]
        label = self.labels[idx]
        timestamp = self.timestamps[idx]
        electrode = self.electrodes[idx]

        # Convert to a single NumPy array
        window_array = np.array(window)
        label_array = np.array(label)
        timestamp_array = np.array(timestamp)
        electrode_array = np.array(electrode)

        # Convert to PyTorch tensors
        window_tensor = torch.from_numpy(window_array).float()
        label_tensor = torch.from_numpy(label_array).int()
        timestamp_tensor = torch.from_numpy(timestamp_array).float()
        electrode_tensor = torch.from_numpy(electrode_array).int()

        return window_tensor, label_tensor, timestamp_tensor, electrode_tensor

def get_window_size_of_frame(frame):
    return len(frame[0][0])

def convert_into_tensors_and_create_dataloader(frame, batch_size):
    import torch
    from torch.utils.data import Dataset, DataLoader
    window_size_of_frame = get_window_size_of_frame(frame)
    dataset = TimeSeriesDataset(frame, window_size=window_size_of_frame)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Finished conversion into tensors. Data is loaded.")
    return dataloader

def create_data_loader(frame, window_size, batch_size):
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    # Extract raw data and labels from frame
    data = frame[:, 0]
    labels = frame[:, 1]
    timestamps = frame[:,2]
    electrodes = frame[:,3]

    # Convert the list of windows to single NumPy arrays for each "column"
    data_array = np.array([data[i:i+window_size] for i in range(len(data) - window_size + 1)], dtype=np.float32)
    label_array = np.array([labels[i:i+window_size] for i in range(len(labels) - window_size + 1)], dtype=np.int)
    timestamp_array = np.array([timestamps[i:i+window_size] for i in range(len(timestamps) - window_size + 1)], dtype=np.float32)
    electrodes_array = np.array(electrodes, dtype=np.int)

    # Create a PyTorch TensorDataset from the window and label arrays
    dataset = Dataset(torch(data_array), torch.from_numpy(label_array), torch.from_numpy(timestamp_array), torch.from_numpy(electrodes_array))

    # Create a PyTorch DataLoader from the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return data_loader


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
class TimeSeriesDataset2(Dataset):
    def __init__(self, data, window_size):
        self.window_size = window_size
        self.data = data[:, 0]  # extract data
        self.labels = data[:, 1]  # extract label vectors
        self.timestamps = data[:, 2]  # extract time points
        self.electrodes = data[:, 3]  # extract electrode numbers

    def __len__(self):
        return len(self.data) - self.window_size + 1

"""    def __getitem__(self, idx):
        window = np.array(self.data[idx:idx + self.window_size]).astype(np.float32)
        label = np.array(self.labels[idx:idx + self.window_size]).astype(np.float32)
        timestamp = np.array(self.timestamps[idx:idx + self.window_size]).astype(np.float32)
        electrode = np.array(self.electrodes[idx:idx + self.window_size]).astype(np.float32)"""

"""        window = self.data[idx:idx + self.window_size]
        label = self.labels[idx:idx + self.window_size]
        timestamp = self.timestamps[idx:idx + self.window_size]
        electrode = self.electrodes[idx:idx + self.window_size]"""


"""        window = np.array(window).astype(np.float32)
        label = np.array(label).astype(np.float32)
        timestamp = np.array(timestamp).astype(np.float32)
        electrode = np.array(electrode).astype(np.float32)"""
"""
        # Convert the window and label to PyTorch tensors
        window_tensor = torch.from_numpy(window).float()
        label_tensor = torch.from_numpy(label).float()
        timestamp_tensor = torch.from_numpy(timestamp).float()
        electrode_tensor = torch.from_numpy(electrode).float()
"""
        #return torch.FloatTensor(window), torch.FloatTensor(label), torch.FloatTensor(timestamp), torch.FloatTensor(electrode)
        #return window_tensor, label_tensor, timestamp_tensor, electrode_tensor


