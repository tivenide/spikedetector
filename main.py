
def print_hi(name):
    print(f'Hi, {name}, Welcome to spikedetector')

if __name__ == '__main__':
    print_hi('Spike interested data engineer')

#for one recording
path_file='data/recording_allen_3sec_1000_1000_100_1000.h5'
#for multiple recordings
path_dir='data/'

from tools import preprocessing_for_one_recording
#preprocessing_for_one_recording(path_file)

from visualization import visualize_assignments_of_one_recording
#visualize_assignments_of_one_recording(path_file, 15)

from tools import preprocessing_for_multiple_recordings, normalize_frame
frame = preprocessing_for_multiple_recordings(path_dir)
#frame_normalized = normalize_frame(frame)

from tools import sum_and_count
# check if more than 1 spike occurs per window. For labels use as input frame[:,1] (equals the second column)
results = sum_and_count(frame[:,1])
print("for labels:", results)

from tools import convert_into_tensors_and_create_dataloader, get_window_size_of_frame, TimeSeriesDataset, create_data_loader, TimeSeriesDataset2
#dataloader = convert_into_tensors_and_create_dataloader(frame, batch_size=1)
#win_size = get_window_size_of_frame(frame)
#data_loader = create_data_loader(frame, win_size, batch_size=3)

import torch
from torch.utils.data import Dataset, DataLoader

# create a DataLoader
batch_size = 3
window_size = frame[0][0].shape[0]  # extract window size from the first window of the data
dataset = TimeSeriesDataset2(frame, window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for data in dataloader:
    inputs = data[0]  # windows
    labels = data[1]  # label vectors
    timestamps = data[2]  # time points
    electrodes = data[3]  # electrode numbers
    # Do something with the data...


"""
for batch in dataloader:
    windows, labels, timestamps, electrodes = batch
    print('Windows:', windows)
    print('Labels:', labels)
    print('timestamps:', timestamps)
    print('Electrodes:', electrodes)
"""

print('finish')