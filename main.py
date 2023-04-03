
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
# for labels use as input frame[:,1] (equals the second column)
results = sum_and_count(frame[:,1])
print("for labels:", results)

print('finish')