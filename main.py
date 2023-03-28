
def print_hi(name):
    print(f'Hi, {name}, Welcome to spikedetector')

if __name__ == '__main__':
    print_hi('Spike interested data engineer')

#for one recording
path='data/recording.h5'

from tools import preprocessing_for_one_recording
#preprocessing_for_one_recording(path)

from visualization import visualize_assignments_of_one_recording
#visualize_assignments_of_one_recording(path, 100)

print('finish')