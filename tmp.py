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
import numpy as np


def create_random_array(n, p):
    """
    create a numpy array with random 0s and 1s with an user-defined ratio between the classes
    :param n: length of the output array
    :param p: p is the probability of choosing 1. The probability of choosing 0 is therefore 1-p.
    :return: array
    """
    # generate an array of size n with random values between 0 and 1
    random_array = np.random.rand(n)
    # create a mask array where values less than p are True, otherwise False
    mask = random_array < p
    # create an array of zeros and ones with True values as 1 and False as 0
    random_binary_array = np.zeros(n, dtype=int)
    random_binary_array[mask] = 1
    return random_binary_array


import numpy as np


def create_random_label(labels, probs):
    """
    create a random label with an user-defined probability for each label
    :param labels: a list of labels
    :param probs: a list of probabilities for each label
    :return: label
    """
    # generate a random value between 0 and 1
    random_value = np.random.rand()

    # loop through the given labels and probabilities
    for label, prob in zip(labels, probs):
        # if the random value is less than the probability, return the label
        if random_value < prob:
            return label
        # otherwise, subtract the probability from the random value and continue
        else:
            random_value -= prob

    # if none of the probabilities result in a label, return the last label
    return labels[-1]


def generate_demo_frame(n_windows=30, a=10, b=5):
    windows = np.empty((n_windows,), dtype=[
        ('arr1', np.float64, (a,)),
        ('arr2', np.int32, (a,)),
        ('arr3', np.float32, (a,)),
        ('int_val', np.int64),
        ('bool_val', np.bool_),
        ('arr4', np.float64, (b,)),
        ('label', np.float32)
    ])
    for i in range(n_windows):
        windows[i]['arr1'] = np.random.rand(a)
        windows[i]['arr2'] = np.random.randint(0, 10, size=a)
        windows[i]['arr3'] = np.random.rand(a).astype(np.float32)
        windows[i]['int_val'] = np.random.randint(0, 100)
        windows[i]['bool_val'] = np.random.choice([True, False])
        windows[i]['arr4'] = np.random.rand(b)
        windows[i]['label'] = create_random_label([0, 1], [0.999, 0.001])  # [0.8, 0.2] [0.99, 0.01]
    return windows

def generate_demo_frame_simple(n_windows=100, a=10, label_dist=[0.8,0.2]):
    windows = np.empty((n_windows,), dtype=[
        ('arr1', np.int32, (a,)),
        ('label', np.int32)
    ])
    for i in range(n_windows):
        windows[i]['arr1'] = np.random.randint(0, 10, size=a)
        windows[i]['label'] = create_random_label([0, 1], label_dist)  # [0.8, 0.2] [0.99, 0.01]
    return windows


def splitting_data(data, labels):
    from sklearn.model_selection import train_test_split
    X = data
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , stratify=y)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.2 , stratify=y_test)
    return x_train, y_train, x_test, y_test, x_val, y_val


def splitting_data_into_train_test_val_set(data, labels, test_and_val_size=0.6, val_size_of_test_and_val_size=0.5):
    """
    Splits data and labels into training, test and validation set.
    :param data: input set which contains data
    :param labels: input set which contains labels for data
    :param test_and_val_size: size of test and validation set combined. Rest equals training set.
    :param val_size_of_test_and_val_size: size of validation set corresponding to test_and_val_size. Rest equals test set.
    :return: training, test and validation set for data and labels
    """
    from sklearn.model_selection import train_test_split
    X = data
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_and_val_size , stratify=y)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test,
                                                    test_size=val_size_of_test_and_val_size , stratify=y_test)
    return x_train, y_train, x_test, y_test, x_val, y_val


# frame = generate_demo_frame()
# data=frame['arr1']
# labels=frame['bool_val']
# x_train, y_train, x_test, y_test, x_val, y_val = splitting_data(data, labels)


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


# working_directory = "data"
# create_directory_structure(working_directory)


def balance_classes(arr, ratio):
    import numpy as np
    from sklearn.utils import resample
    # Get the number of samples in each class
    count_class_0, count_class_1 = np.bincount(arr['label'].astype(np.int64))

    # Calculate the desired number of samples for each class
    count_class_majority = max(count_class_0, count_class_1)
    count_class_minority = min(count_class_0, count_class_1)
    count_class_minority_upsampled = int(count_class_majority * ratio)

    # If the majority class has fewer samples than the desired ratio, upsample it
    if count_class_majority < count_class_minority_upsampled:
        arr_majority = arr[arr['label'] == np.argmax([count_class_0, count_class_1])]
        arr_minority = arr[arr['label'] != np.argmax([count_class_0, count_class_1])]
        arr_majority_upsampled = resample(arr_majority, replace=True, n_samples=count_class_minority_upsampled,
                                          random_state=123)
        arr_balanced = np.concatenate([arr_majority_upsampled, arr_minority])

    # If the minority class has more samples than the desired ratio, downsample it
    elif count_class_minority > count_class_minority_upsampled:
        arr_minority = arr[arr['label'] == np.argmin([count_class_0, count_class_1])]
        arr_majority = arr[arr['label'] != np.argmin([count_class_0, count_class_1])]
        arr_minority_downsampled = resample(arr_minority, replace=False, n_samples=count_class_minority_upsampled,
                                            random_state=123)
        arr_balanced = np.concatenate([arr_majority, arr_minority_downsampled])

    # If the classes are already balanced, return the input array
    else:
        arr_balanced = arr

    return arr_balanced


import numpy as np


def balance_classes2(arr, ratio, oversampling=False):
    """
    Balance the classes of a numpy ndarray by undersampling or oversampling.

    Args:
        arr (numpy.ndarray): A numpy ndarray with a 'label' column representing the class labels.
        ratio (float): The desired ratio of the number of samples in the minority class to the number of samples in the majority class.
        oversampling (bool): Whether to use oversampling (True) or undersampling (False) to balance the classes.

    Returns:
        numpy.ndarray: A numpy ndarray with balanced class distribution.
    """
    # Find the minority and majority classes
    unique_classes, class_counts = np.unique(arr['label'], return_counts=True)
    minority_class_label = unique_classes[np.argmin(class_counts)]
    majority_class_label = unique_classes[np.argmax(class_counts)]

    # Determine the size of the minority and majority classes
    num_minority_samples = class_counts[np.argmin(class_counts)]
    num_majority_samples = class_counts[np.argmax(class_counts)]

    if oversampling:
        # Oversample the minority class
        num_samples_to_add = num_majority_samples - num_minority_samples  # int(np.round(num_majority_samples * ratio)) - num_minority_samples
        if num_samples_to_add > 0:
            minority_indices = np.where(arr['label'] == minority_class_label)[0]
            minority_samples = arr[minority_indices]
            samples_to_add = minority_samples[
                np.random.choice(minority_samples.shape[0], num_samples_to_add, replace=True)]
            arr = np.concatenate((arr, samples_to_add))
    else:
        # Undersample the majority class
        num_samples_to_remove = num_majority_samples - num_minority_samples  # num_majority_samples - int(np.round(num_minority_samples / ratio))
        if num_samples_to_remove > 0:
            majority_indices = np.where(arr['label'] == majority_class_label)[0]
            samples_to_remove = np.random.choice(majority_indices, size=num_samples_to_remove, replace=False)
            arr = np.delete(arr, samples_to_remove, axis=0)

    # Shuffle the array to mix up the classes
    np.random.shuffle(arr)

    return arr


import numpy as np
from sklearn.utils import resample


def balance_classes3(arr, ratio, oversampling=False):
    """
    Balance the classes of a numpy ndarray by undersampling or oversampling.

    Args:
        arr (numpy.ndarray): A numpy ndarray with a 'label' column representing the class labels.
        ratio (float): The desired ratio of the number of samples in the minority class to the number of samples in the majority class.
        oversampling (bool): Whether to use oversampling (True) or undersampling (False) to balance the classes.

    Returns:
        numpy.ndarray: A numpy ndarray with balanced class distribution.
    """
    # Find the minority and majority classes
    unique_classes, class_counts = np.unique(arr['label'], return_counts=True)
    minority_class_label = unique_classes[np.argmin(class_counts)]
    majority_class_label = unique_classes[np.argmax(class_counts)]

    # Determine the size of the minority and majority classes
    num_minority_samples = class_counts[np.argmin(class_counts)]
    num_majority_samples = class_counts[np.argmax(class_counts)]

    if oversampling:
        # Oversample the minority class
        num_samples_to_add = int(np.round(num_majority_samples * ratio)) - num_minority_samples
        if num_samples_to_add > 0:
            minority_indices = np.where(arr['label'] == minority_class_label)[0]
            minority_samples = arr[minority_indices]
            samples_to_add = resample(minority_samples, n_samples=num_samples_to_add, replace=True)
            arr = np.concatenate((arr, samples_to_add))
    else:
        # Undersample the majority class
        num_samples_to_remove = num_majority_samples - int(np.round(num_minority_samples / ratio))
        if num_samples_to_remove > 0:
            majority_indices = np.where(arr['label'] == majority_class_label)[0]
            majority_samples = arr[majority_indices]
            indices_to_remove = resample(majority_samples, n_samples=num_samples_to_remove, replace=False)
            arr = np.delete(arr, majority_indices[indices_to_remove], axis=0)

    # Shuffle the array to mix up the classes
    np.random.shuffle(arr)

    return arr


def count_label_appearances(array, labels):
    """
    count the number of appearances of each label value in an array
    :param array: a numpy array of labels
    :param labels: a list of possible labels
    :return: a dictionary of label counts, and a list of label probabilities
    """
    # get the total length of the array
    n = len(array)

    # create a dictionary to store the label counts
    label_counts = {label: 0 for label in labels}

    # loop through the array and increment the label counts for each appearance
    for label in array:
        label_counts[label] += 1

    # calculate the label probabilities based on the label counts
    label_probs = [label_counts[label] / n for label in labels]

    return label_counts, label_probs


def splitting_test_train_with_sklearn(data, labels, test_size):
    from sklearn.model_selection import train_test_split
    X = data
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    return x_train, x_test, y_train, y_test


def random_under_sampling_with_imblearn(data, labels):
    from imblearn.under_sampling import RandomUnderSampler
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X = data
    y = labels
    X_res, y_res = undersample.fit_resample(X, y)

    return X_res, y_res


def cropping_set(data, labels, cropping_size):
    from sklearn.model_selection import train_test_split
    X = data
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=cropping_size, stratify=y)
    return x_test, y_test


# new pipeline:
frame = generate_demo_frame_simple(20, a=3, label_dist=[0.7, 0.3])
"""
import numpy as np
a=3
frame = np.array([
    [[1,2,3],0],
    [[4,5,6],1]
], dtype=[
        ('arr1', np.int32, (a,)),
        ('label', np.int32)
    ])
"""
label_count_frame = count_label_appearances(frame['label'], [0, 1])
print('frame', label_count_frame)
x_train, y_train, x_test, y_test, x_val, y_val = splitting_data_into_train_test_val_set(frame['arr1'], frame['label'])
label_count_y_train = count_label_appearances(y_train, [0, 1])
print('y_train', label_count_y_train)
label_count_y_val = count_label_appearances(y_val, [0, 1])
print('y_val', label_count_y_val)
label_count_y_test = count_label_appearances(y_test, [0, 1])
print('y_test', label_count_y_test)

x_train_res, y_train_res = random_under_sampling_with_imblearn(x_train, y_train)
label_count_y_train_res = count_label_appearances(y_train_res, [0, 1])
print('y_train_res', label_count_y_train_res)

spikes_per_frame = (frame['label'].sum()) / (len(frame['label']))
imbalance_ratio = spikes_per_frame
print('ratio', imbalance_ratio)

x_val_crp, y_val_crp = cropping_set(x_val, y_val, imbalance_ratio)
label_count_y_val_crp = count_label_appearances(y_val_crp, [0, 1])
print('y_val_crp', label_count_y_val_crp)

x_test_crp, y_test_crp = cropping_set(x_test, y_test, imbalance_ratio)
label_count_y_test_crp = count_label_appearances(y_test_crp, [0, 1])
print('y_test_crp', label_count_y_test_crp)

# sum = frame['label'].sum()
"""
#new try:
import numpy as np
frame1 = generate_demo_frame(1_000_000)
frame2 = generate_demo_frame(1_000_000)
frame3 = generate_demo_frame(1_000_000)

print('original frames:')
label_counts_frame1, label_probs_frame1 = count_label_appearances(frame1['label'], [0, 1])
print('frame1', label_counts_frame1, label_probs_frame1)

label_counts_frame2, label_probs_frame2 = count_label_appearances(frame2['label'], [0, 1])
print('frame2', label_counts_frame2, label_probs_frame2)

label_counts_frame3, label_probs_frame3 = count_label_appearances(frame3['label'], [0, 1])
print('frame3', label_counts_frame3, label_probs_frame3)

print('starting resampling:')
frame1_X_res, frame1_y_res = random_under_sampling_with_imblearn(frame1['arr1'], frame1['label'])
label_counts_frame1_y_res, label_probs_frame1_y_res = count_label_appearances(frame1_y_res, [0, 1])
print('frame1_y_res', label_counts_frame1_y_res, label_probs_frame1_y_res)

frame2_X_res, frame2_y_res = random_under_sampling_with_imblearn(frame2['arr1'], frame2['label'])
label_counts_frame2_y_res, label_probs_frame2_y_res = count_label_appearances(frame2_y_res, [0, 1])
print('frame2_y_res', label_counts_frame2_y_res, label_probs_frame2_y_res)

frame1_X_res_frame2_X_res = np.vstack((frame1_X_res, frame2_X_res))
frame1_y_res_frame2_y_res = np.hstack((frame1_y_res, frame2_y_res))
label_counts_frame1_y_res_frame2_y_res, label_probs_frame1_y_res_frame2_y_res = count_label_appearances(frame1_y_res_frame2_y_res, [0,1])
print('frame1_y_res_frame2_y_res', label_counts_frame1_y_res_frame2_y_res, label_probs_frame1_y_res_frame2_y_res)

print('starting splitting')
x_a, x_b, y_a, y_b = splitting_test_train_with_sklearn(frame3['arr1'], frame3['label'], 0.01)
label_counts_frame3_y_a, label_probs_frame3_y_a = count_label_appearances(y_a, [0,1])
print('frame3_y_a', label_counts_frame3_y_a, label_probs_frame3_y_a)
label_counts_frame3_y_b, label_probs_frame3_y_b = count_label_appearances(y_b, [0,1])
print('frame3_y_b', label_counts_frame3_y_b, label_probs_frame3_y_b)


print('try finished')
"""
"""
data = frame['arr1']
labels = frame['label']
x_train, x_test, y_train, y_test = splitting_test_train(data, labels)
label_counts, label_probs = count_label_appearances(y_train, [0, 1])
print("y_train", label_counts, label_probs)
label_counts, label_probs = count_label_appearances(y_test, [0, 1])
print("y_test", label_counts, label_probs)
"""

# x_train, y_train, x_test, y_test, x_val, y_val = splitting_data(frame['arr1'], frame['label'])
# label_counts_train, label_prob_train = count_label_appearances(y_train, [0, 1])
# print(label_counts_train, label_prob_train)
# print('trainsum: ', y_train.sum(), 'length:', len(y_train))
"""
import imblearn
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy='majority')
X = frame['arr1']
y= frame['label']
X_over, y_over = undersample.fit_resample(x_train, y_train)
label_counts_after, label_probs_after = count_label_appearances(y_over, [0, 1])
#balanced_frame =  #balance_classes3(frame, ratio=1, oversampling=False)
#minority_balanced = balanced_frame['label'].sum()
print(label_counts_after,label_probs_after)
"""

"""
min_noise = 10
max_noise = 30
min_amp = 14
max_amp = 40
noise_step = 5
amp_step = 5


min_noise = 0
max_noise = 10
min_amp = 1
max_amp = 5
noise_step = 0.5
amp_step = 1

for noise_level in range(int(min_noise * 10), int(max_noise * 10) + 1, int(noise_step * 10)):
    noise_level = noise_level / 10.0  # Convert back to float value with one decimal place
    for min_amp in range(int((noise_level - 2) * 10), int(max_amp * 10) + 1, int(amp_step * 10)):
        min_amp = max(min_amp / 10.0, min_amp)  # Ensure min_amp is at least min_amp parameter given at the beginning
        for max_amp in range(int(min_amp * 10), int(max_amp * 10) + 1, int(amp_step * 10)):
            max_amp = min(max_amp / 10.0, max_amp)  # Ensure max_amp is at most max_amp parameter given at the beginning
            noise_level = max(min_noise, min(noise_level, max_noise))  # Ensure noise_level is within the range [min_noise, max_noise]
            print("Noise level:", noise_level, "Min amp:", min_amp, "Max amp:", max_amp)
"""


def generate_combinations(min_noise, max_noise, min_amp, max_amp, noise_step=1, amp_step=1):
    combinations = []
    for noise_level in range(min_noise, max_noise + 1, noise_step):
        for max_amp_level in range(min_amp, max_amp + 1, amp_step):
            if max_amp_level - noise_level <= 0:
                continue
            min_amp_level = max(2, noise_level - (max_amp_level - noise_level))
            for amp_level in range(min_amp_level, max_amp_level + 1, amp_step):
                combinations.append((noise_level, amp_level, max_amp_level))
    return combinations


"""
tada = generate_combinations(3, 10, 2, 20, 1, 1)
for item in tada:
    print(item)
"""
print("finish")
