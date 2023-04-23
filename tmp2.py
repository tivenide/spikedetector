# STRIPE code
def splitting_data_into_train_test_val_set(data, labels, test_and_val_size=0.4, val_size_of_test_and_val_size=0.5):
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
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_and_val_size, stratify=y)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=val_size_of_test_and_val_size, stratify=y_test)
    return x_train, y_train, x_test, y_test, x_val, y_val

def balancing_dataset_with_undersampling(data, labels):
    """
    balancing dataset with random undersampling with sampling strategy 'majority'
    :param data: input data
    :param labels: corresponding labels for input data.
    :return: balanced data and labels (unshuffeled)
    """
    from imblearn.under_sampling import RandomUnderSampler
    print('balancing started')
    undersample = RandomUnderSampler(sampling_strategy='majority')
    data_result, labels_result = undersample.fit_resample(data, labels)
    print('balancing finished')
    return data_result, labels_result

def cropping_dataset(data, labels, cropping_size):
    """
    crop dataset to size of cropping_size to a subset
    :param data: input data
    :param labels: corresponding labels for input data.
    :param cropping_size: float between 0 and 1. proportion of resulting dataset from input dataset
    :return: cropped data and cropped labels
    """
    from sklearn.model_selection import train_test_split
    X = data
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=cropping_size, stratify=y)
    return x_test, y_test

def dataset_pipeline_for_training_process(frame, verbose=True):
    """
    pipeline for splitting, balancing and cropping datasets for training process
    :param frame: input ndarray which contains data and corresponding labels
    :param verbose: (bool) Default True. Prints proportions.
    :return: undersampled training set, cropped test set, cropped validation set
    """
    # init
    data = frame['signals']
    labels = frame['label_per_window']
    # splitting data
    x_train, y_train, x_test, y_test, x_val, y_val = splitting_data_into_train_test_val_set(data, labels)

    if verbose:
        print('frame: spikes:', labels.sum(), 'total:', len(labels))
        print('train: spikes:', y_train.sum(), 'total:', len(y_train))
        print('test: spikes:', y_test.sum(), 'total:', len(y_test))
        print('val: spikes:', y_val.sum(), 'total:', len(y_val))

    # undersample training set
    x_train_res, y_train_res = balancing_dataset_with_undersampling(x_train, y_train)
    # calculation of cropping size
    spikes_per_frame = (labels.sum()) / (len(labels))
    # cropping test set
    x_test_crp, y_test_crp = cropping_dataset(x_test, y_test, spikes_per_frame)
    # cropping validation set
    x_val_crp, y_val_crp = cropping_dataset(x_val, y_val, spikes_per_frame)

    if verbose:
        print('spikes_per_frame:', spikes_per_frame)
        print('train_res: spikes:', y_train_res.sum(), 'total:', len(y_train_res))
        print('test_crp: spikes:', y_test_crp.sum(), 'total:', len(y_test_crp))
        print('val_crp: spikes:', y_val_crp.sum(), 'total:', len(y_val_crp))

    return x_train_res, y_train_res, x_test_crp, y_test_crp, x_val_crp, y_val_crp

def save_frame_to_disk(frame, path_target):
    import numpy as np
    print('started saving frame to disk')
    np.save(path_target, frame, allow_pickle=True)
    print('frame saved to disk')

def load_frame_from_disk(path_source):
    import numpy as np
    print('started loading frame from disk')
    frame = np.load(path_source, allow_pickle=True)
    print('frame loaded from disk')
    return frame

# test env code
def create_random_label(labels, probs):
    """
    create a random label with an user-defined probability for each label
    :param labels: a list of labels
    :param probs: a list of probabilities for each label
    :return: label
    """
    import numpy as np
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

def generate_demo_frame_simple(n_windows=100, a=10, label_dist=[0.8,0.2]):
    import numpy as np
    windows = np.empty((n_windows,), dtype=[
        ('signals', np.int32, (a,)),
        ('label_per_window', np.int32)
    ])
    for i in range(n_windows):
        windows[i]['signals'] = np.random.randint(0, 10, size=a)
        windows[i]['label_per_window'] = create_random_label([0, 1], label_dist)  # [0.8, 0.2] [0.99, 0.01]
    return windows

# starting testing
"""
frame1 = generate_demo_frame_simple(100, 3, [0.7, 0.3])
frame1_x_train_res, frame1_y_train_res, frame1_x_test_crp, frame1_y_test_crp, frame1_x_val_crp, frame1_y_val_crp = dataset_pipeline_for_training_process(frame1)

frame2 = generate_demo_frame_simple(100, 3, [0.8, 0.2])
frame2_x_train_res, frame2_y_train_res, frame2_x_test_crp, frame2_y_test_crp, frame2_x_val_crp, frame2_y_val_crp = dataset_pipeline_for_training_process(frame2)

frame3 = generate_demo_frame_simple(100, 3, [0.6, 0.4])
frame3_x_train_res, frame3_y_train_res, frame3_x_test_crp, frame3_y_test_crp, frame3_x_val_crp, frame3_y_val_crp = dataset_pipeline_for_training_process(frame3)

save_frame_to_disk(frame1_x_train_res, 'save2/x/train/frame1_x_train_res.npy')
save_frame_to_disk(frame1_y_train_res, 'save2/y/train/frame1_y_train_res.npy')
save_frame_to_disk(frame1_x_test_crp, 'save2/x/test/frame1_x_test_crp.npy')
save_frame_to_disk(frame1_y_test_crp, 'save2/y/test/frame1_y_test_crp.npy')
save_frame_to_disk(frame1_x_val_crp,'save2/x/val/frame1_x_val_crp.npy')
save_frame_to_disk(frame1_y_val_crp,'save2/y/val/frame1_y_val_crp.npy')

save_frame_to_disk(frame2_x_train_res, 'save2/x/train/frame2_x_train_res.npy')
save_frame_to_disk(frame2_y_train_res, 'save2/y/train/frame2_y_train_res.npy')
save_frame_to_disk(frame2_x_test_crp, 'save2/x/test/frame2_x_test_crp.npy')
save_frame_to_disk(frame2_y_test_crp, 'save2/y/test/frame2_y_test_crp.npy')
save_frame_to_disk(frame2_x_val_crp,'save2/x/val/frame2_x_val_crp.npy')
save_frame_to_disk(frame2_y_val_crp,'save2/y/val/frame2_y_val_crp.npy')

save_frame_to_disk(frame3_x_train_res, 'save2/x/train/frame3_x_train_res.npy')
save_frame_to_disk(frame3_y_train_res, 'save2/y/train/frame3_y_train_res.npy')
save_frame_to_disk(frame3_x_test_crp, 'save2/x/test/frame3_x_test_crp.npy')
save_frame_to_disk(frame3_y_test_crp, 'save2/y/test/frame3_y_test_crp.npy')
save_frame_to_disk(frame3_x_val_crp,'save2/x/val/frame3_x_val_crp.npy')
save_frame_to_disk(frame3_y_val_crp,'save2/y/val/frame3_y_val_crp.npy')

"""
def loading_and_stacking_frame(path, vstack=False):
    """
    loads multiple frames (.npy-files) from disk and stacks them to one frame
    :param path: path to directory where frames are located. Only .npy-files with the same object types and structure are allowed.
    :param vstack: bool. Defines stacking method (vstack or hstack). Default: False.
    :return: stacked_frame
    """
    from pathlib import Path
    import numpy as np
    frames = [p for p in Path(path).iterdir()]
    stacked_frame = None
    for frm in frames:
        one_frame = load_frame_from_disk(frm)
        if stacked_frame is None:
            stacked_frame = one_frame.copy()
        else:
            if vstack is True:
                stacked_frame = np.vstack((stacked_frame, one_frame))
            elif vstack is False:
                stacked_frame = np.hstack((stacked_frame, one_frame))
    return stacked_frame

def dataset_pipeline_creating_even_larger_datasets(path_source, path_target=None, verbose=False):
    """
    Pipeline for creating even larger datasets for training process. Function uses dataset_pipeline_for_training_process()
    for splitting, balancing and cropping datasets.
    :param path_source: path to directory where frames are located. Only .npy-files with the same object types and
        structure are allowed. Data has to be in 'frame['signals']' and labels in 'frame['label_per_window']'.
        See for further requirements in dataset_pipeline_for_training_process().
    :param path_target: path to directory where results have to be saved to disk as .npy-files.
        Default None, so that no saving is done.
    :return: undersampled training set, cropped test set, cropped validation set from input frames
    """
    import os
    from pathlib import Path
    import numpy as np

    print('pipeline starts now, take a coffee :-) ')
    frames = [p for p in Path(path_source).iterdir()]

    frames_x_train_res = None
    frames_y_train_res = None
    frames_x_test_crp = None
    frames_y_test_crp = None
    frames_x_val_crp = None
    frames_y_val_crp = None

    for frm in frames:
        print('current frame:', frm)
        one_frame = load_frame_from_disk(frm)
        x_train_res, y_train_res, x_test_crp, y_test_crp, x_val_crp, y_val_crp = dataset_pipeline_for_training_process(one_frame, verbose)
        if frames_x_train_res is None:
            frames_x_train_res = x_train_res.copy()
            frames_y_train_res = y_train_res.copy()
            frames_x_test_crp = x_test_crp.copy()
            frames_y_test_crp = y_test_crp.copy()
            frames_x_val_crp = x_val_crp.copy()
            frames_y_val_crp = y_val_crp.copy()

        else:
            frames_x_train_res = np.vstack((frames_x_train_res, x_train_res))
            frames_y_train_res = np.hstack((frames_y_train_res, y_train_res))
            frames_x_test_crp = np.vstack((frames_x_test_crp, x_test_crp))
            frames_y_test_crp = np.hstack((frames_y_test_crp, y_test_crp))
            frames_x_val_crp = np.vstack((frames_x_val_crp, x_val_crp))
            frames_y_val_crp = np.hstack((frames_y_val_crp, y_val_crp))
    print('stacking finished')

    if path_target is not None:
        save_frame_to_disk(frames_x_train_res, os.path.join(path_target, 'frames_x_train_res.npy'))
        save_frame_to_disk(frames_y_train_res, os.path.join(path_target, 'frames_y_train_res.npy'))
        save_frame_to_disk(frames_x_test_crp, os.path.join(path_target, 'frames_x_test_crp.npy'))
        save_frame_to_disk(frames_y_test_crp, os.path.join(path_target, 'frames_y_test_crp.npy'))   
        save_frame_to_disk(frames_x_val_crp, os.path.join(path_target, 'frames_x_val_crp.npy'))
        save_frame_to_disk(frames_y_val_crp, os.path.join(path_target, 'frames_y_val_crp.npy'))
        print('successfully saved frames to disk in path:', path_target)
    else:
        print('no saving is done, continuing with returned arrays')
    return frames_x_train_res, frames_y_train_res, frames_x_test_crp, frames_y_test_crp, frames_x_val_crp, frames_y_val_crp

frame1 = generate_demo_frame_simple(100, 3)
save_frame_to_disk(frame1, 'save2/try/origin/frame1.npy')

frame2 = generate_demo_frame_simple(100, 3)
save_frame_to_disk(frame2, 'save2/try/origin/frame2.npy')

frame3 = generate_demo_frame_simple(100, 3)
save_frame_to_disk(frame3, 'save2/try/origin/frame3.npy')

frames_x_train_res, frames_y_train_res, frames_x_test_crp, frames_y_test_crp, frames_x_val_crp, frames_y_val_crp = dataset_pipeline_creating_even_larger_datasets('save2/try/origin', verbose=False)

stacked_frame_x_train = loading_and_stacking_frame('save2/x/train', True)
stacked_frame_y_train = loading_and_stacking_frame('save2/y/train', False)

import numpy as np
frame_x_train_res = np.vstack((frame1_x_train_res, frame2_x_train_res, frame3_x_train_res))
frame_y_train_res = np.hstack((frame1_y_train_res, frame2_y_train_res, frame3_y_train_res))
frame_x_test_crp = np.vstack((frame1_x_test_crp, frame2_x_test_crp, frame3_x_test_crp))
frame_y_test_crp = np.hstack((frame1_y_test_crp, frame2_y_test_crp, frame3_y_test_crp))
frame_x_val_crp = np.vstack((frame1_x_val_crp, frame2_x_val_crp, frame3_x_val_crp))
frame_y_val_crp = np.hstack((frame1_y_val_crp, frame2_y_val_crp, frame3_y_val_crp))

print('finish')