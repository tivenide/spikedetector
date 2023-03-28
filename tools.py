
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

    # Compute the distance between each electrode location and each neuron location
    distances = np.sqrt(((electrode_locations.values[:, np.newaxis, :] - neuron_locations.values)**2).sum(axis=2))

    # Create an empty DataFrame to store the results
    assignments = pd.DataFrame(index=electrode_locations.index, columns=neuron_locations.index, dtype=bool)

    # Assign each channel position to its closest neuron_locations (if within the threshold distance)
    for i, point_idx in enumerate(neuron_locations.index):
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
