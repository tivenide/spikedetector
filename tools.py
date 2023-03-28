
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