# classical detection approaches
# see at the end of the file the pipeline

def filter_data(data, method='bandpass', fs=10000, f_low=300, f_high=3000, order=4):
    """
    Filter data using different filtering methods.

    Parameters:
    -----------
    data: numpy array
        The data to be filtered.
    method: str
        The filtering method to be used. Available methods are 'bandpass', 'bandstop', 'lowpass', and 'highpass'.
    fs: float
        The sampling frequency of the data.
    f_low: float
        The lower cutoff frequency for the filter.
    f_high: float
        The higher cutoff frequency for the filter.
    order: int
        The order of the filter.

    Returns:
    --------
    filtered_data: numpy array
        The filtered data.
    """
    import numpy as np
    from scipy import signal

    nyquist = 0.5 * fs

    if method == 'bandpass':
        b, a = signal.butter(order, [f_low / nyquist, f_high / nyquist], btype='bandpass')
    elif method == 'bandstop':
        b, a = signal.butter(order, [f_low / nyquist, f_high / nyquist], btype='bandstop')
    elif method == 'lowpass':
        b, a = signal.butter(order, f_high / nyquist, btype='lowpass')
    elif method == 'highpass':
        b, a = signal.butter(order, f_low / nyquist, btype='highpass')
    else:
        raise ValueError("Invalid filtering method.")

    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def check_window_for_spikes(signal_window, std_threshold=5):
    """
    Checks if a signal window is spike-free by fitting its histogram with a Gaussian distribution.

    Parameters
    ----------
    signal_window : numpy.ndarray
        The signal window to analyze.
    std_threshold : float, optional
        The maximum standard deviation allowed for the Gaussian fit to be considered pure noise.

    Returns
    -------
    numpy.ndarray
        The input signal window, if it is spike-free. Otherwise, a window of the same length containing 0s.
    """
    import numpy as np
    from scipy.stats import norm
    from scipy.optimize import curve_fit

    # Calculate the histogram of the signal window
    hist, bin_edges = np.histogram(signal_window, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Define the Gaussian function to fit the histogram
    def gaussian_func(x, a, mu, sigma):
        return a * norm.pdf(x, loc=mu, scale=sigma)

    # Fit the histogram with the Gaussian function
    try:
        popt, pcov = curve_fit(gaussian_func, bin_centers, hist)
        mu, sigma = popt[1], popt[2]
    except RuntimeError:
        return np.zeros_like(signal_window)

    # Check if the standard deviation is less than or equal to the threshold
    if sigma <= std_threshold:
        return signal_window
    else:
        # If the standard deviation is greater than the threshold, return a window of the same length containing 0s
        return np.zeros_like(signal_window)


def find_spike_free_windows_simplified(signal_data, window_size=500, step_size=500, std_threshold=5):
    """
    Finds spike-free windows in a signal by iterating over it with a sliding window.

    Parameters
    ----------
    signal_data : numpy.ndarray
        The signal data to analyze.
    window_size : int
        The size of the sliding window to use.
    step_size : int
        The number of samples to slide the window at each iteration.
    std_threshold : float, optional
        The maximum standard deviation allowed for the Gaussian fit to be considered pure noise.

    Returns
    -------
    numpy.ndarray
        A 2D array of spike-free windows found in the signal.
    """
    import numpy as np
    spike_free_windows = []

    # Iterate over the signal with a sliding window
    for i in range(0, len(signal_data) - window_size + 1, step_size):
        # Get the current window from the signal
        signal_window = signal_data[i:i + window_size]

        # Check if the current window is spike-free
        spike_free_window = check_window_for_spikes(signal_window, std_threshold)

        # If the current window is spike-free, save it to the spike_free_windows array
        if not np.array_equal(spike_free_window, np.zeros_like(spike_free_window)):
            spike_free_windows.append(spike_free_window)
            #spike_free_windows = np.vstack((spike_free_windows, spike_free_window))
            #spike_free_windows[i:i + window_size, :] = spike_free_window

    # Return the spike-free windows as a 2D array
    return np.array(spike_free_windows)


def getting_timepoints_of_interest_from_threshold(electrode_data, timestamps, threshold, factor_user_pos=None, factor_user_neg=None):
    """
    Checks if a threshold value is raised by user defined factor.

    Parameters
    ----------
    electrode_data : numpy.ndarray
        The data to analyze.
    timestamps: numpy.ndarray
        Corresponding timestamps vector for data.
    threshold : float
        Value to be multiplied by user defined factor.
    factor_user_pos : float, optional
        User defined factor for positive threshold.
    factor_user_neg : float, optional
        User defined factor for negative threshold.

    Returns
    -------
    numpy.ndarray
        An array which contains timepoints, when threshold was raised.
    """
    import numpy as np
    timepoints_of_interest = []
    if factor_user_pos is not None and factor_user_neg is None:
        th_pos = threshold * factor_user_pos
        th_neg = None
    elif factor_user_pos is None and factor_user_neg is not None:
        th_pos = None
        th_neg = -threshold * factor_user_neg
    elif factor_user_pos is not None and factor_user_neg is not None:
        th_pos = threshold * factor_user_pos
        th_neg = -threshold * factor_user_neg
    else:
        raise ValueError('At least one threshold (pos or neg) has to been chosen')

    for i in range(len(electrode_data)):
        if (th_pos is not None and electrode_data[i] > th_pos) or (th_neg is not None and electrode_data[i] < th_neg):
            timepoints_of_interest.append(timestamps[i])
    return np.array(timepoints_of_interest)


def clean_timepoints_of_interest(timepoints_of_interest, refractory_period):
    """
    cleaning timepoints within refactory period.

    Parameters
    ----------
    timepoints_of_interest: np.array
        An array which contains timepoints, when threshold was raised.
    refractory_period: float
        Time in which multiple occurring spikes will be cleaned (deleted)

    Returns
    -------
    numpy.ndarray
        A cleaned version on an array which contains timepoints, when threshold was raised.
    """
    # function for cleaning timepoints within refactory period.
    import numpy as np
    if timepoints_of_interest.size > 0:
        timepoints_of_interest_cleaned = [timepoints_of_interest[0]]
        n = len(timepoints_of_interest)
        for i in range(1, n):
            if timepoints_of_interest[i] - timepoints_of_interest_cleaned[-1] > refractory_period:
                timepoints_of_interest_cleaned.append(timepoints_of_interest[i])
        timepoints_of_interest_cleaned = np.array(timepoints_of_interest_cleaned)

        return timepoints_of_interest_cleaned
    else:
        return timepoints_of_interest

def calculate_counts(arr):
    import numpy as np
    lengths = []  # list to store the length of each subarray
    for subarray in arr:
        lengths.append(len(subarray))  # calculate the length of each subarray
    total_length = sum(lengths)  # calculate the total length of all subarrays
    average_length = np.mean(lengths)  # calculate the average length of all subarrays
    return total_length, average_length


def application_of_threshold_algorithm(signal_raw, timestamps, method='std', factor_pos=None, factor_neg=3.5, refractory_period=0.001):
    # function which uses above functions with default params, be careful about sample frequency !
    import numpy as np
    spiketrains = []
    print(f'Threshold detection with method {method} started')
    print('factor_pos:', factor_pos, 'factor_neg:', factor_neg)
    for i in range(signal_raw.shape[1]):
        print('current electrode index:', i)
        electrode_index = i
        electrode_data = signal_raw[:, electrode_index]
        print('--- filtering')
        electrode_data_filtered = filter_data(electrode_data)
        print('--- calculation of threshold')

        if method == 'std':
            spike_free_windows = find_spike_free_windows_simplified(electrode_data_filtered)
            if spike_free_windows.size != 0: # check if spike_free_windows is not empty
                threshold = np.std(spike_free_windows)
            else:
                threshold = np.std(electrode_data_filtered)
        elif method == 'rms':
            spike_free_windows = find_spike_free_windows_simplified(electrode_data_filtered)
            if spike_free_windows.size != 0: # check if spike_free_windows is not empty
                threshold = np.sqrt(np.mean(np.square(spike_free_windows)))
            else:
                threshold = np.std(electrode_data_filtered)
        elif method == 'quiroga':
            threshold = np.median(np.absolute(electrode_data_filtered)) / 0.6745

        print('--- detection of spikes')
        timepoints_of_interest = getting_timepoints_of_interest_from_threshold(electrode_data_filtered, timestamps,
                                                                               threshold, factor_pos, factor_neg)
        print('--- cleaning refactory period')
        timepoints_of_interest_cleaned = clean_timepoints_of_interest(timepoints_of_interest, refractory_period=refractory_period)
        spiketrains.append(timepoints_of_interest_cleaned)
    print('Threshold detection finished')
    total_counts, average_count = calculate_counts(np.array(spiketrains, dtype=object))
    print("Spikes detected:", total_counts)
    print("Average spikes per electrode:", average_count)
    return np.array(spiketrains, dtype=object)

def preprocessing_for_one_recording_without_windowing(path):
    """
    preprocessing pipeline for one recording (without windowing)
    :param path: path to recording file
    :return: frame: A numpy array representing the merged data.
    """
    from tools import import_recording_h5, create_labels_of_all_spiketrains, assign_neuron_locations_to_electrode_locations, merge_data_to_location_assignments
    signal_raw, timestamps, ground_truth, electrode_locations, neuron_locations = import_recording_h5(path)
    labels_of_all_spiketrains = create_labels_of_all_spiketrains(ground_truth, timestamps)
    assignments = assign_neuron_locations_to_electrode_locations(electrode_locations, neuron_locations, 20)
    merged_data = merge_data_to_location_assignments(assignments, signal_raw.transpose(), labels_of_all_spiketrains, timestamps)
    return merged_data


def calculate_metrics_simple(data_gt, detection_output):
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Assuming 'detection_output' is the ndarray returned by the detection algorithm with shape (electrodes,)
    # and 'data_gt' is the ndarray with labels of shape (electrodes, 3, timestamps)

    # Initialize empty lists to store evaluation metrics for each sensor
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Iterate over each sensor
    for sensor in range(len(data_gt)):
        # Retrieve the recorded values, labels, and timestamps for the current sensor
        recorded_values = data_gt[sensor, 0, :]
        labels = data_gt[sensor, 1, :]
        timestamps = data_gt[sensor, 2, :]

        # Create binary arrays indicating the ground truth and detected spikes
        ground_truth_spikes = np.zeros_like(timestamps)
        ground_truth_spikes[labels == 1] = 1
        detected_spikes = np.zeros_like(timestamps)
        detected_spikes[np.isin(timestamps, detection_output[sensor])] = 1

        # Calculate evaluation metrics for the current sensor
        accuracy = accuracy_score(ground_truth_spikes, detected_spikes)
        precision = precision_score(ground_truth_spikes, detected_spikes)
        recall = recall_score(ground_truth_spikes, detected_spikes)
        f1 = f1_score(ground_truth_spikes, detected_spikes)

        # Append the scores to the respective lists
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Calculate the mean of evaluation metrics across all sensors
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)

    # Print the evaluation metrics
    print("Mean Accuracy:", mean_accuracy)
    print("Mean Precision:", mean_precision)
    print("Mean Recall:", mean_recall)
    print("Mean F1-score:", mean_f1)

def calculate_metrics_with_windows(data_gt, detection_output, window_size=10):
    import numpy as np

    # Assuming 'detection_output' is the ndarray returned by the detection algorithm with shape (electrodes,)
    # and 'data_gt' is the ndarray with labels of shape (electrodes, 3, timestamps)

    # Initialize empty lists to store evaluation metrics for each sensor
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    specificity_scores = []

    spikes_total = []

    # Iterate over each sensor
    for sensor in range(len(data_gt)):
        # Retrieve the recorded values, labels, and timestamps for the current sensor
        recorded_values = data_gt[sensor, 0, :]
        labels = data_gt[sensor, 1, :]
        timestamps = data_gt[sensor, 2, :]

        # Initialize empty lists to store evaluation metrics for each sensor
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        # Create binary arrays indicating the ground truth and detected spikes

        ground_truth_spikes = np.zeros_like(timestamps)
        ground_truth_spikes[labels == 1] = 1
        detected_spikes = np.zeros_like(timestamps)
        detected_spikes[np.isin(timestamps, detection_output[sensor])] = 1

        spikes_sum = np.sum(ground_truth_spikes)

        # Apply windowing to ground truth and detected spikes
        num_windows = len(timestamps) // window_size
        ground_truth_spikes_windowed = np.split(ground_truth_spikes[:num_windows * window_size], num_windows)
        detected_spikes_windowed = np.split(detected_spikes[:num_windows * window_size], num_windows)

        # Compare the windows to calculate TP, FP, TN, FN
        for i in range(num_windows):
            if np.max(ground_truth_spikes_windowed[i]) > 0:
                if np.max(detected_spikes_windowed[i]) > 0:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if np.max(detected_spikes_windowed[i]) > 0:
                    false_positives += 1
                else:
                    true_negatives += 1

        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if (true_positives + true_negatives + false_positives + false_negatives) > 0 else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = true_negatives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0


        # Append the scores to the respective lists
        spikes_total.append(spikes_sum)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        specificity_scores.append(specificity)

    # Calculate the mean of precision and recall scores across all sensors
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_specificity = np.mean(specificity_scores)

    # Print the evaluation metrics
    print("Ground Truth")
    print("Spikes:", np.sum(spikes_total))
    print("Average Spikes per electrode:", np.mean(spikes_total))
    print("Mean Accuracy:", mean_accuracy)
    print("Mean Precision:", mean_precision)
    print("Mean Recall:", mean_recall)
    print("Mean F1:", mean_f1)
    print("Mean Specificity:", mean_specificity)
    return accuracy_scores, precision_scores, recall_scores, f1_scores

def demo_arrays():
    demo_detect_output = np.array([[0.3, 0.7],
                               [0, 0.4, 0.7]],dtype=object)

    demo_ground_truth = np.array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],[0,0,0,1,0,0,1,0,0,0],[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
                    [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],[1,0,0,0,0,0,0,0,1,0],[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]])
    return demo_detect_output, demo_ground_truth

# pipeline
import numpy as np

path_to_data_file_h5 = ''

from tools import import_recording_h5
signal_raw, timestamps, ground_truth, channel_positions, template_locations = import_recording_h5(path_to_data_file_h5)

detection_output = application_of_threshold_algorithm(signal_raw, timestamps, factor_neg=4.0, factor_pos=4.0, refractory_period=0.002)

# evaluation
data_ground_truth = preprocessing_for_one_recording_without_windowing(path_to_data_file_h5)

accuracy_scores, precision_scores, recall_scores, f1_scores = calculate_metrics_with_windows(data_ground_truth, detection_output, window_size=40)

print('classical detection finished')

"""
data: rec_allen_60_17_12_10000_10000_10000_10000_10_20_32.h5

Spikes detected: 7651
Average spikes per electrode: 127.51666666666667
Ground Truth
Spikes: 3914.0
Average Spikes per electrode: 65.23333333333333
Mean Accuracy: 0.99054
Mean Precision: 0.07252085885420147
Mean Recall: 0.04684477192692837
Mean F1: 0.053707040880014774

"""