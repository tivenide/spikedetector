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
    popt, pcov = curve_fit(gaussian_func, bin_centers, hist)
    mu, sigma = popt[1], popt[2]

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


def clean_timepoints_of_interest(timepoints_of_interest, cleaning_size):
    # function for cleaning timepoints in cleaning_size of refactory period.
    pass


def application_of_threshold_algorithm(signal_raw, timestamps, method='std', factor_pos=None, factor_neg=3.5):
    # function which uses above functions with default params, be careful about sample frequency !
    # @TODO: deal with problem, if find_spike_free_windows_simplified() find no spike_free_window
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
        spike_free_windows = find_spike_free_windows_simplified(electrode_data_filtered)

        if method == 'std':
            threshold = np.std(spike_free_windows)
        elif method == 'rms':
            threshold = np.sqrt(np.mean(np.square(spike_free_windows)))
        elif method == 'quiroga':
            threshold = np.median(electrode_data_filtered) / 0.6745

        print('--- detection of spikes')
        timepoints_of_interest = getting_timepoints_of_interest_from_threshold(electrode_data_filtered, timestamps,
                                                                               threshold, factor_pos, factor_neg)
        spiketrains.append(timepoints_of_interest)
    print('Threshold detection finished')
    return np.array(spiketrains)


# pipeline

path_to_data_file_h5 = ''

from tools import import_recording_h5
signal_raw, timestamps, ground_truth, channel_positions, template_locations = import_recording_h5(path_to_data_file_h5)

spiketrains = application_of_threshold_algorithm(signal_raw, timestamps)



print('classical detection finished')