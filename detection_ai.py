# detection ai based

def windowing_electrode_data(electrode_data, timestamps, window_size=20):
    electrode_data_windowed = []
    timestamps_windowed = []
    for i in range(0, len(electrode_data) - window_size + 1, window_size):
        electrode_data_windowed.append(electrode_data[i:i + window_size])
        timestamps_windowed.append(timestamps[i:i + window_size])
    return electrode_data_windowed, timestamps_windowed

import torch

def function1(window):
    # Assuming `model` is your trained PyTorch model
    model = torch.load('TransformerModel.pth')
    model.eval()

    # Preprocess the window data (if needed) and convert it to a PyTorch tensor
    window_tensor = torch.tensor(window, dtype=torch.float32)

    # Perform forward pass through the model to obtain predictions
    with torch.no_grad():
        output = model(window_tensor.unsqueeze(0))  # Add batch dimension

    # Assuming your model performs binary classification with a single output node
    predicted_class = torch.round(output).item()

    # Return True if the predicted class is 1 (spike), False otherwise
    return predicted_class == 1


def detect_spikes_and_get_timepoints(windowed_data, timestamps):
    spikes = []
    for window_index in range(len(windowed_data)):
        window = windowed_data[window_index]
        window_timestamps = timestamps[window_index]

        # Assuming the resulting timepoint of interest is the 5th timestamp
        timepoint_of_interest = window_timestamps[4] if len(window_timestamps) >= 5 else None

        # Use your spike detection logic in function1()
        is_spike = function1(window)

        if is_spike:
            spikes.append(timepoint_of_interest)

    return spikes

def application_of_model(signal_raw, timestamps):
    import numpy as np
    spiketrains = []
    for i in range(signal_raw.shape[1]):
        print('current electrode index:', i)
        electrode_index = i
        electrode_data = signal_raw[:, electrode_index]
        electrode_data_windowed, timestamps_windowed = windowing_electrode_data(electrode_data, timestamps)
        spiketrains.append(detect_spikes_and_get_timepoints(electrode_data_windowed, timestamps_windowed))
    return np.array(spiketrains, dtype=object)

path_to_data_file_h5 = ''

from tools import import_recording_h5
signal_raw, timestamps, ground_truth, channel_positions, template_locations = import_recording_h5(path_to_data_file_h5)

#from detection_classical import preprocessing_for_one_recording_without_windowing, calculate_metrics_with_windows, plot_roc

detection_output = application_of_model(signal_raw, timestamps)

# evaluation
#data_ground_truth = preprocessing_for_one_recording_without_windowing(path_to_data_file_h5)

#accuracy_scores, precision_scores, recall_scores, f1_scores, _ = calculate_metrics_with_windows(data_ground_truth, detection_output, window_size=40)
#plot_roc(data_ground_truth, detection_output, window_size=40)
