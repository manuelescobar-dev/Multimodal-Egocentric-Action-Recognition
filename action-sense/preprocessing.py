import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import os

FS = 160  # Sampling frequency
CUTOFF = 5  # Cutoff frequency
NUM_SAMPLES = 100
CLIP_DURATION = 10 # Duration in seconds
NUM_CLIPS = 10 
DATA_PATH = "data/ActionNet-EMG"
MODE = "train"
FILENAME = f"data/ActionNet-EMG/ActionNet_{MODE}_emg.pkl"
NEW_FILENAME = f"data/ActionNet-EMG/ActionNet_{MODE}_emg_processed.pkl"
NUM_CHANNELS = 8

def rectify_signal(data):
    return np.abs(data).T

def filter_signal(data, fs, cutoff, num_channels):
    filtered_data = np.zeros_like(data)
    for i in range(num_channels):
        filtered_data[i] = low_pass_filter(data[i], fs, cutoff)
    return filtered_data.T

def low_pass_filter(data, fs, cutoff, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def normalization(data):
    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

def preprocessing(data, steps):
    for step in steps:
        data.apply(step)

def preprocess_row(row, start_time=0, duration=10, num_samples=100, side="left", num_channels=8):
    # Interpolate data for each row
    interpolation_time = np.linspace(start_time, start_time + duration, num_samples)  # Assuming the last timestamp represents the end time
    interpolator = interp1d(timestamps, normalized_data, kind='linear', fill_value='extrapolate', axis=0)
    interpolated_data = interpolator(interpolation_time)

    return interpolated_data


# Function to process each row of the DataFrame
def data_augmentation(row, num_samples=100, duration=10, num_clips = 1):
    # Preprocess the data
    sides = ["left", "right"]
    tot_time = row["myo_left_timestamps"][-1] - row["myo_left_timestamps"][0]
    return_rows = []
    highest_offset = max(0, tot_time-duration)
    offsets = np.linspace(0, highest_offset, num_clips)
    for st in offsets:
        final_data = {}
        for side in sides:
            final_data[side] = preprocess_row(row, st, duration, num_samples, side)
        final_data = np.hstack((final_data["left"], final_data["right"]))
        return_rows.append((row["description"], final_data))
    return return_rows

def load_data(path):
    with open(path, "rb") as f:
        data = pd.read_pickle(f)
    return data

def save_data(data, path):
    with open(path, "wb") as f:
        pd.to_pickle(data, f)

if __name__=="__main__":
    steps=[rectify_signal, low_pass_filter, normalization]
    data = load_data(os.path.join(DATA_PATH, FILENAME))
    processed_data = preprocessing(data, steps)
    save_data(processed_data, os.path.join(DATA_PATH, NEW_FILENAME))
