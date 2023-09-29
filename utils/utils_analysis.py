from typing import Optional, Tuple

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats.mstats import gmean, hmean
from scipy.signal import butter, filtfilt


def increase_percent(a: float, b: float) -> float:
    """given 2 values obtain the percent of increase"""
    return ((a - b) / abs(b)) * 100


def calculate_average(arr):
    return np.nanmean(arr)


def geometric_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """ calculate the gmean of a dataframe """
    df_out = pd.DataFrame()
    df_out[column] = pd.Series(gmean(df[column]))
    return df_out


def harmonic_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """ calculate the gmean of a dataframe """
    df_out = pd.DataFrame()
    df_out[column] = pd.Series(hmean(df[column]))
    return df_out


def remove_bad_mice(df: pd.DataFrame, bad_mice: list) -> pd.DataFrame:
    """ function to remove bad mice from dataframe"""
    df_control = df[df.experiment == "CONTROL"]
    df_no_control = df[~(df.experiment == "CONTROL")]
    df_good = df_no_control[~df_no_control.mice.isin(bad_mice)]
    df = pd.concat([df_good, df_control])
    return df


def add_nan_arrays(arr_list: list) -> list:
    """ Function to add NaN arrays to lists with less than 3 arrays """
    while len(arr_list) < 3:
        arr_list.append(np.array([np.nan] * len(arr_list[0])))
    return arr_list


def align_arrays(arr_list: list, min_length: int) -> list:
    """ Function to truncate or pad arrays to match the minimum length"""
    return [arr[:min_length] if len(arr) > min_length else np.pad(arr, (0, min_length - len(arr)), mode='constant',
                                                                  constant_values=np.nan) for arr in arr_list]


def unfold_arrays(row, column1: str, column2: str):
    """ function to unfold the columns given"""
    unfolded_rows = []
    for i in range(len(row['averages'])):
        unfolded_rows.append([row['mice'], row[column1][i], row[column2][i]])
    return unfolded_rows


def average_array_samples(arr: np.array, dimension_to_average: int, samples_to_average: int) -> np.array:
    """ function that averages over a dimension_to_average, X number of samples """
    # Calculate the number of resulting averages
    num_averages = arr.shape[dimension_to_average] // samples_to_average

    # Reshape the data to create views for averaging
    reshaped_data = arr.reshape((num_averages, samples_to_average) + (-1,))[..., :arr.shape[1]]

    # Calculate the averages along the specified dimension
    averages = np.mean(reshaped_data, axis=1)
    return averages.T


def sum_array_samples(arr: np.array, dimension_to_average: int, samples_to_average: int) -> np.array:
    """ function that averages over a dimension_to_average, X number of samples """
    # Calculate the number of resulting averages
    num_averages = arr.shape[dimension_to_average] // samples_to_average

    # Reshape the data to create views for averaging
    reshaped_data = arr.reshape((num_averages, samples_to_average) + (-1,))

    # Calculate the averages along the specified dimension
    sums = np.nansum(reshaped_data, axis=1)
    return sums.T


def remove_redundant(arr: np.array, min_dist: int = 40) -> np.array:
    """ function to remove iteratively redundant events """
    bad = np.arange(1, arr.shape[0])[np.diff(arr) < min_dist]
    finish = arr.shape[0]
    while len(bad) > 0 or finish == 0:
        arr = np.delete(arr, bad[0])
        bad = np.arange(1, arr.shape[0])[np.diff(arr) < min_dist]
        finish -= 1
    return arr


def find_closest(arr_orig: np.array, arr_syn: np.array) -> Tuple[np.array, np.array]:
    """ function to find the closes index of arr_orig for every element of arr_syn"""
    # Calculate the absolute differences between each element in arr_syn and all elements in arr_origin
    # Initialize empty lists to store closest indexes and differences
    closest_indexes = []
    differences = []

    # Iterate through arr_syn
    for value_syn in arr_syn:
        # Calculate the absolute differences between value_syn and all elements in arr_origin
        absolute_differences = np.abs(arr_orig - value_syn)

        # Find the index of the minimum absolute difference
        closest_index = np.argmin(absolute_differences)

        # Get the closest value in arr_origin
        closest_value = arr_orig[closest_index]

        # Calculate the difference, negative if value_syn is smaller, positive if greater
        difference = value_syn - closest_value

        # Append the closest index and difference to the respective lists
        closest_indexes.append(closest_index)
        differences.append(difference)

    # Convert the lists to NumPy arrays for consistency
    closest_indexes = np.array(closest_indexes)
    differences = np.array(differences)
    return closest_indexes, differences


def snr_neuron(folder_suite2p: Path) -> np.array:
    """ function to find snr of a cell """
    Fneu = np.load(Path(folder_suite2p) / "Fneu.npy")
    F_raw = np.load(Path(folder_suite2p) / "F.npy")
    power_signal_all = np.nanmean(np.square(F_raw), 1)
    power_noise_all = np.nanmean(np.square(Fneu), 1)

    # Calculate the SNR
    snr = 10 * np.log10(power_signal_all / power_noise_all)
    return snr


def stability_neuron(folder_suite2p: Path, init: int = 0, end: Optional[int] = None, low_pass_std: float = 1) -> np.array:
    """ function to obtain the stability of all the neurons in F_raw given by changes on mean and low_pass std"""
    F_raw = np.load(Path(folder_suite2p) / "F.npy")
    if end is None:
        end = F_raw.shape[1]
    try:
        bad_frames_dict = np.load(folder_suite2p / "bad_frames_dict.npy", allow_pickle=True).take(0)
        bad_frames_bool = bad_frames_dict['bad_frames_bool'][init:end]
    except FileNotFoundError:
        bad_frames_bool = np.zeros(F_raw.shape[1], dtype=bool)[init:end]
    F_to_analyze = F_raw[:, init:end]
    F_to_analyze = F_to_analyze[:, ~bad_frames_bool]
    arr_stab = np.zeros(F_to_analyze.shape[0], dtype=bool)
    for i in np.arange(F_to_analyze.shape[0]):
        arr_stab[i] = check_arr_stability(F_to_analyze[i, :]) and \
                      np.std(low_pass_arr(F_to_analyze[i, :])) < low_pass_std
    return arr_stab


def check_arr_stability(arr: np.array, num_samp: int = 10000, threshold: float = 0.2) -> bool:
    """ function to check the stability of a neuron"""
    if len(arr) < num_samp:
        return True  # Not enough data points to calculate stability.
    indices = np.arange(0, len(arr), num_samp/2, dtype=int)
    mean_arr = np.zeros(len(indices) - 2)
    for i, index in enumerate(indices[1:-1]):
        mean_arr[i] = np.mean(arr[index-int(num_samp/2):index+int(num_samp/2)])
    mean_max = np.max([mean_arr.mean() * (1 + threshold), mean_arr.mean()+2])
    mean_min = np.min([mean_arr.mean() * (1 - threshold), mean_arr.mean()-2])
    stability = np.sum(mean_arr > mean_max) + np.sum(mean_arr < mean_min)
    if stability > 0:
        return False
    else:
        return True


def low_pass_arr(arr: np.array, order: int = 5, cutoff_frequency: float = 0.01, fs: float = 30):
    """ function to check the std of the low_pass filtered signal"""
    b, a = butter(order, cutoff_frequency / (0.5 * fs), btype='low', analog=False)
    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, arr)
    return filtered_signal
