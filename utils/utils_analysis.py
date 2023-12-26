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


def sum_array_samples(arr: np.array, dimension_to_sum: int, samples_to_sum: int) -> np.array:
    """
     function that sums over a dimension_to_sum, X number of samples
    :param arr: array to downsample
    :param dimension_to_sum:
    :param samples_to_sum: amount of samples to sum
    :return:
    """
    # Calculate the number of resulting averages
    num_averages = arr.shape[dimension_to_sum] // samples_to_sum

    # Reshape the data to create views for averaging
    reshaped_data = arr.reshape((num_averages, samples_to_sum) + (-1,))

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
    """
    function to find the closes index of arr_orig for every element of arr_syn
    :param arr_orig: original array
    :param arr_syn: synchronization array
    :return: closest index array and differences with the synchronization array
    """
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
    """
    function to find snr of a cell
    :param folder_suite2p: folder where the files are stored
    :return: array with the snr of each neuron
    """
    Fneu = np.load(Path(folder_suite2p) / "Fneu.npy")
    F_raw = np.load(Path(folder_suite2p) / "F.npy")
    power_signal_all = np.nanmean(np.square(F_raw), 1)
    power_noise_all = np.nanmean(np.square(Fneu), 1)

    # Calculate the SNR
    snr = 10 * np.log10(power_signal_all / power_noise_all)
    return snr


def stability_neuron(folder_suite2p: Path, init: int = 0, end: Optional[int] = None, low_pass_std: float = 1) -> np.array:
    """
    function to obtain the stability of all the neurons in F_raw given by changes on mean and low_pass std
    :param folder_suite2p: folder where the files are stored
    :param init: initial frame to consider
    :param end: last frame to consider
    :param low_pass_std: the threshold to consider for the low pass check
    :return: array of bools to show stability of each neuron
    """
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
    """
    function to check the stability of a neuron
    :param arr: array normally the raw signal of a neurn
    :param num_samp: number of frames/samples to test stability
    :param threshold: threshold to consider stable
    :return: bool arr is stable or not
    """
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
    """
    function to check the std of the low_pass filtered signal
    :param arr: array to filter
    :param order: order of the filter
    :param cutoff_frequency: cutoff frequency
    :param fs: sample frequency
    :return: filtered signal
    """
    b, a = butter(order, cutoff_frequency / (0.5 * fs), btype='low', analog=False)
    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, arr)
    return filtered_signal


def remove_matching_index(arr: np.array, indices: np.array, num_index: int) -> np.array:
    """
    function to remove the indexes in indices and any other element a number num_index before
    :param arr: np.array where to remove the indices
    :param indices: indices to be removed
    :param num_index: all other elements before the indices to be removed
    :return: array with removed indices
    """
    matching_elements = np.intersect1d(arr, indices)
    for element in matching_elements:
        index = np.where(arr == element)[0][0]
        # Calculate the starting index for removal (300 elements before the matching element)
        start_index = max(0, index - num_index)
        # Remove the elements
        arr = np.delete(arr, np.arange(start_index, index + 1))
    return arr


def replace_cc_val_with_nan(df, column_name, num_values: int = 10):
    """
    Replace array values with NaN if there are less than 10 non-NaN values in the array.

    :param df: Pandas DataFrame containing the column with arrays.
    :param column_name: Name of the column containing the arrays.
    :return: Modified DataFrame with updated arrays.
    """

    def replace_array(arr):
        # Count non-NaN values and replace array if count is less than 10
        non_nan_count = np.count_nonzero(~np.isnan(arr))
        return np.full_like(arr, np.nan) if non_nan_count < num_values else arr

    df[column_name] = df[column_name].apply(replace_array)
    return df