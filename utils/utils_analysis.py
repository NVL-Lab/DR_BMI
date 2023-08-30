from typing import Optional, Tuple

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats.mstats import gmean, hmean


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


def unfold_arrays(row, column1:str, column2:str):
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
    reshaped_data = arr.reshape((num_averages, samples_to_average) + (-1,))[..., :arr.shape[1]]

    # Calculate the averages along the specified dimension
    sums = np.nansum(reshaped_data, axis=1)
    return sums.T
