from typing import Optional

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