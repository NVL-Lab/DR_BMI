# some utils that may be shared among some plot functions
from typing import Optional, Tuple

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

from utils.analysis_constants import AnalysisConstants


def open_plot(sizes: tuple = (8, 6)):
    fig = plt.figure(figsize=sizes)
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def open_2subplots():
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(121)
    bx = fig.add_subplot(122)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    bx.spines["top"].set_visible(False)
    bx.spines["right"].set_visible(False)
    return fig, ax, bx

def open_4subplots_line():
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(141)
    bx = fig.add_subplot(142)
    cx = fig.add_subplot(143)
    dx = fig.add_subplot(144)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    bx.spines["top"].set_visible(False)
    bx.spines["right"].set_visible(False)
    cx.spines["top"].set_visible(False)
    cx.spines["right"].set_visible(False)
    dx.spines["top"].set_visible(False)
    dx.spines["right"].set_visible(False)
    return fig, ax, bx, cx, dx


def open_xsubplots(num_subplots: int = 4):
    fig = plt.figure(figsize=(12, 8))
    subplots = []
    for ind in np.arange(1, num_subplots + 1):
        ax = fig.add_subplot(np.ceil(np.sqrt(num_subplots)), np.sqrt(num_subplots), ind)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        subplots.append(ax)
    return fig, subplots


def save_plot(fig: plt.figure, ax: Optional, folder_path: Path, var_sig: str = '', var_type: str = '',
              set_labels: bool = True):
    if set_labels:
        ax.set_xlabel(var_type)
        ax.set_ylabel(var_sig)
    file_name = var_sig + "_" + var_type
    file_png = file_name + ".png"
    fig.savefig(folder_path / file_png, bbox_inches="tight")
    file_eps = file_name + ".eps"
    fig.savefig(folder_path / file_eps, format='eps', bbox_inches="tight")
    plt.close(fig)


def easy_plot(arr: np.array, xx: Optional[np.array] = None, folder_plots: Optional[Path] = None,
              var_sig: Optional[str] = None, vertical_array: Optional[np.array] = None):
    fig1, ax1 = open_plot()
    if xx is not None:
        ax1.plot(xx, arr)
    else:
        ax1.plot(arr)
    if vertical_array is not None:
        for vl in vertical_array:
            plt.vlines(x=vl, ymin=np.nanmin(arr), ymax=np.nanmean(arr), color='r')
    if folder_plots is not None:
        if var_sig is None: var_sig = 'kk'
        save_plot(fig1, ax1, folder_plots, var_sig)


def easy_imshow(arr: np.array, folder_plots: Optional[Path] = None, var_sig: Optional[str] = None):
    fig1, ax1 = open_plot()
    ax1.imshow(arr)
    if folder_plots is not None:
        if var_sig is None: var_sig = 'kk'
        save_plot(fig1, ax1, folder_plots, var_sig)


def get_pvalues(a:np.array, b:np.array, ax, pos: float = 0, height: float = 0.13, ind: bool = True):
    if ind:
        _, p_value = stats.ttest_ind(a[~np.isnan(a)], b[~np.isnan(b)])
    else:
        _, p_value = stats.ttest_rel(a, b)
    ax.text(pos, height, calc_pvalue(p_value))
    ax.text(pos + pos * 0.1, height - height / 10, "p = %0.2E" % p_value)


def get_1s_pvalues(a:np.array, b: float, ax, pos: float = 0, height: float = 0.13):
    _, p_value = stats.ttest_1samp(a[~np.isnan(a)], b)
    ax.text(pos, height, calc_pvalue(p_value))
    ax.text(pos + pos * 0.1, height - height / 3, "p = %0.2E" % p_value)


def get_anova_pvalues(a:np.array, b: np.array, axis: int, ax, pos: float = 0, height: float = 0.13):
    _, p_value = stats.f_oneway(a, b, axis=axis)
    ax.text(pos, height, calc_pvalue(p_value))
    ax.text(pos + pos * 0.1, height - height / 3, "p = %0.2E" % p_value)


def get_reg_pvalues(arr: np.array, x: np.array, ax, pos: float = 0, height: float = 0.13):
    _, _, _, p_value, _ = stats.linregress(x[~np.isnan(arr)], arr[~np.isnan(arr)])
    ax.text(pos, height, calc_pvalue(p_value))
    ax.text(pos + pos * 0.1, 0.9*height, "p = %0.2E" % p_value)


def calc_pvalue(p_value: float) -> str:
    """ returns a string with the pvalue ready to plot """
    if p_value <= 0.001:
        p = '***'
    elif p_value <= 0.01:
        p = '**'
    elif p_value <= 0.05:
        p = '*'
    elif np.isnan(p_value):
        p = 'nan'
    else:
        p = 'ns'
    return p


def generate_palette_all_figures(mice: np.array = AnalysisConstants.mice, palette: str = 'copper') -> dict:
    """ function to generate palette for all mice for all figures """
    custom_palette = sns.color_palette(palette, n_colors=len(mice))
    return {mouse: color for mouse, color in zip(mice, custom_palette)}


def array_regplot(df: pd.DataFrame, column: str) -> Tuple[np.array, np.array]:
    """ function to return x and y values for a regplot """
    expanded_values = df[column].apply(pd.Series)
    df_cleaned = expanded_values.dropna(axis=1, how='all')
    array_data = df_cleaned.to_numpy()
    return flatten_array(array_data)


def flatten_array(array_data: np.array) -> Tuple[np.array, np.array]:
    """ function to return x and y from a np.array """
    arr_1d = array_data.flatten()
    rows, cols = array_data.shape
    x = np.tile(np.arange(cols), rows)
    return x, arr_1d


def scale_array(arr: np.array, upper_val: int = 255, lower_val: int = 0) -> np.array:
    """ function to scale an array from lower_val to upper_val """
    min_value = arr.min()
    max_value = arr.max()

    scaled_matrix = (arr - min_value) / (max_value - min_value) * (upper_val - lower_val) + lower_val
    return scaled_matrix
