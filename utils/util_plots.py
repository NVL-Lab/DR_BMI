# some utils that may be shared among some plot functions
from typing import Optional

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def open_plot(sizes: tuple = [8, 6]):
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


def open_xsubplots(num_subplots: int = 4):
    fig = plt.figure(figsize=(12, 8))
    subplots = []
    for ind in np.arange(1, num_subplots+1):
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


def get_pvalues(a, b, ax, pos: float = 0, height: float = 0.13, ind: bool = True):
    if ind:
        _, p_value = stats.ttest_ind(a, b)
    else:
        _, p_value = stats.ttest_rel(a, b)
    p = str(p_value)
    ax.text(pos, height, p)
    ax.text(pos + pos/3, height - height/3, "p = %0.2E" % p_value)