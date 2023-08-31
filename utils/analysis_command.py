# dataclass for analysis

import numpy as np
from pathlib import Path
from dataclasses import dataclass

from utils.analysis_constants import AnalysisConstants


@dataclass
class AnalysisConfiguration:
    """
    Class containing various configuration parameters for analysis. Reasonable defaults are
    provided.
    """

    # dirs
    local_dir = Path("C:/Users/nuria/DATA/Analysis/")  # None
    experiment_dir = Path("F:/data")

    # general
    tol = 0.05  # tolerance at which the pvalues will be consider significant

    # plotting
    to_plot: bool = False
    plot_path = Path("C:/Users/nuria/DATA/plots")
    hist_bins_default = 100  # number of bins by default to use on histograms

    # movement
    run_speed_min: float = 5.  # minimum speed to consider boats of running
    speed_smooth_factor: int = 31  # seconds to smooth motion

    # learning
    learning_baseline: int = 5  # time in min to consider as "baseline" during BMI to obtain gain
    learning_baseline_hits: int = 5  # number of hits to consider "baseline time"

    # filtering background calcium signal
    filter_size: int = 500  # in frames

    # time_locked
    time_lock_seconds: int = 3  # number of seconds before and after target to get info

    # manifold
    FA_spks_av_win: int = 30  # number of samples to count spikes. 30 -> 1sec
    FA_n_neu: int = 50  # number of neurons for the FA
    FA_n_iter: int = 100  # number of iterations to run the FA for different subset of neurons
    FA_VAF: float = 0.7  # minimum VAF for FA
    FA_components: int = 10  # number of components for FA

