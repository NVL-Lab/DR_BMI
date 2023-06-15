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
    run_speed_min: float = 10.  # minimum speed to consider boats of running
    speed_smooth_factor: int = 31  # seconds to smooth motion

    # learning
    learning_baseline: int = 7  # time in min to consider as "baseline" during BMI to obtain gain (5 + 2 to start BMI)

    # filtering background calcium signal
    filter_size: int = 500  # in frames

