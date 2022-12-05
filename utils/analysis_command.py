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
    experiment_types = ['BMI_STIM_AGO', 'BMI_RANDOM', 'BMI_STIM', 'BMI_AGO']
    behav_type = ['Initial_behavior', 'Behavior_before']

    # plotting
    to_plot: bool = False
    plot_path = Path("C:/Users/nuria/DATA/plots")

    # movement
    run_speed_min: int = 1e6  # minimum speed to consider boats of running
    walk_speed_min: int = 1e4  # minimum speed to consider boats of walking
    speed_smooth_factor: int = 31  # seconds to smooth motion

    # learning
    learning_baseline: int = 5  # time in min to consider as "baseline" during BMI to obtain gain

