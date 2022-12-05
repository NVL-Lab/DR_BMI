
__author__ = 'Nuria'

import numpy as np
import pandas as pd
import scipy.io as sio

from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants


def gain_self_stim(file_path: Path) -> [float, float]:
    """ Function to obtain the gain in self DR stim """
    bmi_online = sio.loadmat(file_path, simplify_cells=True)
    trial_start = bmi_online['data']['trialStart']
    self_DR_stim = bmi_online['data']['selfDRstim']
    init_bmi = np.where(trial_start == 1)[0][0]
    baseline_time = int(AnalysisConstants.framerate * AnalysisConfiguration.learning_baseline * 60)
    baseline_hits = self_DR_stim[init_bmi:init_bmi+baseline_time].sum() / AnalysisConfiguration.learning_baseline
    BMI_time = len(self_DR_stim[init_bmi+baseline_time:])/ AnalysisConstants.framerate / 60
    BMI_hits = self_DR_stim[init_bmi+baseline_time:].sum() / BMI_time
    BMI_gain = BMI_hits / baseline_hits
    return BMI_hits, BMI_gain
