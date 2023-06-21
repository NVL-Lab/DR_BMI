
__author__ = 'Nuria'

import numpy as np
import scipy.io as sio
from scipy.stats import binned_statistic

from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants


def gain_self_stim(file_path: str, time_or_hit: str = 'time') -> [float, float, float, np.array, np.array]:
    """ Function to obtain the gain in self DR stim """
    bmi_online = sio.loadmat(file_path, simplify_cells=True)
    trial_start = bmi_online['data']['trialStart']
    if np.sum(trial_start) > 0:
        self_hits = bmi_online['data']['selfHits']
        end_bmi = bmi_online["data"]["frame"]
        init_bmi = np.where(trial_start == 1)[0][0]
        if time_or_hit == 'time':
            baseline_time = int(AnalysisConstants.framerate * AnalysisConfiguration.learning_baseline * 60) + init_bmi
        elif time_or_hit == 'hit':
            if np.nansum(self_hits) > AnalysisConfiguration.learning_baseline_hits:
                baseline_time = np.where(self_hits)[0][AnalysisConfiguration.learning_baseline_hits - 1]
            else:
                baseline_time = end_bmi
        else:
            raise ValueError('time_or_hit can only be as the name explains the str: time or hit')
        baseline_hits = self_hits[init_bmi:baseline_time].sum() / \
                        ((baseline_time - init_bmi)/AnalysisConstants.framerate / 60)
        BMI_time = len(self_hits[baseline_time:end_bmi])
        BMI_minutes = BMI_time / AnalysisConstants.framerate / 60
        exp_frames = self_hits[init_bmi:end_bmi]
        bin_edges = np.linspace(0, len(exp_frames), int(len(exp_frames) / AnalysisConstants.framerate / 60) + 1)
        hit_array, _, _ = binned_statistic(np.arange(len(exp_frames)), exp_frames, bins=bin_edges, statistic='sum')
        time_to_hit = np.diff(np.where(trial_start)[0])
        if baseline_hits == 0 or BMI_time == 0:
            BMI_gain = np.nan
            BMI_hits = np.nan
        else:
            BMI_hits = self_hits[baseline_time:end_bmi].sum() / BMI_minutes
            BMI_gain = BMI_hits / baseline_hits
    else:
        baseline_hits = np.nan
        BMI_hits = np.nan
        BMI_gain = np.nan
        hit_array = np.nan
        time_to_hit = np.nan
    return BMI_hits, BMI_gain, baseline_hits, hit_array, time_to_hit/AnalysisConstants.framerate
