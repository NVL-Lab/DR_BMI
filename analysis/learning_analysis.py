
__author__ = 'Nuria'

import numpy as np
import scipy.io as sio

from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants


def gain_self_stim(file_path: str) -> [float, float]:
    """ Function to obtain the gain in self DR stim """
    bmi_online = sio.loadmat(file_path, simplify_cells=True)
    trial_start = bmi_online['data']['trialStart']
    if np.sum(trial_start) > 0:
        self_hits = bmi_online['data']['selfHits']
        end_bmi = bmi_online["data"]["frame"]
        init_bmi = np.where(trial_start == 1)[0][0]
        baseline_time = int(AnalysisConstants.framerate * AnalysisConfiguration.learning_baseline * 60) + init_bmi
        baseline_hits = self_hits[init_bmi:baseline_time].sum() / \
                        ((baseline_time - init_bmi)/AnalysisConstants.framerate / 60)
        BMI_time = len(self_hits[baseline_time:end_bmi])
        BMI_hits = self_hits[baseline_time:end_bmi].sum() / (BMI_time / AnalysisConstants.framerate / 60)
        if baseline_hits == 0:
            BMI_gain = np.nan
        else:
            BMI_gain = BMI_hits / baseline_hits
    else:
        BMI_hits = 0
        BMI_gain = 1
    return BMI_hits, BMI_gain

