import numpy as np
import pandas as pd

from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants


def obtain_occupancy(mat: dict) -> dict:
    """ obtain the baseline and experiment occupancy given a dict containing the values of the experiment mat file """
    occupancy_dict = dict()
    hits_no_b2base = mat["data"]["cursor"] > mat["bData"]["T1"]
    self_hits = mat["data"]["selfHits"]
    trial_start = mat['data']['trialStart']
    init_bmi = np.where(trial_start == 1)[0][0]
    baseline_time = int(AnalysisConstants.framerate * AnalysisConfiguration.learning_baseline * 60)
    BMI_time = len(self_hits[baseline_time:])

    occupancy_dict['full_occupancy'] = np.sum(hits_no_b2base)
    occupancy_dict['full_hits'] = np.sum(self_hits)
    occupancy_dict['base_occupancy'] = hits_no_b2base[init_bmi:baseline_time].sum() / \
                                       ((baseline_time - init_bmi) / AnalysisConstants.framerate / 60)
    occupancy_dict['base_hits'] = self_hits[init_bmi:baseline_time].sum() / \
                                  ((baseline_time - init_bmi) / AnalysisConstants.framerate / 60)
    occupancy_dict['bmi_occupancy'] = hits_no_b2base[baseline_time:].sum() / \
                                      (BMI_time / AnalysisConstants.framerate / 60)
    occupancy_dict['bmi_hits'] = self_hits[baseline_time:].sum() / \
                                 (BMI_time / AnalysisConstants.framerate / 60)
