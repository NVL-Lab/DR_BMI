__author__ = 'Nuria'

import scipy
import collections
import pandas as pd
import numpy as np
import sklearn as sk
from pathlib import Path
from scipy import stats

from preprocess import sessions as ss
from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants
from analysis import learning


def obtain_gain(folder_experiments: Path) -> pd.DataFrame:
    """ function to obtain gain for all experiments """
    ret = collections.defaultdict(list)
    for experiment_type in AnalysisConstants.experiment_types:
        df_sessions = ss.get_sessions_df(folder_experiments, experiment_type)
        mice = df_sessions.mice_name.unique()
        for aa, mouse in enumerate(mice):
            df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
            for index, row in df_sessions_mouse.iterrows():
                folder_path = folder_experiments / row['session_path']
                ret['mice'].append(mouse)
                ret['session_path'].append(row['session_path'])
                ret['session_day'].append(row['session_day'])
                ret['previous_session'].append(row['previous_session'])
                ret['day_index'].append(row['day_index'])
                ret['experiment'].append(experiment_type)
                bmi_hits, bmi_gain = learning.gain_self_stim(folder_path / row['BMI_online'])
                ret['gain'].append(bmi_gain)
                ret['hits_per_min'].append(bmi_hits)
    return pd.DataFrame(ret)

