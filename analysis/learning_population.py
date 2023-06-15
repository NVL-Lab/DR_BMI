__author__ = 'Nuria'


import collections
import pandas as pd

from pathlib import Path

from preprocess import sessions as ss
from utils.analysis_constants import AnalysisConstants
from analysis import learning_analysis


def obtain_gain(folder_list: list) -> pd.DataFrame:
    """ function to obtain gain for all experiments """
    ret = collections.defaultdict(list)
    for experiment_type in AnalysisConstants.experiment_types:
        df_sessions = ss.get_sessions_df(folder_list, experiment_type)
        mice = df_sessions.mice_name.unique()
        for aa, mouse in enumerate(mice):
            df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
            folder_raw = Path(folder_list[ss.find_folder_path(mouse)]) / 'raw'
            for index, row in df_sessions_mouse.iterrows():
                folder_path = folder_raw / row['session_path']
                ret['mice'].append(mouse)
                ret['session_path'].append(row['session_path'])
                ret['session_day'].append(row['session_day'])
                ret['previous_session'].append(row['previous_session'])
                ret['day_index'].append(row['day_index'])
                ret['experiment'].append(experiment_type)
                bmi_hits, bmi_gain = learning_analysis.gain_self_stim(folder_path / row['BMI_online'])
                ret['gain'].append(bmi_gain)
                ret['hits_per_min'].append(bmi_hits)
    return pd.DataFrame(ret)


def obtain_extinction(folder_list: list) -> pd.DataFrame:
    """ function to obtain gain for all experiments """
    ret = collections.defaultdict(list)
    df_sessions = ss.get_extinction(folder_list)
    mice = df_sessions.mice_name.unique()
    for aa, mouse in enumerate(mice):
        df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
        folder_raw = Path(folder_list[ss.find_folder_path(mouse)]) / 'raw'
        for index, row in df_sessions_mouse.iterrows():
            folder_path = folder_raw / row['session_path']
            ret['mice'].append(mouse)
            ret['session_path'].append(row['session_path'])
            bmi_hits, bmi_gain = learning_analysis.gain_self_stim(folder_path / row['BMI_online'])
            ret['BMI_gain'].append(bmi_gain)
            ret['BMI_hpm'].append(bmi_hits)
            ext_hits, ext_gain = learning_analysis.gain_self_stim(folder_path / row['extinction'])
            ret['ext_gain'].append(ext_gain)
            ret['ext_hpm'].append(ext_hits)
            if row['extinction_2'] == 'None':
                ret['ext2_gain'].append('None')
                ret['ext2_hpm'].append('None')
            else:
                ext2_hits, ext2_gain = learning_analysis.gain_self_stim(folder_path / row['extinction_2'])
                ret['ext2_gain'].append(ext2_gain)
                ret['ext2_hpm'].append(ext2_hits)
    return pd.DataFrame(ret)