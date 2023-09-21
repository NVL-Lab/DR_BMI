__author__ = 'Nuria'


import collections
from typing import Tuple

import pandas as pd
import numpy as np
from pathlib import Path

from preprocess import sessions as ss
from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants
from utils.utils_analysis import harmonic_mean, geometric_mean
from analysis import learning_analysis


def obtain_gain(folder_list: list, time_or_hit: str = 'time') -> pd.DataFrame:
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
                bmi_hits, bmi_gain, _, hit_array, time_to_hit = \
                    learning_analysis.gain_self_stim(folder_path / row['BMI_online'], time_or_hit)
                ret['gain'].append(bmi_gain)
                ret['hits_per_min'].append(bmi_hits)
                ret['hit_array'].append(hit_array)
                ret['time_to_hit'].append(time_to_hit)
    return pd.DataFrame(ret)


def obtain_gain_posthoc(folder_list: list, time_or_hit: str = 'time') -> pd.DataFrame:
    """ function to obtain gain for all experiments """
    ret = collections.defaultdict(list)
    for experiment_type in AnalysisConstants.experiment_types:
        df_sessions = ss.get_simulations_posthoc_df(folder_list, experiment_type)
        mice = df_sessions.mice_name.unique()
        for aa, mouse in enumerate(mice):
            df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
            folder_process = Path(folder_list[ss.find_folder_path(mouse)]) / 'process'
            for index, row in df_sessions_mouse.iterrows():
                folder_path = folder_process / row['session_path'] / 'simulation_posthoc'
                ret['mice'].append(mouse)
                ret['session_path'].append(row['session_path'])
                ret['experiment'].append(experiment_type)
                ret['T'].append(row['T'])
                if len(row['Simulation'])>0:
                    bmi_hits, bmi_gain, _, _, _ = \
                        learning_analysis.gain_self_stim(folder_path / row['Simulation'], time_or_hit)
                    ret['gain'].append(bmi_gain)
                    ret['hits_per_min'].append(bmi_hits)
                else:
                    ret['gain'].append(np.nan)
                    ret['hits_per_min'].append(np.nan)
    return pd.DataFrame(ret)


def obtain_extinction(folder_list: list) -> pd.DataFrame:
    """ function to obtain gain for all experiments """
    ret = collections.defaultdict(list)
    df_sessions = ss.get_extinction(folder_list)
    mice = df_sessions.mice.unique()
    for aa, mouse in enumerate(mice):
        df_sessions_mouse = df_sessions[df_sessions.mice == mouse]
        folder_raw = Path(folder_list[ss.find_folder_path(mouse)]) / 'raw'
        for index, row in df_sessions_mouse.iterrows():
            hpm = []
            tth = []
            folder_path = folder_raw / row['session_path']
            ret['mice'].append(mouse)
            ret['session_path'].append(row['session_path'])
            bmi_hits, bmi_gain, base_hits, hpm_aux, tth_aux = learning_analysis.gain_self_stim(folder_path / row['BMI_online'])
            ret['BMI_gain'].append(bmi_gain)
            ret['BMI_hpm'].append(bmi_hits)
            hpm.append(hpm_aux)
            tth.append(tth_aux)
            ext_hits, _, _, hpm_aux, tth_aux = learning_analysis.gain_self_stim(folder_path / row['extinction'])
            ret['ext_gain'].append(ext_hits/base_hits)
            ret['ext_hpm'].append(ext_hits)
            hpm.append(hpm_aux)
            tth.append(tth_aux)
            if row['extinction_2'] == 'None':
                ret['ext2_gain'].append('None')
                ret['ext2_hpm'].append('None')
            else:
                ext2_hits, _, _, hpm_aux, tth_aux = learning_analysis.gain_self_stim(folder_path / row['extinction_2'])
                ret['ext2_gain'].append(ext2_hits/base_hits)
                ret['ext2_hpm'].append(ext2_hits)
                hpm.append(hpm_aux)
                tth.append(tth_aux)
            ret['hits_per_min'].append(hpm)
            ret['time_to_hit'].append(tth)
    return pd.DataFrame(ret)


def get_bad_mice(df: pd.DataFrame) -> Tuple[np.array, float, float]:
    """ function to obtain the bad mice given the df learning """
    df = df.dropna()
    df_control = df[df.experiment == "CONTROL"]

    # remove the bad animals
    average_control = geometric_mean(df_control, 'gain').values[0][0]
    df_group_control = df_control.groupby(["mice", "experiment"]).apply(geometric_mean, 'gain')
    df_group = df.groupby(["mice", "experiment"]).apply(geometric_mean, 'gain').sort_values('experiment').reset_index()
    df_group_d1act = df_group[df_group.experiment == 'D1act']
    deviations = df_group_control.gain - average_control
    squared_deviations = deviations ** 2
    mean_squared_deviations = np.mean(squared_deviations)
    std_harmonic = np.sqrt(mean_squared_deviations)
    bad_mice = df_group_d1act[df_group_d1act.gain < (std_harmonic + average_control)].mice.unique()
    return bad_mice, average_control, std_harmonic


def calculate_time_to_hit_per_min(row) -> np.array:
    """ function to calculate the average time_to_hit for each hit_array element"""
    hit_array = row['hit_array'].astype(int)
    time_to_hit = row['time_to_hit']

    averages = []

    for hit in hit_array:
        average = np.mean(time_to_hit[:hit])
        averages.append(average)
        time_to_hit = time_to_hit[hit:]

    return np.asarray(averages)
