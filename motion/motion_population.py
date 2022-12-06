__author__ = 'Nuria'

import scipy
import collections
import pandas as pd
import numpy as np
import sklearn as sk
from pathlib import Path

from preprocess import sessions as ss
from utils.analysis_command import AnalysisConfiguration
from motion import motion_analysis as ma


def obtain_motion_data(folder_experiments: Path) -> pd.DataFrame:
    """ function to compare motion characteristics between baseline and experiment """
    ret = collections.defaultdict(list)
    for experiment_type in AnalysisConfiguration.experiment_types:
        df_sessions = ss.get_sessions_df(folder_experiments, experiment_type)
        mice = df_sessions.mice_name.unique()
        for aa, mouse in enumerate(mice):
            df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
            for index, row in df_sessions_mouse.iterrows():
                file_path = Path(folder_experiments) / row['session_path'] / 'motor'
                XY_baseline = ma.extract_XY_data(file_path / row['trigger_baseline'], file_path/ row['XY_baseline'])
                baseline_motor_features = ma.obtain_movement_parameters(XY_baseline)
                XY_BMI = ma.extract_XY_data(file_path / row['trigger_BMI'], file_path / row['XY_BMI'])
                BMI_motor_features = ma.obtain_movement_parameters(XY_BMI)
                if baseline_motor_features is not None:
                    ret['mice'].append(mouse)
                    ret['session_date'].append(row['session_date'])
                    ret['experiment'].append(experiment_type)
                    ret['BB'].append('baseline')
                    for key in baseline_motor_features.index:
                        ret[key].append(baseline_motor_features[key])
                if BMI_motor_features is not None:
                    ret['mice'].append(mouse)
                    ret['session_date'].append(row['session_date'])
                    ret['experiment'].append(experiment_type)
                    ret['BB'].append('BMI')
                    for key in BMI_motor_features.index:
                        ret[key].append(BMI_motor_features[key])
    return pd.DataFrame(ret)


def obtain_motion_behav_data(folder_experiments: Path) -> pd.DataFrame:
    """ function to compare motion characteristics between baseline and experiment """
    ret = collections.defaultdict(list)
    for experiment_type in AnalysisConfiguration.behav_type:
        df_sessions = ss.get_behav_df(folder_experiments, experiment_type)
        mice = df_sessions.mice_name.unique()
        for aa, mouse in enumerate(mice):
            df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
            for index, row in df_sessions_mouse.iterrows():
                file_path = Path(folder_experiments) / row['session_path'] / 'motor'
                XY = ma.extract_XY_data(file_path / row['trigger'], file_path / row['XY'])
                motor_features = ma.obtain_movement_parameters(XY)
                if motor_features is not None:
                    ret['mice'].append(mouse)
                    ret['session_date'].append(row['session_date'])
                    ret['experiment'].append(experiment_type)
                    for key in motor_features.index:
                        ret[key].append(motor_features[key])
    return pd.DataFrame(ret)

