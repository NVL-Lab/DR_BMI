__author__ = 'Nuria'

import collections
from typing import Tuple

import pandas as pd
import numpy as np
from pathlib import Path

from preprocess import sessions as ss
from motion import motion_analysis as ma
from utils.analysis_constants import AnalysisConstants
from utils.analysis_command import AnalysisConfiguration


def obtain_motion_data(folder_list: list, speed_min=AnalysisConfiguration.run_speed_min) -> Tuple[pd.DataFrame, np.array]:
    """ function to compare motion characteristics between baseline and experiment """
    ret = collections.defaultdict(list)
    speed = np.empty(0)
    for experiment_type in AnalysisConstants.experiment_types:
        df_sessions = ss.get_sessions_df(folder_list, experiment_type)
        mice = df_sessions.mice_name.unique()
        for aa, mouse in enumerate(mice):
            df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
            folder_raw = Path(folder_list[ss.find_folder_path(mouse)]) / 'raw'
            for index, row in df_sessions_mouse.iterrows():
                file_path = Path(folder_raw) / row['session_path'] / 'motor'
                XY_baseline = ma.extract_XY_data(file_path / row['trigger_baseline'], file_path/ row['XY_baseline'])
                baseline_motor_features, aux_speed = ma.obtain_movement_parameters(XY_baseline, speed_min)
                speed = np.concatenate((speed, np.asarray(aux_speed)))
                XY_BMI = ma.extract_XY_data(file_path / row['trigger_BMI'], file_path / row['XY_BMI'])
                BMI_motor_features, aux_speed = ma.obtain_movement_parameters(XY_BMI, speed_min)
                speed = np.concatenate((speed, np.asarray(aux_speed)))
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
    return pd.DataFrame(ret), speed


def obtain_motion_behav_data(folder_list: list, speed_min=AnalysisConfiguration.run_speed_min) -> pd.DataFrame:
    """ function to compare motion characteristics between baseline and experiment """
    ret = collections.defaultdict(list)
    for experiment_type in AnalysisConstants.behav_type:
        df_sessions = ss.get_motor_data_behav(folder_list, experiment_type)
        mice = df_sessions.mice_name.unique()
        for aa, mouse in enumerate(mice):
            df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
            folder_raw = Path(folder_list[ss.find_folder_path(mouse)]) / 'raw'
            for index, row in df_sessions_mouse.iterrows():
                file_path = Path(folder_raw) / row['session_path'] / 'motor'
                XY = ma.extract_XY_data(file_path / row['trigger'], file_path / row['XY'])
                motor_features, _ = ma.obtain_movement_parameters(XY, speed_min)
                if motor_features is not None:
                    ret['mice'].append(mouse)
                    ret['session_date'].append(row['session_date'])
                    ret['experiment'].append(experiment_type)
                    for key in motor_features.index:
                        ret[key].append(motor_features[key])
    return pd.DataFrame(ret)


def create_df_motion():
    df_aux1 = pd.read_parquet("C:/Users/Nuria/Documents/DATA/D1exp/df_data/motion_behavior.parquet")
    df_aux2 = pd.read_parquet("C:/Users/Nuria/Documents/DATA/D1exp/df_data/motion_data.parquet")

    df_control = df_aux2[df_aux2.BB == "BMI"]
    df_control = df_control.drop(columns="BB")

    df_aux1["Laser"] = "ON"
    df_aux1 = df_aux1[df_aux1["experiment"] == "Behavior_before"]
    df_aux1 = df_aux1.drop(columns="experiment")

    df_aux2["Laser"] = "OFF"
    df_aux2.loc[df_aux2.BB == "BMI", "Laser"] = "BMI"
    df_aux2 = df_aux2.drop(columns="BB")
    df_aux2 = df_aux2[df_aux2.experiment.isin(['D1act', 'RANDOM', 'NO_AUDIO', 'DELAY'])]
    df_aux2 = df_aux2.drop(columns="experiment")
    df_motion = pd.concat((df_aux1, df_aux2))

    df_control.to_parquet("C:/Users/Nuria/Documents/DATA/D1exp/df_data/df_motion_controls.parquet")
    df_motion.to_parquet("C:/Users/Nuria/Documents/DATA/D1exp/df_data/df_motion.parquet")


def obtain_motion_trial(folder_list: list, seconds:int = 5) -> pd.DataFrame:
    """ Function to obtain the motion during the online experiment for all experiments """
    list_df = []
    df = ss.get_sessions_df(folder_list, 'D1act')
    for index, row in df.iterrows():
        folder_raw = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'raw'
        file_path = Path(folder_raw) / row['session_path'] / 'motor'
        XY_baseline = ma.extract_XY_data(file_path / row['trigger_baseline'], file_path / row['XY_baseline'])
        XY_BMI = ma.extract_XY_data(file_path / row['trigger_BMI'], file_path / row['XY_BMI'])
        if XY_BMI is not None and XY_baseline is not None:
            df = ma.obtain_motion_hits(XY_BMI, XY_baseline, folder_raw / row['session_path'] / row['BMI_online'], seconds = seconds)
            df['mice'] = row['mice_name']
            df['session_path'] = row['session_path']
            df['day_index'] = row['day_index']
            list_df.append(df)
    return pd.concat(list_df)


def obtain_r2_motion_cursor(folder_list: list, seconds:int = 5) -> pd.DataFrame:
    """ Function to obtain the motion during the online experiment for all experiments """
    ret = collections.defaultdict(list)
    df_sessions = ss.get_sessions_df(folder_list, 'D1act')
    for index, row in df_sessions.iterrows():
        folder_raw = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'raw'
        file_path = Path(folder_raw) / row['session_path'] / 'motor'
        XY_BMI = ma.extract_XY_data(file_path / row['trigger_BMI'], file_path / row['XY_BMI'])
        if XY_BMI is not None:
            ret['mice'].append(row['mice_name'])
            ret['session_path'].append(row['session_path'])
            r2 = ma.obtain_motion_cursor(XY_BMI, folder_raw / row['session_path'] / row['BMI_online'])
            ret['r2'].append(r2)
    return pd.DataFrame(ret)