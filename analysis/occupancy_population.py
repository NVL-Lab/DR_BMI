import collections

import pandas as pd
import numpy as np
import scipy.io as sio
from pathlib import Path

from utils import utils_analysis as ut
from preprocess import sessions as ss
from utils.analysis_constants import AnalysisConstants


def obtain_occupancy_data(folder_list: list) -> pd.DataFrame:
    """ function to obtain the occupancy data from mat files """
    ret = collections.defaultdict(list)
    for experiment_type in AnalysisConstants.experiment_types:
        df_simulations = ss.get_simulations_df(folder_list, experiment_type)
        df_sessions = ss.get_sessions_df(folder_list, experiment_type)
        for index, row in df_simulations.iterrows():
            session_row = df_sessions[df_sessions['session_path'] == row['session_path']].iloc[0]
            folder_raw_experiment = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / \
                                    'raw' / row['session_path']
            folder_process_experiment = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / \
                                        'process' / row['session_path'] / 'simulation'
            mat_T1 = sio.loadmat(folder_raw_experiment / session_row['target_calibration'], simplify_cells=True)
            mat_T2 = sio.loadmat(folder_process_experiment / row['target_calibration'], simplify_cells=True)

            ret['mice'].append(row['mice_name'])
            ret['session_date'].append(row['session_date'])
            ret['session_path'].append(row['session_path'])
            ret['experiment'].append(experiment_type)
            ret['cal_T1_occurrence'].append(mat_T1['num_hits_no_b2base'])
            ret['cal_T1_hits'].append(mat_T1['num_valid_hits'])
            ret['cal_T2_occurrence'].append(mat_T2['num_hits_no_b2base'])
            ret['cal_T2_hits'].append(mat_T2['num_valid_hits'])

            mat_T1 = sio.loadmat(folder_process_experiment / row['Sim_T1'], simplify_cells=True)
            mat_T2 = sio.loadmat(folder_process_experiment / row['Sim_T2'], simplify_cells=True)

            T2_hits_no_b2base = mat_T2["data"]["cursor"] > mat_T2["bData"]["T1"]
            T2_valid_hits = mat_T2["data"]["selfHits"]

            ret['full_T1_occurrence'].append(np.sum(T1_hits_no_b2base))
            ret['full_T1_hits'].append(np.sum(T1_valid_hits))
            ret['full_T2_occurrence'].append(np.sum(T2_hits_no_b2base))
            ret['full_T2_hits'].append(np.sum(T2_valid_hits))
            ret['base_T1_occurrence'].append(np.sum(mat_T1["data"]["cursor"] > mat_T1["bData"]["T1"]))
            ret['base_T1_hits'].append(np.sum(mat_T1["data"]["selfHits"]))
            ret['base_T2_occurrence'].append(np.sum(mat_T2["data"]["cursor"] > mat_T2["bData"]["T1"]))
            ret['base_T2_hits'].append(np.sum(mat_T2["data"]["selfHits"]))
            ret['bmi_T1_occurrence'].append(np.sum(mat_T1["data"]["cursor"] > mat_T1["bData"]["T1"]))
            ret['bmi_T1_hits'].append(np.sum(mat_T1["data"]["selfHits"]))
            ret['bmi_T2_occurrence'].append(np.sum(mat_T2["data"]["cursor"] > mat_T2["bData"]["T1"]))
            ret['bmi_T2_hits'].append(np.sum(mat_T2["data"]["selfHits"]))


    return pd.DataFrame(ret)


def calculate_occupancy(df_occupancy: pd.DataFrame) -> pd.DataFrame:
    """ function that given a dataframe with information about the hits and occupancy obtains occupancy gain
    and other occupancy measures """
    df_occupancy['T1_hits_gain'] = ut.increase_percent(df_occupancy.bmi_T1_hits, df_occupancy.base_T1_hits)
    df_occupancy['T1_occupancy_gain'] = ut.increase_percent(df_occupancy.bmi_T1_occurrence,
                                                            df_occupancy.base_T1_occurrence)
    df_occupancy['T2_hits_gain'] = ut.increase_percent(df_occupancy.bmi_T2_hits, df_occupancy.base_T2_hits)
    df_occupancy['T2_occupancy_gain'] = ut.increase_percent(df_occupancy.bmi_T2_occurrence,
                                                            df_occupancy.base_T2_occurrence)

