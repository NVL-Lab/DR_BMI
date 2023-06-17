import collections

import pandas as pd
import numpy as np
import scipy.io as sio
from pathlib import Path

from utils import utils_analysis as ut
from preprocess import sessions as ss
from utils.analysis_constants import AnalysisConstants
from analysis.occupancy_analysis import obtain_occupancy


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

            ret['mice'].append(row['mice_name'])
            ret['session_date'].append(row['session_date'])
            ret['session_path'].append(row['session_path'])
            ret['experiment'].append(experiment_type)

            mat_T1 = sio.loadmat(folder_raw_experiment / session_row['target_calibration'], simplify_cells=True)
            mat_T2 = sio.loadmat(folder_process_experiment / row['target_calibration'], simplify_cells=True)
            occupancy_T1 = obtain_occupancy(sio.loadmat(folder_process_experiment / row['Sim_T1'], simplify_cells=True))
            occupancy_T2 = obtain_occupancy(sio.loadmat(folder_process_experiment / row['Sim_T2'], simplify_cells=True))

            ret['cal_T1_occupancy'].append(mat_T1['num_hits_no_b2base']/AnalysisConstants.len_calibration)
            ret['cal_T1_hits'].append(mat_T1['num_valid_hits']/AnalysisConstants.len_calibration)
            ret['cal_T2_occupancy'].append(mat_T2['num_hits_no_b2base']/AnalysisConstants.len_calibration)
            ret['cal_T2_hits'].append(mat_T2['num_valid_hits']/AnalysisConstants.len_calibration)

            ret['full_T1_occupancy'].append(occupancy_T1['full_occupancy'])
            ret['full_T1_hits'].append(occupancy_T1['full_hits'])
            ret['full_T2_occupancy'].append(occupancy_T2['full_occupancy'])
            ret['full_T2_hits'].append(occupancy_T2['full_hits'])

            ret['base_T1_occupancy'].append(occupancy_T1['base_occupancy'])
            ret['base_T1_hits'].append(occupancy_T1['base_hits'])
            ret['base_T2_occupancy'].append(occupancy_T2['base_occupancy'])
            ret['base_T2_hits'].append(occupancy_T2['base_hits'])

            ret['bmi_T1_occupancy'].append(occupancy_T1['bmi_occupancy'])
            ret['bmi_T1_hits'].append(occupancy_T1['bmi_hits'])
            ret['bmi_T2_occupancy'].append(occupancy_T2['bmi_occupancy'])
            ret['bmi_T2_hits'].append(occupancy_T2['bmi_hits'])

    return pd.DataFrame(ret)


def calculate_occupancy(df_occupancy: pd.DataFrame) -> pd.DataFrame:
    """ function that given a dataframe with information about the hits and occupancy obtains occupancy gain
    and other occupancy measures """
    df_occupancy['T1_hits_cal_gain'] = ut.increase_percent(df_occupancy.full_T1_hits,
                                                           df_occupancy.cal_T1_hits)
    df_occupancy['T1_occupancy_cal_gain'] = ut.increase_percent(df_occupancy.full_T1_occupancy,
                                                                df_occupancy.cal_T1_occupancy)
    df_occupancy['T1_hits_gain'] = ut.increase_percent(df_occupancy.bmi_T1_hits,
                                                       df_occupancy.base_T1_hits)
    df_occupancy['T1_occupancy_gain'] = ut.increase_percent(df_occupancy.bmi_T1_occupancy,
                                                            df_occupancy.base_T1_occupancy)
    df_occupancy['T2_hits_cal_gain'] = ut.increase_percent(df_occupancy.full_T2_hits,
                                                           df_occupancy.cal_T2_hits)
    df_occupancy['T2_occupancy_cal_gain'] = ut.increase_percent(df_occupancy.full_T2_occupancy,
                                                                df_occupancy.cal_T2_occupancy)
    df_occupancy['T2_hits_gain'] = ut.increase_percent(df_occupancy.bmi_T2_hits,
                                                       df_occupancy.base_T2_hits)
    df_occupancy['T2_occupancy_gain'] = ut.increase_percent(df_occupancy.bmi_T2_occupancy,
                                                            df_occupancy.base_T2_occupancy)
    df_occupancy['T1T2_cal_gain'] = df_occupancy['T1_hits_cal_gain'] - df_occupancy['T2_hits_cal_gain']
    df_occupancy['T1T2_cal_occupancy'] = df_occupancy['T1_occupancy_cal_gain'] - df_occupancy['T2_occupancy_cal_gain']
    df_occupancy['T1T2_gain'] = df_occupancy['T1_hits_gain'] - df_occupancy['T2_hits_gain']
    df_occupancy['T1T2_occupancy'] = df_occupancy['T1_occupancy_gain'] - df_occupancy['T2_occupancy_gain']
