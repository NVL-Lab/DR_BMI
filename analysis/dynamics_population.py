__author__ = 'Nuria'

import collections

import pandas as pd
import numpy as np
from pathlib import Path

from preprocess import sessions as ss
from preprocess import prepare_data as pp
from analysis import dynamics_analysis as da
from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants


def obtain_manifold_spontaneous(folder_list: list) -> pd.DataFrame:
    """ function to obtain manifold for all behav experiments """
    ret = collections.defaultdict(list)
    df_sessions = ss.get_sessions_df(folder_list, 'BEHAVIOR')
    mice = df_sessions.mice_name.unique()
    for aa, mouse in enumerate(mice):
        df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
        folder_process = Path(folder_list[ss.find_folder_path(mouse)]) / 'process'
        for index, row in df_sessions_mouse.iterrows():
            folder_processed_experiment = Path(folder_process) / row['session_path'] / 'behavior'
            folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
            ret['mice'].append(mouse)
            dim_array, SOT_array, VAF_array = da.obtain_manifold(folder_suite2p)
            ret['dim_sa'].append(dim_array[2])
            ret['dim_all'].append(dim_array[0])
            ret['dim_d1r'].append(dim_array[1])
            ret['SOT_sa'].append(SOT_array[2])
            ret['SOT_all'].append(SOT_array[0])
            ret['SOT_d1r'].append(SOT_array[1])
            ret['VAF_sa'].append(VAF_array[2])
            ret['VAF_all'].append(VAF_array[0])
            ret['VAF_d1r'].append(VAF_array[1])
    return pd.DataFrame(ret)


def obtain_SOT(folder_list: list) -> pd.DataFrame:
    """ function to obtain SOT for all experiments with neurons x time """
    ret = collections.defaultdict(list)
    for experiment_type in AnalysisConstants.experiment_types:
        if experiment_type not in ['CONTROL', 'CONTROL_AGO']:
            df_sessions = ss.get_sessions_df(folder_list, experiment_type)
            mice = df_sessions.mice_name.unique()
            for aa, mouse in enumerate(mice):
                df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
                folder_process = Path(folder_list[ss.find_folder_path(mouse)]) / 'process'
                for index, row in df_sessions_mouse.iterrows():
                    print('SOT of ' + row['session_path'])
                    folder_suite2p = Path(folder_process) / row['session_path']  / 'suite2p' / 'plane0'
                    ret['mice'].append(mouse)
                    ret['session_path'].append(row['session_path'])
                    ret['experiment'].append(experiment_type)
                    SOT_stim_dn, SOT_stim_in, SOT_stim_all, DIM_stim_all = \
                        da.obtain_SOT_over_time(folder_suite2p, tos='stim')
                    SOT_dn_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    SOT_in_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    SOT_all_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    DIM_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    min_x = np.min([AnalysisConfiguration.FA_len_SOT, len(SOT_stim_in)])
                    SOT_dn_array[:min_x] = SOT_stim_dn[:min_x]
                    SOT_in_array[:min_x] = SOT_stim_in[:min_x]
                    SOT_all_array[:min_x] = SOT_stim_all[:min_x]
                    DIM_array[:min_x] = DIM_stim_all[:min_x]
                    ret['SOT_stim_dn'].append(SOT_dn_array)
                    ret['SOT_stim_in'].append(SOT_in_array)
                    ret['SOT_stim_all'].append(SOT_all_array)
                    ret['DIM_stim_all'].append(DIM_array)
                    SOT_stim_dn, SOT_stim_in, SOT_stim_all, DIM_stim_all = \
                        da.obtain_SOT_over_time(folder_suite2p, tos='target')
                    SOT_dn_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    SOT_in_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    SOT_all_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    DIM_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    min_x = np.min([AnalysisConfiguration.FA_len_SOT, len(SOT_stim_in)])
                    SOT_dn_array[:min_x] = SOT_stim_dn[:min_x]
                    SOT_in_array[:min_x] = SOT_stim_in[:min_x]
                    SOT_all_array[:min_x] = SOT_stim_all[:min_x]
                    DIM_array[:min_x] = DIM_stim_all[:min_x]
                    ret['SOT_target_dn'].append(SOT_dn_array)
                    ret['SOT_target_in'].append(SOT_in_array)
                    ret['SOT_target_all'].append(SOT_all_array)
                    ret['DIM_target_all'].append(DIM_array)
                    SOT_stim_dn, SOT_stim_in, SOT_stim_all, DIM_stim_all = \
                        da.obtain_SOT_over_time(folder_suite2p, tos='calib')
                    SOT_dn_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    SOT_in_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    SOT_all_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    DIM_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    min_x = np.min([AnalysisConfiguration.FA_len_SOT, len(SOT_stim_in)])
                    SOT_dn_array[:min_x] = SOT_stim_dn[:min_x]
                    SOT_in_array[:min_x] = SOT_stim_in[:min_x]
                    SOT_all_array[:min_x] = SOT_stim_all[:min_x]
                    DIM_array[:min_x] = DIM_stim_all[:min_x]
                    ret['SOT_calib_dn'].append(SOT_dn_array)
                    ret['SOT_calib_in'].append(SOT_in_array)
                    ret['SOT_calib_all'].append(SOT_all_array)
                    ret['DIM_calib_all'].append(DIM_array)

    return pd.DataFrame(ret)


def obtain_SOT_early_late(folder_list: list) -> pd.DataFrame:
    """ function to obtain SOT in early/late depending on temporal windows """
    ret = collections.defaultdict(list)
    for experiment_type in AnalysisConstants.experiment_types:
        if experiment_type not in ['CONTROL', 'CONTROL_AGO']:
            df_sessions = ss.get_sessions_df(folder_list, experiment_type)
            mice = df_sessions.mice_name.unique()
            for aa, mouse in enumerate(mice):
                df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
                folder_process = Path(folder_list[ss.find_folder_path(mouse)]) / 'process'
                for index, row in df_sessions_mouse.iterrows():
                    print('SOT of ' + row['session_path'])
                    folder_suite2p = Path(folder_process) / row['session_path']  / 'suite2p' / 'plane0'
                    SOT_stim_dn_e, SOT_stim_in_e, SOT_stim_all_e, DIM_stim_all_e,\
                    SOT_stim_dn_l, SOT_stim_in_l, SOT_stim_all_l, DIM_stim_all_l = \
                        da.obtain_SOT_EL(folder_suite2p, tos='stim')
                    ret['mice'].append(mouse)
                    ret['session_path'].append(row['session_path'])
                    ret['experiment'].append(experiment_type)
                    ret['period'].append('early')
                    ret['SOT_stim_dn'].append(SOT_stim_dn_e)
                    ret['SOT_stim_in'].append(SOT_stim_in_e)
                    ret['SOT_stim_all'].append(SOT_stim_all_e)
                    ret['DIM_stim_all'].append(DIM_stim_all_e)

                    ret['mice'].append(mouse)
                    ret['session_path'].append(row['session_path'])
                    ret['experiment'].append(experiment_type)
                    ret['period'].append('late')
                    ret['SOT_stim_dn'].append(SOT_stim_dn_l)
                    ret['SOT_stim_in'].append(SOT_stim_in_l)
                    ret['SOT_stim_all'].append(SOT_stim_all_l)
                    ret['DIM_stim_all'].append(DIM_stim_all_l)

    return pd.DataFrame(ret)


def obtain_SOT_line(folder_list: list) -> pd.DataFrame:
    """ function to obtain SOT for all experiments with trials x time """
    ret = collections.defaultdict(list)
    for experiment_type in AnalysisConstants.experiment_types:
        if experiment_type not in ['CONTROL', 'CONTROL_AGO']:
            df_sessions = ss.get_sessions_df(folder_list, experiment_type)
            mice = df_sessions.mice_name.unique()
            for aa, mouse in enumerate(mice):
                df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
                folder_process = Path(folder_list[ss.find_folder_path(mouse)]) / 'process'
                for index, row in df_sessions_mouse.iterrows():
                    print('SOT of ' + row['session_path'])
                    folder_suite2p = Path(folder_process) / row['session_path']  / 'suite2p' / 'plane0'
                    ret['mice'].append(mouse)
                    ret['session_path'].append(row['session_path'])
                    ret['experiment'].append(experiment_type)
                    SOT_stim_dn, SOT_stim_in = da.obtain_SOT_over_time_line(folder_suite2p, tos='stim')
                    SOT_dn_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    SOT_in_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    min_x = np.min([AnalysisConfiguration.FA_len_SOT, len(SOT_stim_in)])
                    SOT_dn_array[:min_x] = SOT_stim_dn[:min_x]
                    SOT_in_array[:min_x] = SOT_stim_in[:min_x]
                    ret['SOT_stim_dn'].append(SOT_dn_array)
                    ret['SOT_stim_in'].append(SOT_in_array)
                    SOT_stim_dn, SOT_stim_in = da.obtain_SOT_over_time_line(folder_suite2p, tos='target')
                    SOT_dn_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    SOT_in_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    min_x = np.min([AnalysisConfiguration.FA_len_SOT, len(SOT_stim_in)])
                    SOT_dn_array[:min_x] = SOT_stim_dn[:min_x]
                    SOT_in_array[:min_x] = SOT_stim_in[:min_x]
                    ret['SOT_target_dn'].append(SOT_dn_array)
                    ret['SOT_target_in'].append(SOT_in_array)
                    SOT_stim_dn, SOT_stim_in = da.obtain_SOT_over_time_line(folder_suite2p, tos='calib')
                    SOT_dn_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    SOT_in_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    min_x = np.min([AnalysisConfiguration.FA_len_SOT, len(SOT_stim_in)])
                    SOT_dn_array[:min_x] = SOT_stim_dn[:min_x]
                    SOT_in_array[:min_x] = SOT_stim_in[:min_x]
                    ret['SOT_calib_dn'].append(SOT_dn_array)
                    ret['SOT_calib_in'].append(SOT_in_array)

    return pd.DataFrame(ret)


def obtain_engagement(folder_list: list) -> pd.DataFrame:
    """ function to obtain engagement of indirect neurons for all experiments with neurons x time """
    ret = collections.defaultdict(list)
    for experiment_type in AnalysisConstants.experiment_types:
        if experiment_type not in ['CONTROL', 'CONTROL_AGO']:
            df_sessions = ss.get_sessions_df(folder_list, experiment_type)
            mice = df_sessions.mice_name.unique()
            for aa, mouse in enumerate(mice):
                df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
                folder_process = Path(folder_list[ss.find_folder_path(mouse)]) / 'process'
                for index, row in df_sessions_mouse.iterrows():
                    print('eng of ' + row['session_path'])
                    folder_suite2p = Path(folder_process) / row['session_path']  / 'suite2p' / 'plane0'
                    ret['mice'].append(mouse)
                    ret['session_path'].append(row['session_path'])
                    ret['experiment'].append(experiment_type)
                    r2_l, r2_l2, r2_rcv, r2_dff_rcv = da.obtain_engagement(folder_suite2p, tos='stim')
                    r2_l_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    r2_l2_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    r2_rcv_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    r2_dff_rcv_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    min_x = np.min([AnalysisConfiguration.FA_len_SOT, len(r2_l)])
                    r2_l_array[:min_x] = r2_l[:min_x]
                    r2_l2_array[:min_x] = r2_l2[:min_x]
                    r2_rcv_array[:min_x] = r2_rcv[:min_x]
                    r2_dff_rcv_array[:min_x] = r2_dff_rcv[:min_x]
                    ret['r2_l_stim'].append(r2_l_array)
                    ret['r2_l2_stim'].append(r2_l2_array)
                    ret['r2_rcv_stim'].append(r2_rcv_array)
                    ret['r2_dff_rcv_stim'].append(r2_dff_rcv_array)
                    r2_l, r2_l2, r2_rcv, r2_dff_rcv = da.obtain_engagement(folder_suite2p, tos='target')
                    r2_l_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    r2_l2_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    r2_rcv_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    r2_dff_rcv_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    min_x = np.min([AnalysisConfiguration.FA_len_SOT, len(r2_l)])
                    r2_l_array[:min_x] = r2_l[:min_x]
                    r2_l2_array[:min_x] = r2_l2[:min_x]
                    r2_rcv_array[:min_x] = r2_rcv[:min_x]
                    r2_dff_rcv_array[:min_x] = r2_dff_rcv[:min_x]
                    ret['r2_l_target'].append(r2_l_array)
                    ret['r2_l2_target'].append(r2_l2_array)
                    ret['r2_rcv_target'].append(r2_rcv_array)
                    ret['r2_dff_rcv_target'].append(r2_dff_rcv_array)
                    r2_l, r2_l2, r2_rcv, r2_dff_rcv = da.obtain_engagement(folder_suite2p, tos='calib')
                    r2_l_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    r2_l2_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    r2_rcv_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    r2_dff_rcv_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    min_x = np.min([AnalysisConfiguration.FA_len_SOT, len(r2_l)])
                    r2_l_array[:min_x] = r2_l[:min_x]
                    r2_l2_array[:min_x] = r2_l2[:min_x]
                    r2_rcv_array[:min_x] = r2_rcv[:min_x]
                    r2_dff_rcv_array[:min_x] = r2_dff_rcv[:min_x]
                    ret['r2_l_calib'].append(r2_l_array)
                    ret['r2_l2_calib'].append(r2_l2_array)
                    ret['r2_rcv_calib'].append(r2_rcv_array)
                    ret['r2_dff_rcv_calib'].append(r2_dff_rcv_array)

    return pd.DataFrame(ret)