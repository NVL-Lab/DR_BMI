__author__ = 'Nuria'

import collections
from typing import Tuple

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


def obtain_manifold_spontaneous_time(folder_list: list) -> pd.DataFrame:
    """ function to obtain manifold for all behav experiments """
    ret = collections.defaultdict(list)
    df_sessions = ss.get_sessions_df(folder_list, 'BEHAVIOR')
    mice = df_sessions.mice_name.unique()
    time_points = 30
    for aa, mouse in enumerate(mice):
        df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
        folder_process = Path(folder_list[ss.find_folder_path(mouse)]) / 'process'
        for index, row in df_sessions_mouse.iterrows():
            folder_processed_experiment = Path(folder_process) / row['session_path'] / 'behavior'
            folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
            ret['mice'].append(mouse)
            dim_array, SOT_array, VAF_array = da.obtain_manifold_time(folder_suite2p, time_points=time_points)
            ret['dim'].append(dim_array)
            ret['SOT'].append(SOT_array)
            ret['VAF'].append(VAF_array)
    return pd.DataFrame(ret)


def obtain_population_SOT(folder_list: list) -> pd.DataFrame:
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
                    folder_suite2p = Path(folder_process) / row['session_path'] / 'suite2p' / 'plane0'
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


def obtain_population_SOT_windows(folder_list: list, remove_target: bool = True) -> pd.DataFrame:
    """ function to obtain SOT in early/late depending on temporal windows removing events """
    ret = collections.defaultdict(list)

    windows = []

    aux_w = np.arange(AnalysisConstants.calibration_frames, 0,
                                       - int(AnalysisConfiguration.FA_time_win * AnalysisConstants.framerate * 60))[::-1]
    for ww, win in enumerate(aux_w[1:]):
        windows.append((aux_w[ww], win))
    len_base_windows = len(windows)

    aux_w = np.arange(AnalysisConstants.calibration_frames, AnalysisConfiguration.max_len_spks,
                                  int(AnalysisConfiguration.FA_time_win * AnalysisConstants.framerate * 60))
    for ww, win in enumerate(aux_w[1:]):
        windows.append((aux_w[ww], win))

    df_sessions = ss.get_sessions_df(folder_list, 'D1act')
    mice = df_sessions.mice_name.unique()
    for aa, mouse in enumerate(mice):
        df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
        folder_process = Path(folder_list[ss.find_folder_path(mouse)]) / 'process'
        for index, row in df_sessions_mouse.iterrows():
            print('SOT windows of ' + row['session_path'])
            folder_suite2p = Path(folder_process) / row['session_path'] / 'suite2p' / 'plane0'
            for ww, win in enumerate(windows):
                SOT_dn, SOT_in, DIM_dn, DIM_in = da.obtain_SOT_windows(folder_suite2p, win, remove_target=remove_target)
                ret['mice'].append(mouse)
                ret['session_path'].append(row['session_path'])
                ret['window'].append(ww)
                if ww < len_base_windows:
                    ret['win_type'].append('calib')
                else:
                    ret['win_type'].append('exp')
                ret['SOT_dn'].append(SOT_dn)
                ret['SOT_in'].append(SOT_in)
                ret['DIM_dn'].append(DIM_dn)
                ret['DIM_in'].append(DIM_in)

    return pd.DataFrame(ret)


def obtain_population_SOT_line(folder_list: list) -> pd.DataFrame:
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
                    folder_suite2p = Path(folder_process) / row['session_path'] / 'suite2p' / 'plane0'
                    ret['mice'].append(mouse)
                    ret['session_path'].append(row['session_path'])
                    ret['experiment'].append(experiment_type)
                    SOT_stim_dn, SOT_stim_in = da.obtain_SOT_over_all_lines(folder_suite2p, tos='stim')
                    SOT_dn_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    SOT_in_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    min_x = np.min([AnalysisConfiguration.FA_len_SOT, len(SOT_stim_in)])
                    SOT_dn_array[:min_x] = SOT_stim_dn[:min_x]
                    SOT_in_array[:min_x] = SOT_stim_in[:min_x]
                    ret['SOT_stim_dn'].append(SOT_dn_array)
                    ret['SOT_stim_in'].append(SOT_in_array)
                    SOT_stim_dn, SOT_stim_in = da.obtain_SOT_over_all_lines(folder_suite2p, tos='target')
                    SOT_dn_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    SOT_in_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    min_x = np.min([AnalysisConfiguration.FA_len_SOT, len(SOT_stim_in)])
                    SOT_dn_array[:min_x] = SOT_stim_dn[:min_x]
                    SOT_in_array[:min_x] = SOT_stim_in[:min_x]
                    ret['SOT_target_dn'].append(SOT_dn_array)
                    ret['SOT_target_in'].append(SOT_in_array)
                    SOT_stim_dn, SOT_stim_in = da.obtain_SOT_over_all_lines(folder_suite2p, tos='calib')
                    SOT_dn_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    SOT_in_array = np.full(AnalysisConfiguration.FA_len_SOT, np.nan)
                    min_x = np.min([AnalysisConfiguration.FA_len_SOT, len(SOT_stim_in)])
                    SOT_dn_array[:min_x] = SOT_stim_dn[:min_x]
                    SOT_in_array[:min_x] = SOT_stim_in[:min_x]
                    ret['SOT_calib_dn'].append(SOT_dn_array)
                    ret['SOT_calib_in'].append(SOT_in_array)

    return pd.DataFrame(ret)


def obtain_population_engagement(folder_list: list, line_flag: bool = False) -> pd.DataFrame:
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
                    folder_suite2p = Path(folder_process) / row['session_path'] / 'suite2p' / 'plane0'
                    ret['mice'].append(mouse)
                    ret['session_path'].append(row['session_path'])
                    ret['experiment'].append(experiment_type)
                    if line_flag:
                        r2_l, r2_l2, r2_rcv, r2_dff_rcv = da.obtain_engagement_line(folder_suite2p, tos='stim')
                    else:
                        r2_l, r2_l2, r2_rcv, r2_dff_rcv = da.obtain_engagement_event(folder_suite2p, tos='stim')
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
                    if line_flag:
                        r2_l, r2_l2, r2_rcv, r2_dff_rcv = da.obtain_engagement_line(folder_suite2p, tos='target')
                    else:
                        r2_l, r2_l2, r2_rcv, r2_dff_rcv = da.obtain_engagement_event(folder_suite2p, tos='target')
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
                    if line_flag:
                        r2_l, r2_l2, r2_rcv, r2_dff_rcv = da.obtain_engagement_line(folder_suite2p, tos='calib')
                    else:
                        r2_l, r2_l2, r2_rcv, r2_dff_rcv = da.obtain_engagement_event(folder_suite2p, tos='calib')
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


def obtain_population_trial_engagement(folder_list: list, experiment_type: str) -> Tuple[np.array, np.array, np.array, np.array]:
    """ function to obtain engagement of indirect neurons for all experiments with neurons x time """
    indices_lag = np.arange(-150, 60, 15)
    AnalysisConfiguration.FA_rew_frames = 30
    AnalysisConfiguration.eng_event_frames = 30
    if experiment_type not in ['CONTROL', 'CONTROL_AGO']:
        df_sessions = ss.get_sessions_df(folder_list, experiment_type)
        mice = df_sessions.mice_name.unique()
        r2_l_e = np.full((30, len(indices_lag), len(mice), 10), np.nan)
        r2_l2_e = np.full((30, len(indices_lag), len(mice), 10), np.nan)
        r2_rcv_e = np.full((30, len(indices_lag), len(mice), 10), np.nan)
        r2_dff_rcv_e = np.full((30, len(indices_lag), len(mice), 10), np.nan)
        for aa, mouse in enumerate(mice):
            df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse].reset_index(drop=True)
            folder_process = Path(folder_list[ss.find_folder_path(mouse)]) / 'process'
            for index, row in df_sessions_mouse.iterrows():
                print('eng of ' + row['session_path'])
                folder_suite2p = Path(folder_process) / row['session_path'] / 'suite2p' / 'plane0'
                r2_l, r2_l2, r2_rcv, r2_dff_rcv = da.obtain_engagement_trial(folder_suite2p, indices_lag, tos='stim')
                min_x = np.min([r2_l.shape[0], r2_l_e.shape[0]])
                r2_l_e[:min_x, :, aa, index] = r2_l[:min_x, :]
                r2_l2_e[:min_x, :, aa, index] = r2_l2[:min_x, :]
                r2_rcv_e[:min_x, :, aa, index] = r2_rcv[:min_x, :]
                r2_dff_rcv_e[:min_x, :, aa, index] = r2_dff_rcv[:min_x, :]
    else:
        return np.nan, np.nan, np.nan, np.nan
    return r2_l_e, r2_l2_e, r2_rcv_e, r2_dff_rcv_e


def obtain_population_trial_SOT(folder_list: list, experiment_type: str, line: bool = False) -> Tuple[np.array, np.array]:
    """ function to obtain engagement of indirect neurons for all experiments with neurons x time """
    indices_lag = np.arange(-120, 90, 6)
    AnalysisConfiguration.FA_rew_frames = 15
    AnalysisConfiguration.FA_event_frames = 15
    if experiment_type not in ['CONTROL', 'CONTROL_AGO']:
        df_sessions = ss.get_sessions_df(folder_list, experiment_type)
        mice = df_sessions.mice_name.unique()
        SOT_all_dn = np.full((30, len(indices_lag), len(mice), 10), np.nan)
        SOT_all_in = np.full((30, len(indices_lag), len(mice), 10), np.nan)

        for aa, mouse in enumerate(mice):
            df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse].reset_index(drop=True)
            folder_process = Path(folder_list[ss.find_folder_path(mouse)]) / 'process'
            for index, row in df_sessions_mouse.iterrows():
                print('SOT of ' + row['session_path'])
                folder_suite2p = Path(folder_process) / row['session_path'] / 'suite2p' / 'plane0'
                if line:
                    SOT_t_dn, SOT_t_in = da.obtain_SOT_over_all_trials(folder_suite2p, indices_lag)
                else:
                    SOT_t_dn, SOT_t_in = da.obtain_SOT_over_trial(folder_suite2p, indices_lag)
                min_x = np.min([SOT_t_dn.shape[0], SOT_all_dn.shape[0]])
                SOT_all_dn[:min_x, :, aa, index] = SOT_t_dn[:min_x, :]
                SOT_all_in[:min_x, :, aa, index] = SOT_t_in[:min_x, :]
    else:
        return np.nan, np.nan
    return SOT_all_dn, SOT_all_in


def obtain_df_events_population(folder_list: list, win: int = 10) -> pd.DataFrame:
    """ function to obtain diverse metrics of events """
    # df_sessions = pd.concat([ss.get_sessions_df(folder_list, 'RANDOM'),
                            # ss.get_sessions_df(folder_list, 'DELAY')])
    df_sessions = ss.get_sessions_df(folder_list, 'D1act')
    mice = df_sessions.mice_name.unique()
    # list_dfs = []
    list_df_stim = []

    for aa, mouse in enumerate(mice):
        df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
        folder_process = Path(folder_list[ss.find_folder_path(mouse)]) / 'process'
        for index, row in df_sessions_mouse.iterrows():
            print("analyzing now:  " + row.session_path)
            folder_suite2p = Path(folder_process) / row['session_path'] / 'suite2p' / 'plane0'
            # aux_df = da.obtain_dp_events_per_event(folder_suite2p, AnalysisConfiguration.zscore_thres, win)
            aux_df_stim = da.obtain_dp_events_per_stim(folder_suite2p, AnalysisConfiguration.zscore_thres, win)
            # aux_df['mice'] = mouse
            aux_df_stim['mice'] = mouse
            # aux_df['session_path'] = row.session_path
            aux_df_stim['session_path'] = row.session_path
            # list_dfs.append(aux_df)
            list_df_stim.append(aux_df_stim)
    # df = pd.concat(list_dfs)
    df_stim = pd.concat(list_df_stim)
    return df_stim