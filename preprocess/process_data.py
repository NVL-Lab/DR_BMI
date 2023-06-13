__author__ = 'Nuria'

import numpy as np
from pathlib import Path

import preprocess.prepare_data as pp
from suite2p.run_s2p import run_s2p
from preprocess import sessions as ss
from utils.analysis_constants import AnalysisConstants


def run_all_experiments(folder_list: list, folder_temp_save: str = 'C:/Users/Nuria/Documents/DATA/D1exp'):
    """ function to run and process all experiments with suite2p """
    folder_temp_save = Path(folder_temp_save)
    default_path = folder_temp_save / "default_var"
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_list, experiment_type)
        for index, row in df.iterrows():
            if row['mice_name'] not in ['m13', 'm15', 'm16', 'm18']:
                folder_raw = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'raw'
                folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
                folder_raw_experiment = Path(folder_raw) / row['session_path']
                folder_processed_experiment = Path(folder_process) / row['session_path']
                folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
                if not Path(folder_suite2p).exists():
                    Path(folder_suite2p).mkdir(parents=True, exist_ok=True)
                file_origin = folder_raw_experiment / 'im/baseline' / row['Baseline_im']
                data_path = [str(file_origin),
                             str(folder_raw_experiment / 'im' / row["Experiment_dir"] / row['Experiment_im'])]
                db = {
                    'data_path': data_path,
                    'save_path0': str(folder_processed_experiment),
                    'fast_disk': str(Path(folder_temp_save)),
                }

                ops_1st_pass = pp.prepare_ops_1st_pass(default_path, folder_suite2p / 'ops_before_1st.npy')
                ops_after_1st_pass = run_s2p(ops_1st_pass, db)
                np.save(folder_suite2p / 'ops_after_1st_pass.npy', ops_after_1st_pass, allow_pickle=True)


def create_bad_frames_after_first_pass(folder_list: list):
    """ function to obtain and save bad_frames to do a second pass """
    # only experiments that have artifacts
    for experiment_type in ['D1act', 'CONTROL_LIGHT', 'RANDOM', 'NO_AUDIO', 'DELAY']:
        df = ss.get_sessions_df(folder_list, experiment_type)
        for index, row in df.iterrows():
            folder_raw = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'raw'
            folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
            folder_experiment = Path(folder_raw) / row['session_path']
            folder_processed_experiment = Path(folder_process) / row['session_path']
            folder_fneu_old = folder_processed_experiment / 'suite2p' / 'fneu_old'
            fneu_old = np.load(Path(folder_fneu_old) / "Fneu.npy")
            bad_frames, _, _, _, _, _ = pp.obtain_bad_frames_from_fneu(fneu_old)
            file_origin = folder_experiment / "im/baseline" / row["Baseline_im"]
            np.save(file_origin / 'bad_frames.npy', bad_frames)


def create_dff(folder_list: list):
    """ function to obtain the dffs of each experiment """
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_list, experiment_type)
        for index, row in df.iterrows():
            folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
            folder_processed_experiment = Path(folder_process) / row['session_path']
            folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
            dff = pp.obtain_dffs(folder_suite2p)
            np.save(folder_suite2p / 'dff.npy', dff)


def run_refines_and_sanity_checks(folder_list: list):
    """ function to run sanity checks in all experiments"""
    sessions_to_double_check = []
    stim_flag = True
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_list, experiment_type)
        if experiment_type.isin(['CONTROL', 'CONTROL_AGO']):
            stim_flag = False
        for index, row in df.iterrows():
            folder_raw = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'raw'
            folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
            folder_suite2p = folder_process / row['session_path'] / 'suite2p' / 'plane0'
            folder_fneu_old = folder_process / row['session_path'] / 'suite2p' / 'fneu_old'
            folder_process_plots = folder_process / row['session_path'] / 'suite2p' / 'plots'
            folder_experiment = folder_raw / row['session_path']
            if not Path(folder_process_plots).exists():
                Path(folder_process_plots).mkdir(parents=True, exist_ok=True)
            pp.refine_classifier(folder_suite2p)
            check_session = pp.sanity_checks(folder_suite2p, folder_fneu_old, folder_process_plots, stim_flag)
            if len(check_session) > 0: sessions_to_double_check.append([row['session_path'], check_session])
    return sessions_to_double_check
