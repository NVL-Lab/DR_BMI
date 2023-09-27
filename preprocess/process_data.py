__author__ = 'Nuria'

import os
import collections
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import scipy.io as sio

import preprocess.prepare_data as pp
from suite2p.run_s2p import run_s2p
from suite2p.extraction import dcnv
from preprocess import sessions as ss
from preprocess import sessions_all as ssa
from utils.analysis_constants import AnalysisConstants


def run_all_experiments(folder_list: list, folder_temp_save: str = 'C:/Users/Nuria/Documents/DATA/D1exp',
                        npass: int = 1):
    """ function to run and process all experiments with suite2p """
    folder_temp_save = Path(folder_temp_save)
    default_path = folder_temp_save / "default_var"
    for experiment_type in AnalysisConstants.experiment_types:
        # ['D1act', 'CONTROL_LIGHT', 'RANDOM', 'NO_AUDIO', 'DELAY']:
        df = ss.get_sessions_df(folder_list, experiment_type)
        for index, row in df.iterrows():
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
            if npass == 1:
                ops_1st_pass = pp.prepare_ops_1st_pass(default_path, folder_suite2p / 'ops_before_1st.npy')
                ops_after_1st_pass = run_s2p(ops_1st_pass, db)
                np.save(folder_suite2p / 'ops_after_1st_pass.npy', ops_after_1st_pass, allow_pickle=True)
            else:
                aux_ops = np.load(Path(default_path) / "default_ops.npy", allow_pickle=True)
                ops = aux_ops.take(0)
                ops_after_n_pass = run_s2p(ops, db)
                ops_name = 'ops_after_' + str(npass) + '_pass.npy'
                np.save(folder_suite2p / ops_name, ops_after_n_pass, allow_pickle=True)


def run_behav_baseline(folder_list: list, folder_temp_save: str = 'C:/Users/Nuria/Documents/DATA/D1exp'):
    """ function to run the suite2p bor laser on-off on behavior neural data"""
    folder_temp_save = Path(folder_temp_save)
    default_path = folder_temp_save / "default_var"
    df = ss.get_neural_data_behav(folder_list)
    for index, row in df.iterrows():
        folder_raw = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'raw'
        folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
        folder_raw_experiment = Path(folder_raw) / row['session_path']
        folder_processed_experiment = Path(folder_process) / row['session_path'] / 'behavior'
        folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
        if not Path(folder_suite2p).exists():
            Path(folder_suite2p).mkdir(parents=True, exist_ok=True)
        file_origin = folder_raw_experiment / 'im/baseline' / row['Baseline_im']
        data_path = [str(folder_raw_experiment / 'im/behavior' / row['Behavior_im']), str(file_origin)]
        db = {
            'data_path': data_path,
            'save_path0': str(folder_processed_experiment),
            'fast_disk': str(Path(folder_temp_save)),
        }

        ops_behav = pp.prepare_ops_behav_pass(default_path, folder_suite2p / 'ops_behav.npy')
        ops_after_behav_pass = run_s2p(ops_behav, db)
        np.save(folder_suite2p / 'ops_after_behav_pass.npy', ops_after_behav_pass, allow_pickle=True)


def create_bad_frames_after_first_pass(folder_list: list):
    """ function to obtain and save bad_frames to do a second pass """
    # only experiments that have artifacts
    for experiment_type in ['D1act', 'CONTROL_LIGHT', 'RANDOM', 'NO_AUDIO', 'DELAY']:
        df = ss.get_sessions_df(folder_list, experiment_type)
        for index, row in df.iterrows():
            if row['mice_name'] not in ['m13', 'm15', 'm16', 'm18']:
                folder_raw = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'raw'
                folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
                folder_experiment = Path(folder_raw) / row['session_path']
                folder_processed_experiment = Path(folder_process) / row['session_path']
                folder_fneu_old = folder_processed_experiment / 'suite2p' / 'fneu_old'
                if os.path.exists(folder_fneu_old):
                    fneu_old = np.load(Path(folder_fneu_old) / "Fneu.npy")
                    bad_frames, _, _, _, _, _ = pp.obtain_bad_frames_from_fneu(fneu_old)
                    file_origin = folder_experiment / "im/baseline" / row["Baseline_im"]
                    np.save(file_origin / 'bad_frames.npy', bad_frames)
                else:
                    print(folder_fneu_old)


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


def create_spks_from_dff(folder_list: list, default_path):
    """ function to obtain the spks from dff for each experiment """
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_list, experiment_type)
        for index, row in df.iterrows():
            print('creating the spks for ' + row['session_path'])
            folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
            folder_processed_experiment = Path(folder_process) / row['session_path']
            folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
            spks_dff = obtain_spks_from_dff(folder_suite2p, default_path)
            np.save(folder_suite2p / 'spks_dff.npy', spks_dff)


def obtain_spks_from_dff(folder_suite2p: Path, default_path: Path, dff: Optional[np.array] = None) -> np.array:
    """ function to obtain spks based on dff to remove stim artifacts """
    aux_ops = np.load(Path(default_path) / "default_ops.npy", allow_pickle=True)
    ops = aux_ops.take(0)
    if dff is None:
        dff = pp.obtain_dffs(folder_suite2p)

    # baseline operation
    Fc = dcnv.preprocess(
        F=dff,
        baseline=ops['baseline'],
        win_baseline=ops['win_baseline'],
        sig_baseline=ops['sig_baseline'],
        fs=ops['fs'],
        prctile_baseline=ops['prctile_baseline']
    )
    # get spikes
    spks_dff = dcnv.oasis(F=Fc, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])
    return spks_dff


def run_refines(folder_list: list, behav_bool: bool = False):
    """ function to refine the is_cells in all experiments"""
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_list, experiment_type)

        for index, row in df.iterrows():
            folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
            folder_path = folder_process / row['session_path']
            if behav_bool:
                folder_path = folder_path / 'behavior'
            folder_suite2p = folder_path / 'suite2p' / 'plane0'
            print('refining: ' + row['session_path'])
            pp.refine_classifier(folder_suite2p, dn_bool=False)


def run_sanity_checks(folder_list: list):
    """ function to run sanity checks in all experiments"""
    sessions_to_double_check = []
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_list, experiment_type)
        if experiment_type in ['CONTROL', 'CONTROL_AGO']:
            stim_flag = False
        else:
            stim_flag = True
        for index, row in df.iterrows():
            folder_raw = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'raw'
            folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
            folder_path = folder_process / row['session_path']
            file_path = folder_raw / row['session_path'] / row['BMI_online']
            folder_suite2p = folder_path / 'suite2p' / 'plane0'
            if Path(folder_suite2p).exists():
                folder_fneu_old = folder_path / 'suite2p' / 'fneu_old'
                folder_process_plots = folder_path / 'suite2p' / 'plots'
                if not Path(folder_process_plots).exists():
                    Path(folder_process_plots).mkdir(parents=True, exist_ok=True)
                check_session = pp.sanity_checks(folder_suite2p, folder_fneu_old, file_path, folder_process_plots,
                                                 stim_flag=stim_flag, save_flag=False)
                if len(check_session) > 0:
                    sessions_to_double_check.append([row['session_path'], check_session])
    return sessions_to_double_check


def run_obtain_target_index(folder_list: list):
    """ function to obtain the indexes of the targets """
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_list, experiment_type)
        if experiment_type in ['CONTROL', 'CONTROL_AGO']:
            continue
        for index, row in df.iterrows():
            folder_raw = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'raw'
            folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
            folder_path = folder_process / row['session_path']
            file_path = folder_raw / row['session_path'] / row['BMI_online']
            folder_suite2p = folder_path / 'suite2p' / 'plane0'
            pp.save_targets(experiment_type, file_path, folder_suite2p)


def obtain_snr(folder_list: list) -> pd.DataFrame:
    """ function to obtain the snr of all the experiments """
    ret = collections.defaultdict(list)
    for experiment_type in AnalysisConstants.experiment_types:
        df = ssa.get_sessions_df(folder_list, experiment_type)
        for index, row in df.iterrows():
            ret['mice_name'].append(row['mice_name'])
            ret['experiment_type'].append(row['experiment_type'])
            ret['session_path'].append(row['session_path'])
            ret['day_index'].append(row['day_index'])
            folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
            folder_processed_experiment = Path(folder_process) / row['session_path']
            folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
            snr_all, snr_dn, snr_dn_min = pp.obtain_SNR_per_neuron(folder_suite2p)
            ret['snr_all'].append(snr_all)
            ret['snr_dn'].append(snr_dn)
            ret['snr_dn_min'].append(snr_dn_min)
    return pd.DataFrame(ret)


def create_direct_neurons_mat(folder_list: list):
    """ Function to open the direct_neurons dict and create a mat out of them """
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_list, experiment_type)
        for index, row in df.iterrows():
            folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
            folder_processed_experiment = Path(folder_process) / row['session_path']
            folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
            direct_neurons_aux = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
            direct_neurons = direct_neurons_aux.take(0)
            sio.savemat(folder_suite2p / 'direct_neurons.mat', direct_neurons)


def create_bad_frames_mat(folder_list: list):
    """ Function to open the bad_frames dict and create a mat out of them """
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_list, experiment_type)
        for index, row in df.iterrows():
            folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
            folder_processed_experiment = Path(folder_process) / row['session_path']
            folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
            file_path = Path(folder_suite2p) / "bad_frames_dict.npy"
            if os.path.exists(file_path):
                bad_frames_aux = np.load(file_path, allow_pickle=True)
                bad_frames = bad_frames_aux.take(0)
                sio.savemat(folder_suite2p / 'bad_frames.mat', bad_frames)
