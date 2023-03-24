
__author__ = 'Nuria'

import numpy as np
from pathlib import Path

from suite2p.run_s2p import run_s2p
from preprocess import sessions as ss
from preprocess.prepare_data import prepare_ops_1st_pass, obtain_bad_frames_from_fneu, obtain_dffs
from utils.analysis_constants import AnalysisConstants


def run_process_data():
    """ Function to run the data process of a given experiment """


def run_all_experiments(folder_raw: Path, folder_save: Path):
    """ function to run and process all experiments with suite2p """
    default_path = folder_save / "default_var"
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_raw, experiment_type)
        for index, row in df.iterrows():
            folder_experiment = Path(folder_raw) / row['session_path']
            folder_processed_experiment = Path(folder_save) / row['session_path']
            folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
            if not Path(folder_suite2p).exists():
                Path(folder_suite2p).mkdir(parents=True, exist_ok=True)
            file_origin = folder_experiment / 'im/baseline' / row['Baseline_im']
            data_path = [str(file_origin),
                         str(folder_experiment / 'im' / row["Experiment_dir"] / row['Experiment_im'])]
            db = {
                'data_path': data_path,
                'save_path0': str(folder_processed_experiment),
                'fast_disk': str(Path('C:/Users/Nuria/Documents/DATA')),
            }

            ops_1st_pass = prepare_ops_1st_pass(default_path, folder_suite2p / 'ops_before_1st.npy')
            ops_after_1st_pass = run_s2p(ops_1st_pass, db)
            np.save(folder_suite2p / 'ops_after_1st_pass.npy', ops_after_1st_pass, allow_pickle=True)


def create_bad_frames_after_first_pass(folder_raw: Path, folder_save: Path):
    """ function to obtain and save bad_frames to do a second pass """
    # only experiments that have artifacts
    for experiment_type in ['BMI_STIM_AGO', 'BMI_CONTROL_RANDOM', 'BMI_CONTROL_LIGHT']:
        df = ss.get_sessions_df(folder_raw, experiment_type)
        for index, row in df.iterrows():
            folder_experiment = Path(folder_raw) / row['session_path']
            folder_processed_experiment = Path(folder_save) / row['session_path']
            folder_fneu_old = folder_processed_experiment / 'suite2p' / 'fneu_old'
            bad_frames, _, _ = obtain_bad_frames_from_fneu(folder_fneu_old)
            file_origin = folder_experiment / "im/baseline" / row["Baseline_im"]
            np.save(file_origin / 'bad_frames.npy', bad_frames)


def create_dff(folder_raw: Path, folder_save: Path):
    """ function to obtain the dffs of each experiment """
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_raw, experiment_type)
        for index, row in df.iterrows():
            folder_processed_experiment = Path(folder_save) / row['session_path']
            folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
            dff = obtain_dffs(folder_suite2p)
            np.save(folder_suite2p / 'dff.npy', dff)

