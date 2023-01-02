
__author__ = 'Nuria'

import pandas as pd
import numpy as np
from pathlib import Path

from suite2p.run_s2p import run_s2p
from preprocess import sessions as ss
from preprocess.prepare_data import prepare_ops_file_2nd_pass, prepare_ops_1st_pass
from utils.analysis_constants import AnalysisConstants


def run_process_data():
    """ Function to run the process data of a given experiment """


def run_all_experiments(folder_raw: Path, folder_save: Path):
    """ function to run and process all experiments with suite2p """
    default_path = folder_save / "default_var"
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_raw, experiment_type)
        for index, row in df.iterrows():
            folder_experiment = Path(folder_raw) / row['session_path']
            folder_processed_experiment = Path(folder_save) / row['session_path']
            file_path = folder_processed_experiment / 'suite2p' / 'plane0'
            if not Path(file_path).exists():
                Path(file_path).mkdir(parents=True, exist_ok=True)
            file_origin = folder_experiment / 'im/baseline' / row['Baseline_im']
            data_path = [str(file_origin),
                         str(folder_experiment / 'im' / row["Experiment_dir"] / row['Experiment_im'])]
            db = {
                'data_path': data_path,
                'save_path0': str(folder_processed_experiment),
                'ops_path': str(file_path / 'ops.npy'),
                'fast_disk': str(Path('C:/Users/Nuria/Documents/DATA')),
            }

            ops_1st_pass = prepare_ops_1st_pass(default_path, Path(db['ops_path']))
            ops_after_1st_pass = run_s2p(ops_1st_pass, db)
            np.save(file_path / 'ops_before_1st.npy', ops_1st_pass, allow_pickle=True)
            np.save(file_path / 'ops_after_1st_pass.npy', ops_after_1st_pass, allow_pickle=True)

            # ops_2nd_pass = prepare_ops_file_2nd_pass(default_path, file_path, file_origin)
            # np.save(file_path / 'ops_before_2nd.npy', ops_2nd_pass, allow_pickle=True)
            # ops_after_2nd_pass = run_s2p(ops_2nd_pass, db)


def run_all_second_pass(folder_raw: Path, folder_save: Path):
    """ function to run and process all experiments with suite2p """
    default_path = folder_save / "default_var"
    for experiment_type in AnalysisConstants.experiment_types:
        df = ss.get_sessions_df(folder_raw, experiment_type)
        for index, row in df.iterrows():
            folder_experiment = Path(folder_raw) / row['session_path']
            folder_processed_experiment = Path(folder_save) / row['session_path']
            file_path = folder_processed_experiment / 'suite2p' / 'plane0'
            if not Path(file_path).exists():
                Path(file_path).mkdir(parents=True, exist_ok=True)
            file_origin = folder_experiment / 'im/baseline' / row['Baseline_im']
            data_path = [str(file_origin),
                         str(folder_experiment / 'im' / row["Experiment_dir"] / row['Experiment_im'])]
            db = {
                'data_path': data_path,
                'save_path0': str(folder_processed_experiment),
                'ops_path': str(file_path / 'ops.npy'),
                'fast_disk': str(Path('C:/Users/Nuria/Documents/DATA')),
            }

            ops_2nd_pass = prepare_ops_file_2nd_pass(default_path, file_path, file_origin)
            np.save(file_path / 'ops_before_2nd.npy', ops_2nd_pass, allow_pickle=True)
            ops_after_2nd_pass = run_s2p(ops_2nd_pass, db)
            np.save(file_path / 'ops_before_2nd.npy', ops_2nd_pass, allow_pickle=True)
            np.save(file_path / 'ops_after_2nd_pass.npy', ops_after_2nd_pass, allow_pickle=True)

