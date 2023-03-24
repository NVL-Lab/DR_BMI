
__author__ = 'Nuria'

import os
import shutil

import pandas as pd
import numpy as np
import scipy.io as sio

from pathlib import Path
from typing import Tuple

from utils.analysis_constants import AnalysisConstants
from preprocess import sessions as ss


def obtain_online_data(folder_suite2p: Path) -> dict:
    """ Function to retrieve the info inside BMI online """
    bmi_online = sio.loadmat(str(folder_suite2p), simplify_cells=True)
    return bmi_online


def obtain_bad_frames_from_fneu(folder_fneu_old: Path) -> Tuple[np.array, np.array, np.array]:
    """ Function to obtain the frames of stim that need to go """
    fneu_old = np.load(Path(folder_fneu_old) / "Fneu.npy")
    Fmean = np.nanmean(fneu_old, 0) - np.nanmean(fneu_old)
    Fmean[Fmean < AnalysisConstants.height_stim_artifact * np.nanstd(Fmean)] = 0
    bad_frames_index = np.where(Fmean > 0)[0]
    diff_bad_frames = np.diff(bad_frames_index)
    missing_bad_frames = np.where(diff_bad_frames == 2)[0]
    for mbf in missing_bad_frames:
        bad_frames_index = np.append(bad_frames_index, bad_frames_index[mbf]+1)
        Fmean[bad_frames_index[mbf]+1] = np.nanmean([Fmean[bad_frames_index[mbf]], Fmean[bad_frames_index[mbf]+2]])
    bad_frames_index.sort()
    frames_include = np.where(Fmean == 0)[0]
    return bad_frames_index, Fmean.astype(bool), frames_include


def prepare_ops_1st_pass(default_path: Path, ops_path: Path) -> dict:
    """ Function to modify the default ops file before 1st pass"""
    aux_ops = np.load(Path(default_path) / "default_ops.npy", allow_pickle=True)
    ops = aux_ops.take(0)
    np.save(ops_path, ops, allow_pickle=True)
    return ops


def copy_only_mat_files(folder_raw: Path, folder_destination: Path):
    """ function to copy all the mat files without the images (to keep working offline) """
    df_sessions = ss.get_all_sessions()
    for folder_path in df_sessions['index']:
        folder_src = Path(folder_raw) / folder_path
        folder_dst = Path(folder_destination) / folder_path
        if not Path(folder_dst).exists():
            Path(folder_dst).mkdir(parents=True, exist_ok=True)
        list_files = os.listdir(folder_src)
        for file in list_files:
            if file[-3:] == 'mat':
                shutil.copyfile(folder_src / file, folder_dst / file)


def save_post_process(folder_save: Path, exp_info: pd.Series, E1: list, E2: list, exclude: list, added_neurons: list):
    """    Function to save the number of the direct neurons, the bad frames etc
        actually this function is run during the manual sanity check of the raw data"""

    folder_suite_2p = folder_save / exp_info['session_path'] / 'suite2p' / 'plane0'
    folder_fneu_old = folder_save / exp_info['session_path'] / 'suite2p' / 'fneu_old'

    # obtain bad frames
    bad_frames_index, bad_frames_bool, _ = obtain_bad_frames_from_fneu(folder_fneu_old)
    bad_frames_dict = {'bad_frames_index': bad_frames_index, 'bad_frames_bool': bad_frames_bool}
    np.save(Path(folder_suite_2p) / "bad_frames_dict.npy", bad_frames_dict, allow_pickle=True)

    # save
    exclude = []
    added_neurons = []
    E1.sort()
    E2.sort()
    exclude.sort()
    added_neurons.sort()
    direct_neurons = {'E1': E1, 'E2': E2, 'exclude': exclude, 'added_neurons': added_neurons}
    np.save(Path(folder_suite_2p) / "direct_neurons.npy", direct_neurons, allow_pickle=True)


def obtain_dffs(folder_suite2p: Path) -> np.array:
    """ function to obtain the dffs based on F and Fneu """
    Fneu = np.load(Path(folder_suite2p) / "Fneu.npy")
    F_raw = np.load(Path(folder_suite2p) / "F.npy")
    dff = np.full(Fneu.shape, np.nan)
    for neuron in np.arange(dff.shape[0]):
        dff[neuron, :] = (F_raw[neuron, :] - Fneu[neuron, :]) / np.nanmean(Fneu[neuron, :])
    return dff


def obtain_stim_time(bad_frames_bool: np.array) -> Tuple[np.array, np.array]:
    """ function that reports the time of stim (by returning the first frame of each stim) """
    stim_time = np.insert(np.diff(bad_frames_bool.astype(int)), 0, 0)
    stim_time[stim_time < 1] = 0
    return np.where(stim_time)[0], stim_time.astype(bool)









