
__author__ = 'Nuria'

import collections
import os
import shutil

import pandas as pd
import numpy as np
import sklearn as sk
import scipy.io as sio

from pathlib import Path
from typing import Optional, Tuple

from utils.analysis_constants import AnalysisConstants
from preprocess import sessions as ss


def obtain_target_info (file_path: Path) -> pd.DataFrame:
    """ Function to retrieve the info inside target_info"""
    # Todo check what is wrong with scipy.io and other ways to load mat in python


def obtain_online_data(file_path: Path) -> dict:
    """ Function to retrieve the info inside BMI online """
    bmi_online = sio.loadmat(file_path, simplify_cells=True)
    return bmi_online


def obtain_bad_frames_from_fneu(file_path: Path) -> Tuple[np.array, np.array, np.array]:
    """ Function to obtain the frames of stim that need to go """
    Fneu = np.load(Path(file_path) / "Fneu.npy")
    Fdiff = np.diff(np.nanmean(Fneu, 0))
    Fdiff[Fdiff < AnalysisConstants.height_stim_artifact * np.nanstd(Fdiff)] = 0
    Fconv = np.convolve(Fdiff, np.ones(AnalysisConstants.size_stim_frames))
    Fconv[Fconv > 0] = 1
    Fconv = Fconv[:Fneu.shape[1]]
    bad_frames = np.where(Fconv > 0)[0]
    frames_include = np.where(Fconv == 0)[0]
    return bad_frames, Fconv.astype(bool), frames_include


def prepare_ops_1st_pass(default_path: Path, ops_path: Path) -> dict:
    """ Function to modify the default ops file before 1st pass"""
    aux_ops = np.load(Path(default_path) / "default_ops.npy", allow_pickle=True)
    ops = aux_ops.take(0)
    np.save(ops_path, ops, allow_pickle=True)
    return ops


def prepare_ops_file_2nd_pass(default_path: Path, file_path: Path, file_origin: Path) -> dict:
    """ Function to modify the ops file before 2nd pass"""
    # copy the directory to save 1st pass data
    shutil.copytree(file_path, file_path/'1st_pass', dirs_exist_ok=True)
    # obtain bad frames
    bad_frames_to_save, bad_frames_ops, frames_include = obtain_bad_frames_from_fneu(file_path)
    # load the default ops and change for the second round
    aux_ops = np.load(Path(default_path) / "default_ops.npy", allow_pickle=True)
    ops = aux_ops.take(0)
    # ops['frames_include'] = frames_include
    # ops["badframes"] = bad_frames_ops
    ops["delete_bin"] = False
    ops['anatomical_only'] = 0
    # save new ops and the file of bad_frames
    np.save(Path(file_path) / "ops.npy", ops, allow_pickle=True)
    np.save(Path(file_origin) / 'bad_frames.npy', bad_frames_to_save)
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



