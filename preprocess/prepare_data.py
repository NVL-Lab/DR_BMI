
__author__ = 'Nuria'

import collections
import os
import shutil

import pandas as pd
import numpy as np
import sklearn as sk
import scipy.io as sio

from pathlib import Path
from typing import Optional

from utils.analysis_constants import AnalysisConstants
from preprocess import sessions as ss


def obtain_target_info (file_path: Path) -> pd.DataFrame:
    """ Function to retrieve the info inside target_info"""
    # Todo check what is wrong with scipy.io and other ways to load mat in python


def obtain_online_data(file_path: Path) -> dict:
    """ Function to retrieve the info inside BMI online """
    bmi_online = sio.loadmat(file_path, simplify_cells=True)
    return bmi_online


def obtain_bad_frames_from_fneu(file_path: Path) -> np.array():
    """ Function to obtain the frames of stim that need to go """
    Fneu = np.load(Path(file_path) / "Fneu.npy")
    Fdiff = np.diff(np.nanmean(Fneu, 0))
    Fdiff[Fdiff < AnalysisConstants.height_stim_artifact * np.nanstd(Fdiff)] = 0
    Fconv = np.convolve(Fdiff, np.ones(AnalysisConstants.size_stim_frames))
    bad_frames = np.where(Fconv > 0)[0]
    return bad_frames


def prepare_ops_file_2nd_pass(file_path: Path, file_origin: Path):
    """ Function to modify the ops file before 2nd pass"""
    bad_frames = obtain_bad_frames_from_fneu(file_path)
    ops = np.load(Path(file_path) / "ops.npy", allow_pickle=True)
    dict_ops = ops.take(0)
    np.save(Path(file_path) / "old_ops.npy", dict_ops, allow_pickle=True)
    dict_ops["badframes"] = bad_frames
    dict_ops["do_registration"] = 0
    dict_ops["delete_bin"] = True
    np.save(Path(file_path) / "ops.npy", dict_ops, allow_pickle=True)
    np.save(Path(file_origin) / 'bad_frames.npy', bad_frames)


def copy_only_mat_files(folder_experiments: Path, folder_destination: Path):
    """ function to copy all the mat files without the images (to keep working offline) """
    df_sessions = ss.get_all_sessions()
    for folder_path in df_sessions['index']:
        folder_src = Path(folder_experiments) / folder_path
        folder_dst = Path(folder_destination) / folder_path
        if not Path(folder_dst).exists():
            Path(folder_dst).mkdir(parents=True, exist_ok=True)
        list_files = os.listdir(folder_src)
        for file in list_files:
            if file[-3:] == 'mat':
                shutil.copyfile(folder_src / file, folder_dst / file)



