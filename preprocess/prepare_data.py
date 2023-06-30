
__author__ = 'Nuria'

import os
import shutil
import copy

import pandas as pd
import numpy as np
import scipy.io as sio

from pathlib import Path
from typing import Tuple
from scipy import signal

import utils.util_plots as ut_plots
from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants
from preprocess import sessions as ss


def obtain_online_data(BMI_data_path: Path) -> dict:
    """ Function to retrieve the info inside BMI online """
    bmi_online = sio.loadmat(str(BMI_data_path), simplify_cells=True)
    return bmi_online


def obtain_bad_frames_from_fneu(fneu_old: np.array) -> Tuple[np.array, np.array, np.array, np.array, np.array, bool]:
    """ Function to obtain the frames of stim that need to go """
    conv_win = np.ones(AnalysisConfiguration.filter_size)
    window = int(AnalysisConfiguration.filter_size/2)
    Fmean = np.nanmean(fneu_old, 0)
    Fconv = signal.fftconvolve(Fmean, conv_win/conv_win.shape, 'valid')
    xx = np.arange(window, Fconv.shape[0]+window)
    poly = np.polyfit(xx, Fconv, 1)
    aux_f = np.zeros(Fmean.shape[0])
    aux_f[:window] = np.polyval(poly, np.arange(window))
    aux_f[Fconv.shape[0]+window-1:] = np.polyval(poly, np.arange(Fconv.shape[0]-window,Fconv.shape[0]))
    aux_f[window:Fconv.shape[0]+window] = Fconv
    F_denoised = Fmean-aux_f
    F_denoised[F_denoised < AnalysisConstants.height_stim_artifact * np.nanstd(F_denoised)] = 0
    bad_frames_index = np.where(F_denoised > 0)[0]
    diff_bad_frames = np.diff(bad_frames_index)
    missing_bad_frames = np.where(diff_bad_frames == 2)[0]
    for mbf in missing_bad_frames:
        bad_frames_index = np.append(bad_frames_index, bad_frames_index[mbf]+1)
        F_denoised[bad_frames_index[mbf]+1] = np.nanmean([F_denoised[bad_frames_index[mbf]], F_denoised[bad_frames_index[mbf]+2]])
    bad_frames_index.sort()
    frames_include = np.where(F_denoised == 0)[0]
    bad_frames_bool = F_denoised.astype(bool)
    stim_time, stim_time_bool = obtain_stim_time(bad_frames_bool)
    if np.sum(stim_time<AnalysisConstants.calibration_frames) > 0:
        sanity_check = True
    else:
        sanity_check = False
    return bad_frames_index, bad_frames_bool, frames_include, stim_time, stim_time_bool, sanity_check


def prepare_ops_1st_pass(default_path: Path, ops_path: Path) -> dict:
    """ Function to modify the default ops file before 1st pass"""
    aux_ops = np.load(Path(default_path) / "default_ops.npy", allow_pickle=True)
    ops = aux_ops.take(0)
    ops['delete_bin'] = True
    ops['move_bin'] = False
    ops['keep_movie_raw'] = False
    ops['anatomical_only'] = 2
    np.save(ops_path, ops, allow_pickle=True)
    return ops


def prepare_ops_behav_pass(default_path: Path, ops_path: Path) -> dict:
    """ Function to modify the default ops file before 1st pass"""
    aux_ops = np.load(Path(default_path) / "default_ops.npy", allow_pickle=True)
    ops = aux_ops.take(0)
    ops['delete_bin'] = True
    ops['move_bin'] = False
    ops['keep_movie_raw'] = False
    ops['anatomical_only'] = 0
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


def save_neurons_post_process(folder_save: Path, exp_info: pd.Series, E1: list, E2: list, exclude: list, added_neurons: list):
    """    Function to save the number of the direct neurons,
        actually this function is run during the manual sanity check of the raw data"""

    exclude = []
    added_neurons = []

    folder_suite2p = folder_save / exp_info['session_path'] / 'suite2p' / 'plane0'
    # save direct_neurons
    E1.sort()
    E2.sort()
    exclude.sort()
    added_neurons.sort()
    direct_neurons = {'E1': E1, 'E2': E2, 'exclude': exclude, 'added_neurons': added_neurons}
    np.save(Path(folder_suite2p) / "direct_neurons.npy", direct_neurons, allow_pickle=True)


def obtain_dffs(folder_suite2p: Path, smooth: bool = True) -> np.array:
    """ function to obtain the dffs based on F and Fneu """
    Fneu = np.load(Path(folder_suite2p) / "Fneu.npy")
    F_raw = np.load(Path(folder_suite2p) / "F.npy")
    dff = np.full(Fneu.shape, np.nan)
    for neuron in np.arange(dff.shape[0]):
        if smooth:
            smooth_filt = np.ones(AnalysisConstants.dff_win) / AnalysisConstants.dff_win
            aux = np.convolve((F_raw[neuron, :] - Fneu[neuron, :]) / np.nanmean(Fneu[neuron, :]), smooth_filt, 'valid')
            # dff during BMI is calculate fromt the previous dff_win frames, so it is shifted as below
            dff[neuron, AnalysisConstants.dff_win - 1:] = aux
        else:
            dff[neuron, :] = (F_raw[neuron, :] - Fneu[neuron, :]) / np.nanmean(Fneu[neuron, :])
    return dff


def obtain_stim_time(bad_frames_bool: np.array) -> Tuple[np.array, np.array]:
    """ function that reports the time of stim (by returning the first frame of each stim) """
    stim_time = np.insert(np.diff(bad_frames_bool.astype(int)), 0, 0)
    stim_time[stim_time < 1] = 0
    return np.where(stim_time)[0], stim_time.astype(bool)


def refine_classifier(folder_suite2p: Path):
    """ function to refine the suite2p classifier """
    neurons = np.load(Path(folder_suite2p) / "stat.npy", allow_pickle=True)
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    is_cell_new = copy.deepcopy(is_cell)
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons_info = aux_dn.take(0)
    direct_neurons = direct_neurons_info["E1"] + direct_neurons_info["E2"]
    direct_neurons.sort()
    for nn, neuron in enumerate(neurons):
        if neuron['skew'] > 10 or neuron['skew'] < 0.4 or neuron['compact'] > 1.4 or \
                neuron['footprint'] == 0 or neuron['footprint'] == 3 or neuron['npix'] < 80:
            is_cell_new[nn, :] = [0, 0]
    for dn in direct_neurons:
        is_cell_new[dn, :] = [1, 1]
    np.save(Path(folder_suite2p) / "iscell_old.npy", is_cell)
    np.save(Path(folder_suite2p) / "iscell.npy", is_cell_new)


def sanity_checks(folder_suite2p: Path, folder_fneu_old: Path, folder_process_plots: Path, stim_flag: bool = True) -> list:
    """ function to check the post_process """
    check_session = []
    # obtain the bad frames and stims
    if stim_flag:
        fneu_old = np.load(Path(folder_fneu_old) / "Fneu.npy")
        bad_frames_index, bad_frames_bool, _, stim_time, stim_time_bool, sanity_bad_frames = \
            obtain_bad_frames_from_fneu(fneu_old)
        ut_plots.easy_plot(np.nanmean(fneu_old, 0), folder_plots=folder_process_plots, var_sig='fneu_mean')
        if sanity_bad_frames:
            check_session.append('bad_frames')
        bad_frames_dict = {'bad_frames_index': bad_frames_index, 'bad_frames_bool': bad_frames_bool}
        stim_time_dict = {'stim_time': stim_time, 'stim_time_bool': stim_time_bool}
        np.save(Path(folder_suite2p) / "bad_frames_dict.npy", bad_frames_dict, allow_pickle=True)
        np.save(Path(folder_suite2p) / "stim_time_dict.npy", stim_time_dict, allow_pickle=True)

    # obtain the position of the neurons and plot it
    ops_after = np.load(Path(folder_suite2p) / "ops.npy", allow_pickle=True)
    ops_after = ops_after.take(0)
    neurons = np.load(Path(folder_suite2p) / "stat.npy", allow_pickle=True)
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")

    G = ops_after["meanImg"]
    R = np.zeros((512, 512))
    B = np.zeros((512, 512))
    Rbad = np.zeros((512, 512))
    Bbad = np.zeros((512, 512))
    for nn, neuron in enumerate(neurons):
        if bool(is_cell[nn, 0]):
            if nn in ensemble:
                for pix in np.arange(neuron["xpix"].shape[0]):
                    B[neuron["ypix"][pix], neuron["xpix"][pix]] = 1
            else:
                for pix in np.arange(neuron["xpix"].shape[0]):
                    R[neuron["ypix"][pix], neuron["xpix"][pix]] = 1
        else:
            Bbad[int(neuron["med"][0]), int(neuron["med"][1])] = 1
            for pix in np.arange(neuron["xpix"].shape[0]):
                Rbad[neuron["ypix"][pix], neuron["xpix"][pix]] = 1

    RGB = np.stack((R * 255, G / G.max() * 255, B * 255), axis=2)
    RGBbad = np.stack((Rbad * 255, G / G.max() * 255, Bbad * 255), axis=2)
    ut_plots.easy_imshow(RGB, folder_process_plots, 'neurons_location')
    ut_plots.easy_imshow(RGBbad, folder_plots=folder_process_plots, var_sig='bad_neurons_location')

    return check_session


def obtain_synchrony_stim(folder_suite2p: Path):
    """ function to find correlation between online cursor and posthoc cursor"""
    bad_frames_dict = np.load(folder_suite2p / "bad_frames_dict.npy", allow_pickle=True)
    dff = obtain_dffs(folder_suite2p, smooth=True)
    stim_time_pp, _ = obtain_stim_time(bad_frames_dict.take(0)['bad_frames_bool'])


def create_time_locked_array(arr: np.array, stim_index: np.array):
    """ function to create the time locked array of an initial array"""
    num_frames = int(AnalysisConfiguration.time_lock_seconds * AnalysisConstants.framerate)

    # Create an empty array to store the time-locked dff values
    arr_time_locked = np.zeros((arr.shape[0], len(stim_index), num_frames * 2 + 1))

    # Iterate over each index in stim_time
    for ii, index in enumerate(stim_index):
        # Extract the corresponding frames from dff
        start_frame = index - num_frames
        end_frame = index + num_frames + 1
        arr_time_locked[:, ii, :] = arr[:, start_frame:end_frame]
    return arr_time_locked


def move_file_to_old_folder(folder_name):
    """ moves fneu to fneu_old folder after first pass
     CAUTION!!! RUN ONLY AFTER 1st PASS"""
    for root, dirs, files in os.walk(folder_name):
        if 'suite2p' in dirs:
            suite2p_dirs = [dir_name for dir_name in dirs if dir_name == 'suite2p']
            for suite2p_dir in suite2p_dirs:
                suite2p_path = os.path.join(root, suite2p_dir)

                # Check if 'plane0' subfolder exists
                plane0_path = os.path.join(suite2p_path, 'plane0')
                if os.path.exists(plane0_path):
                    # Check if 'Fneu.npy' file exists
                    fneu_file_path = os.path.join(plane0_path, 'Fneu.npy')
                    if os.path.exists(fneu_file_path):
                        # Create 'fneu_old' subfolder if it doesn't exist
                        fneu_old_path = os.path.join(suite2p_path, 'fneu_old')
                        if not os.path.exists(fneu_old_path):
                            os.makedirs(fneu_old_path)

                        # Move 'Fneu.npy' to 'fneu_old' subfolder
                        shutil.move(fneu_file_path, fneu_old_path)

                        print(f"File moved successfully: {fneu_file_path}")

    print("File search and move completed.")


def kk_for_now(folder_suite2p: Path, BMI_data_path: Path):
    direct_neurons_pp = np.load(folder_suite2p / "direct_neurons.npy", allow_pickle=True)
    bad_frames_dict = np.load(folder_suite2p / "bad_frames_dict.npy", allow_pickle=True)
    direct_neurons_pp = direct_neurons_pp.take(0)
    dff = obtain_dffs(folder_suite2p, smooth=True)
    dff_direct = dff[direct_neurons_pp["E1"] + direct_neurons_pp['E2'], :]

    # load cursor online and other online data
    bmi_online = obtain_online_data(BMI_data_path)
    decoder = bmi_online['bData']['decoder']
    stim_time_online = bmi_online['data']['selfDRstim'] + bmi_online['data']['randomDRstim']

    stim_time_pp, _ = obtain_stim_time(bad_frames_dict.take(0)['bad_frames_bool'])

    cursor_online = bmi_online['data']['cursor']
    cursor_online_nonnan = cursor_online[~np.isnan(cursor_online)]

    # obtain cursor postprocessing
    cursor_pp = np.sum(dff_direct * decoder[:, None], 0)[AnalysisConstants.calibration_frames:]


    correlation = signal.correlate(cursor_pp, cursor_online_nonnan)
    lag_array = signal.correlation_lags(cursor_pp.size, cursor_online_nonnan.size)
    lag = lag_array[np.where(correlation == np.min(correlation))[0][0]]
    cursor_online_centered = np.full(cursor_pp.shape, np.nan)
    cursor_online_centered[-cursor_online_nonnan.size-2700:-2700] = cursor_online_nonnan

















