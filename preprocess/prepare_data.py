
__author__ = 'Nuria'

import os
import shutil
import copy
import math

import pandas as pd
import numpy as np
import scipy.io as sio

from pathlib import Path
from typing import Tuple
from scipy import signal

import utils.util_plots as ut_plots
import utils.utils_analysis as ut
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
    stim_index, stim_time_bool = obtain_stim_time(bad_frames_bool)
    if np.sum(stim_index<AnalysisConstants.calibration_frames) > 0:
        sanity_check = True
    else:
        sanity_check = False
    return bad_frames_index, bad_frames_bool, frames_include, stim_index, stim_time_bool, sanity_check


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


def obtain_dffs(folder_suite2p: Path, smooth: bool = True, filtered: bool = True) -> np.array:
    """ function to obtain the dffs based on F and Fneu """
    Fneu = np.load(Path(folder_suite2p) / "Fneu.npy")
    F_raw = np.load(Path(folder_suite2p) / "F.npy")
    dff = np.full(Fneu.shape, np.nan)
    for neuron in np.arange(dff.shape[0]):
        if smooth:
            smooth_filt = np.ones(AnalysisConstants.dff_win) / AnalysisConstants.dff_win
            aux = np.convolve((F_raw[neuron, :] - Fneu[neuron, :]) / np.nanmean(Fneu[neuron, :]), smooth_filt, 'valid')
            if filtered:
                aux -= ut.low_pass_arr(aux, cutoff_frequency=0.001, order=2)
            # dff during BMI is calculate fromt the previous dff_win frames, so it is shifted as below
            dff[neuron, AnalysisConstants.dff_win - 1:] = aux
        else:
            dff[neuron, :] = (F_raw[neuron, :] - Fneu[neuron, :]) / np.nanmean(Fneu[neuron, :])
    return dff


def obtain_stim_time(bad_frames_bool: np.array) -> Tuple[np.array, np.array]:
    """ function that reports the time of stim (by returning the first frame of each stim) """
    stim_index = np.insert(np.diff(bad_frames_bool.astype(int)), 0, 0)
    stim_index[stim_index < 1] = 0
    return np.where(stim_index)[0], stim_index.astype(bool)


def refine_classifier(folder_suite2p: Path, dn_bool: bool = True):
    """ function to refine the suite2p classifier """
    neurons = np.load(Path(folder_suite2p) / "stat.npy", allow_pickle=True)
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    is_cell_new = copy.deepcopy(is_cell)
    snr_val = ut.snr_neuron(folder_suite2p)
    stable_neuron = ut.stability_neuron(folder_suite2p, init=AnalysisConstants.calibration_frames)
    for nn, neuron in enumerate(neurons):
        if neuron['skew'] > 10 or neuron['skew'] < 0.4 or neuron['compact'] > 1.4 or \
                neuron['footprint'] == 0 or neuron['footprint'] == 3 or neuron['npix'] < 80 or \
                snr_val[nn] < AnalysisConfiguration.snr_min or ~stable_neuron[nn]:
            is_cell_new[nn, :] = [0, 0]
    if dn_bool:
        aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
        direct_neurons_info = aux_dn.take(0)
        direct_neurons = direct_neurons_info["E1"] + direct_neurons_info["E2"]
        direct_neurons.sort()
        for dn in direct_neurons:
            is_cell_new[dn, :] = [1, 1]
    np.save(Path(folder_suite2p) / "iscell.npy", is_cell_new)


def sanity_checks(folder_suite2p: Path, folder_fneu_old: Path, file_path: Path, folder_process_plots: Path,
                  stim_flag: bool = True, save_flag: bool = False) -> list:
    """ function to check the post_process """
    check_session = []
    # obtain the bad frames and stims

    if stim_flag:
        bmi_online = sio.loadmat(str(file_path), simplify_cells=True)
        total_stims = bmi_online['data']['selfTarget_DR_stim_Counter'] + bmi_online['data']['sched_random_stim']
        if save_flag:
            fneu_old = np.load(Path(folder_fneu_old) / "Fneu.npy")
            bad_frames_index, bad_frames_bool, _, stim_index, stim_bool, sanity_bad_frames = \
                obtain_bad_frames_from_fneu(fneu_old)
            ut_plots.easy_plot(np.nanmean(fneu_old, 0), folder_plots=folder_process_plots, var_sig='fneu_mean',
                               vertical_array=stim_index)
            bad_frames_dict = {'bad_frames_index': bad_frames_index, 'bad_frames_bool': bad_frames_bool}
            stim_time_dict = {'stim_index': stim_index, 'stim_bool': stim_bool}
            np.save(Path(folder_suite2p) / "bad_frames_dict.npy", bad_frames_dict, allow_pickle=True)
            np.save(Path(folder_suite2p) / "stim_time_dict.npy", stim_time_dict, allow_pickle=True)
        else:
            stim_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
            stim_time_dict = stim_aux.take(0)
            stim_index = stim_time_dict['stim_index']
            sanity_bad_frames = np.sum(stim_index < AnalysisConstants.calibration_frames) > 0

        if sanity_bad_frames:
            check_session.append('bad_frames')
        elif np.sum(np.diff(stim_index) < 40) > 0:
            check_session.append('redundance')
        elif total_stims != len(stim_index):
            check_session.append('total_stims')
    else:
        if save_flag:
            fneu_old = np.load(Path(folder_suite2p) / "Fneu.npy")
            ut_plots.easy_plot(np.nanmean(fneu_old, 0), folder_plots=folder_process_plots, var_sig='fneu_mean')

    # obtain the position of the neurons and plot it
    if save_flag:
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

        RGB = np.stack((R, ut_plots.scale_array(G)/100, B), axis=2)
        RGBbad = np.stack((Rbad, ut_plots.scale_array(G)/100, Bbad), axis=2)
        ut_plots.easy_imshow(RGB, folder_process_plots, 'neurons_location')
        ut_plots.easy_imshow(RGBbad, folder_plots=folder_process_plots, var_sig='bad_neurons_location')

    return check_session


def double_check(folder_list: list, sessions_to_double_check):
    """ function to iterate over bad sessions """
    for session in sessions_to_double_check:
        mouse, _, _ = session[0].split('/')
        folder_path = Path(folder_list[ss.find_folder_path(mouse)]) / 'process' / session[0]
        folder_suite2p = folder_path / 'suite2p' / 'plane0'
        folder_fneu_old = folder_path / 'suite2p' / 'fneu_old'
        folder_process_plots = folder_path / 'suite2p' / 'plots'
        fneu_old = np.load(Path(folder_fneu_old) / "Fneu.npy")
        stim_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
        stim_time_dict = stim_aux.take(0)
        stim_index = stim_time_dict['stim_index']
        stim_bool = stim_time_dict['stim_bool']

        stim_index = stim_index[stim_index > 27000]
        stim_bool[stim_bool < 27000] = 0
        if np.sum(np.diff(stim_index)<40) > 0:
            print('this was it')
            stim_index = ut.remove_redundant(stim_index, 40)

        ut_plots.easy_plot(np.nanmean(fneu_old, 0), folder_plots=folder_process_plots, var_sig='fneu_mean',
                           vertical_array=stim_index)
        stim_time_dict = {'stim_index': stim_index, 'stim_bool': stim_bool}
        np.save(Path(folder_suite2p) / "stim_time_dict.npy", stim_time_dict, allow_pickle=True)


def find_random_stims(folder_suite2p: Path, file_path: Path) -> np.array:
    """ function to find stims from random """

    bmi_online = sio.loadmat(str(file_path), simplify_cells=True)
    stim_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
    stim_time_dict = stim_aux.take(0)
    stim_index = stim_time_dict['stim_index']

    stim_online = np.where(bmi_online['data']['randomDRstim'].astype(bool))[0]
    target_online = np.where(bmi_online['data']['selfHits'].astype(bool))[0]
    if len(stim_online) != len(stim_index):
        raise ValueError ('Check the stims are well identified, spot difference online / post process')
    closest_indexes, differences = ut.find_closest(stim_online, target_online)
    target_index = np.zeros(target_online.shape[0], dtype=int)
    for tt in np.arange(target_online.shape[0]):
        target_index[tt] = stim_index[closest_indexes[tt]] + differences[tt]
    return target_index


def save_targets(experiment_type, file_path, folder_suite2p):
    """ function to save the target_dict"""
    stim_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
    stim_time_dict = stim_aux.take(0)
    stim_index = stim_time_dict['stim_index']
    stim_bool = stim_time_dict['stim_bool']
    if experiment_type in ['D1act', 'CONTROL_LIGHT', 'NO_AUDIO']:
        target_index = stim_index
    elif experiment_type == 'DELAY':
        target_index = stim_index - int(np.round(AnalysisConstants.framerate))
    elif experiment_type == 'RANDOM':
        target_index = find_random_stims(folder_suite2p, file_path)
    else:
        return
    target_bool = np.zeros(stim_bool.shape[0], dtype=bool)
    target_bool[target_index] = 1
    target_dict = {'target_index': target_index, 'target_bool': target_bool}
    np.save(Path(folder_suite2p) / "target_time_dict.npy", target_dict, allow_pickle=True)
    return


def obtain_synchrony_stim(folder_suite2p: Path):
    """ function to find correlation between online cursor and posthoc cursor"""
    bad_frames_dict = np.load(folder_suite2p / "bad_frames_dict.npy", allow_pickle=True)
    dff = obtain_dffs(folder_suite2p, smooth=True)
    stim_time_pp, _ = obtain_stim_time(bad_frames_dict.take(0)['bad_frames_bool'])


def create_time_locked_array(arr: np.array, stim_index: np.array, num_frames: tuple) -> np.array:
    """ function to create the time locked array of an initial array"""

    # Create an empty array to store the time-locked dff values
    if len(arr.shape) > 1:
        arr_time_locked = np.zeros((arr.shape[0], len(stim_index), np.sum(num_frames)))
    else:
        arr_time_locked = np.zeros((len(stim_index), np.sum(num_frames)))

    # Iterate over each index in stim_time
    for ii, index in enumerate(stim_index):
        # Extract the corresponding frames from dff
        start_frame = index - num_frames[0]
        end_frame = index + num_frames[1]
        if len(arr.shape) > 1:
            arr_time_locked[:, ii, :] = arr[:, start_frame:end_frame]
        else:
            arr_time_locked[ii, :] = arr[start_frame:end_frame]
    return arr_time_locked


def obtain_SNR_per_neuron(folder_suite2p: Path) -> Tuple[float, float, float]:
    """ Function to calculate the SNR given the F and Fneuropil surrounding"""
    Fneu = np.load(Path(folder_suite2p) / "Fneu.npy")
    F_raw = np.load(Path(folder_suite2p) / "F.npy")
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    direct_neurons_aux = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = direct_neurons_aux.take(0)

    power_signal_all = np.nanmean(np.square(F_raw), 1)
    power_noise_all = np.nanmean(np.square(Fneu), 1)

    power_signal_dn_E1 = np.nanmean(np.square(F_raw[direct_neurons['E1'], :]), 1)
    power_noise_dn_E1 = np.nanmean(np.square(Fneu[direct_neurons['E1'], :]), 1)
    power_signal_dn_E2 = np.nanmean(np.square(F_raw[direct_neurons['E2'], :]), 1)
    power_noise_dn_E2 = np.nanmean(np.square(Fneu[direct_neurons['E2'], :]), 1)

    # Calculate the SNR
    snr_all = 10 * np.log10(power_signal_all / power_noise_all)
    snr_dn_E1 = 10 * np.log10(power_signal_dn_E1 / power_noise_dn_E1)
    snr_dn_E2 = 10 * np.log10(power_signal_dn_E2 / power_noise_dn_E2)
    return snr_all, snr_dn_E1, snr_dn_E2

def obtain_motion_per_experiment(folder_suite2p: Path, fc: int = 10) -> pd.DataFrame:
    """ Function to obtain the motion during the online experiment based on suite2p measures
    :param folder_suite2p: folder where the experiment is
    :param fc: number of frames considered before and after the target acquisition to check motion
    the BMI averages over 10 frames """
    ops = np.load(Path(folder_suite2p) / "ops.npy", allow_pickle=True)
    ops = ops.take(0)
    index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
    index_dict = index_aux.take(0)
    direct_neurons_aux = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = direct_neurons_aux.take(0)
    neurons = np.load(Path(folder_suite2p) / "stat.npy", allow_pickle=True)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    size_dn = np.full(len(ensemble), np.nan)
    for dd, dn in enumerate(ensemble):
        size_dn[dd] = neurons[dd]['radius']*2


    indices = index_dict['target_index']
    random_indices = np.sort(np.random.randint(fc + 1,
                                               AnalysisConstants.calibration_frames - 1, indices.shape[0]))

    xoff_tl = create_time_locked_array(ops['xoff'], indices, (2 * fc, fc))
    yoff_tl = create_time_locked_array(ops['yoff'], indices, (2 * fc, fc))
    xoff_rand_tl = create_time_locked_array(ops['xoff'], random_indices, (fc, 0))
    yoff_rand_tl = create_time_locked_array(ops['yoff'], random_indices, (fc, 0))
    distance_final = np.asarray([np.sqrt(np.diff(xoff_tl[:, :fc]).sum(1)**2 +
                                         np.diff(yoff_tl[:, :fc]).sum(1)**2),
                                 np.sqrt(np.diff(xoff_tl[:, fc:2*fc]).sum(1)**2 +
                                         np.diff(yoff_tl[:, fc:2*fc]).sum(1)**2),
                                 np.sqrt(np.diff(xoff_tl[:, 2*fc:]).sum(1)**2 +
                                         np.diff(yoff_tl[:, 2*fc:]).sum(1)**2),
                                 np.sqrt(np.diff(xoff_rand_tl).sum(1) ** 2 +
                                         np.diff(yoff_rand_tl).sum(1) ** 2)
                                 ])
    distance_max = np.asarray([np.sqrt(np.abs(np.diff(xoff_tl[:, :fc]))**2 +
                                         np.abs(np.diff(yoff_tl[:, :fc]))**2).max(1),
                                 np.sqrt(np.abs(np.diff(xoff_tl[:, fc:2*fc]))**2 +
                                         np.abs(np.diff(yoff_tl[:, fc:2*fc]))**2).max(1),
                                 np.sqrt(np.abs(np.diff(xoff_tl[:, 2*fc:]))**2 +
                                         np.abs(np.diff(yoff_tl[:, 2*fc:]))**2).max(1),
                                 np.sqrt(np.abs(np.diff(xoff_rand_tl))** 2 +
                                         np.abs(np.diff(yoff_rand_tl))** 2).max(1)
                                 ])
    df_final = pd.DataFrame(columns=['before', 'hit', 'reward', 'random'],
                            index=np.arange(indices.shape[0]), data=distance_final.T)
    df_final['trial'] = np.arange(indices.shape[0])
    df_final['type'] = 'final'
    df_final['size'] = size_dn.mean()
    df_max = pd.DataFrame(columns=['before', 'hit', 'reward', 'random'],
                            index=np.arange(indices.shape[0]), data=distance_max.T)
    df_max['trial'] = np.arange(indices.shape[0])
    df_max['type'] = 'max'
    df_max['size'] = size_dn.mean()
    return pd.concat([df_final, df_max]).reset_index(drop=True)

def obtain_image_corr_per_experiment(folder_suite2p: Path, fc: int = 10) -> pd.DataFrame:
    """ Function to obtain the correlation to reference during the online experiment based on suite2p measures
    :param folder_suite2p: folder where the experiment is
    :param fc: number of frames considered before and after the target acquisition to check motion
    the BMI averages over 10 frames """
    ops = np.load(Path(folder_suite2p) / "ops.npy", allow_pickle=True)
    ops = ops.take(0)
    index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
    index_dict = index_aux.take(0)
    direct_neurons_aux = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = direct_neurons_aux.take(0)
    neurons = np.load(Path(folder_suite2p) / "stat.npy", allow_pickle=True)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    dff = np.load(Path(folder_suite2p) / "dff.npy")

    indices = index_dict['target_index']
    random_indices = np.sort(np.random.randint(fc + 1,
                                               AnalysisConstants.calibration_frames - 1, indices.shape[0]))

    corr_tl = create_time_locked_array(ops['corrXY'], indices, (fc, fc))
    corr_rand_tl = create_time_locked_array(ops['corrXY'], random_indices, (fc, 0))
    dff_E1_tl = create_time_locked_array(dff[direct_neurons['E1'], :], indices, (fc, fc))
    dff_E1_rand_tl = create_time_locked_array(dff[direct_neurons['E1'], :], random_indices, (fc, 0))
    dff_E2_tl = create_time_locked_array(dff[direct_neurons['E2'], :], indices, (fc, fc))
    dff_E2_rand_tl = create_time_locked_array(dff[direct_neurons['E2'], :], random_indices, (fc, 0))
    cursor = - np.nanmean(dff[direct_neurons['E1'], :], 0) + np.nanmean(dff[direct_neurons['E2'], :], 0)
    cursor_tl = create_time_locked_array(cursor, indices, (fc, fc))
    cursor_rand_tl = create_time_locked_array(cursor, random_indices, (fc, 0))
    xoff_tl = create_time_locked_array(ops['xoff'], indices, (fc+1, fc))
    yoff_tl = create_time_locked_array(ops['yoff'], indices, (fc+1, fc))
    xoff_rand_tl = create_time_locked_array(ops['xoff'], random_indices, (fc+1, 0))
    yoff_rand_tl = create_time_locked_array(ops['yoff'], random_indices, (fc+1, 0))

    corr_image = np.asarray([np.nanmean(corr_tl[:, :fc],1), np.nanmean(corr_tl[:, fc:],1),
                             np.nanmean(corr_rand_tl,1)])
    df_corr_image = pd.DataFrame(columns=['hit', 'reward', 'random'],
                            index=np.arange(indices.shape[0]), data=corr_image.T)
    df_corr_image['trial'] = np.arange(indices.shape[0])
    df_corr_image['type'] = 'image'

    corr_cursor = np.full([indices.shape[0], 3], np.nan)
    corr_E1 = np.full([indices.shape[0], 3], np.nan)
    corr_E2 = np.full([indices.shape[0], 3], np.nan)
    for tt, trial in enumerate(indices):
        motion = np.sqrt(np.diff(xoff_tl[tt, :])**2 + np.diff(yoff_tl[tt, :])**2)
        motion_rand = np.sqrt(np.diff(xoff_rand_tl[tt, :])**2 + np.diff(yoff_rand_tl[tt, :])**2)
        corr_cursor[tt, 0] = np.corrcoef(cursor_tl[tt, :fc], motion[:fc])[0,1] ** 2
        corr_cursor[tt, 1] = np.corrcoef(cursor_tl[tt, fc:], motion[fc:])[0, 1] ** 2
        corr_cursor[tt, 2] = np.corrcoef(cursor_rand_tl[tt, :], motion_rand)[0, 1] ** 2
        corr_E1[tt, 0] = np.corrcoef(np.nanmean(dff_E1_tl[:, tt, :fc],0), motion[:fc])[0,1] ** 2
        corr_E1[tt, 1] = np.corrcoef(np.nanmean(dff_E1_tl[:, tt, fc:],0), motion[fc:])[0,1] ** 2
        corr_E1[tt, 2] = np.corrcoef(np.nanmean(dff_E1_rand_tl[:, tt, :],0), motion_rand)[0, 1] ** 2
        corr_E2[tt, 0] = np.corrcoef(np.nanmean(dff_E2_tl[:, tt, :fc], 0), motion[:fc])[0, 1] ** 2
        corr_E2[tt, 1] = np.corrcoef(np.nanmean(dff_E2_tl[:, tt, fc:], 0), motion[fc:])[0, 1] ** 2
        corr_E2[tt, 2] = np.corrcoef(np.nanmean(dff_E2_rand_tl[:, tt, :], 0), motion_rand)[0, 1] ** 2
    df_corr_cursor = pd.DataFrame(columns=['hit', 'reward', 'random'],
                                 index=np.arange(indices.shape[0]), data=corr_cursor)
    df_corr_cursor['type'] = 'cursor'
    df_corr_cursor['trial'] = np.arange(indices.shape[0])
    df_corr_E1= pd.DataFrame(columns=['hit', 'reward', 'random'],
                                  index=np.arange(indices.shape[0]), data=corr_E1)
    df_corr_E1['type'] = 'E1'
    df_corr_E1['trial'] = np.arange(indices.shape[0])
    df_corr_E2 = pd.DataFrame(columns=['hit', 'reward', 'random'],
                                  index=np.arange(indices.shape[0]), data=corr_E2)
    df_corr_E2['type'] = 'E2'
    df_corr_E2['trial'] = np.arange(indices.shape[0])
    return pd.concat([df_corr_image, df_corr_cursor, df_corr_E1, df_corr_E2]).reset_index(drop=True)


def obtain_online_time_vector(file_path: Path) -> float:
    """ function to obtain time vector when there was a hit """

    bmi_online = sio.loadmat(file_path, simplify_cells=True)
    time_vector = bmi_online['data']['timeVector']
    self_hits = bmi_online['data']['selfHits']
    time_hits = time_vector[self_hits == 1]
    return time_hits

def obtain_location_direct_neurons(folder_suite2p: Path) -> Tuple[int, int, np.ndarray]:
    """ function to obtain the location of the direct neurons and distance among them """
    direct_neurons_aux = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = direct_neurons_aux.take(0)
    neurons = np.load(Path(folder_suite2p) / "stat.npy", allow_pickle=True)
    y_E1 = np.asarray([neurons[direct_neurons['E1'][0]]['ypix'].mean(),
                       neurons[direct_neurons['E1'][1]]['ypix'].mean()])
    if len(y_E1) != 2:
        print("Warning: " + folder_suite2p)
        y_E1 = np.append(np.nan)
    x_E1 = np.asarray([neurons[direct_neurons['E1'][0]]['xpix'].mean(),
                       neurons[direct_neurons['E1'][1]]['xpix'].mean()])
    if len(x_E1) != 2:
        print("Warning: " + folder_suite2p)
        x_E1 = np.append(np.nan)
    y_E2 = np.asarray([neurons[direct_neurons['E2'][0]]['ypix'].mean(),
                       neurons[direct_neurons['E2'][1]]['ypix'].mean()])
    if len(y_E2) != 2:
        print("Warning: " + folder_suite2p)
        y_E2 = np.append(np.nan)
    x_E2 = np.asarray([neurons[direct_neurons['E2'][0]]['xpix'].mean(),
                       neurons[direct_neurons['E2'][1]]['xpix'].mean()])
    if len(x_E2) != 2:
        print("Warning: " + folder_suite2p)
        x_E2 = np.append(np.nan)
    within_distance = np.asarray([np.sqrt(np.diff(y_E1)**2 + np.diff(x_E1)**2),
                                  np.sqrt(np.diff(y_E2)**2 + np.diff(x_E2)**2)]).squeeze()
    across_distance = np.asarray([np.sqrt((y_E1 - y_E2[0])**2 + (x_E1 - x_E2[0])**2),
                                  np.sqrt((y_E1 - y_E2[1])**2 + (x_E1 - x_E2[1])**2)]).reshape(4)
    return within_distance[0], within_distance[1], across_distance

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


def obtain_correlation_traces(folder_suite2p: Path, file_path:Path, fc:int = 30):
    """ function to obtain correlation online/offline """

    f = np.load(Path(folder_suite2p) / "f.npy")
    index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
    index_dict = index_aux.take(0)
    indices = index_dict['target_index']
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    pre_f_dn = f[ensemble, :]
    bmi_online = sio.loadmat(file_path, simplify_cells=True)
    pre_f_online = bmi_online['data']['bmiAct']
    f_dn = np.full(pre_f_dn.shape, np.nan)
    f_dn_online = np.full(pre_f_online.shape, np.nan)
    for neuron in np.arange(f_dn.shape[0]):
        smooth_filt = np.ones(AnalysisConstants.dff_win) / AnalysisConstants.dff_win
        f_dn[neuron, AnalysisConstants.dff_win - 1:] = np.convolve(pre_f_dn[neuron,:], smooth_filt, 'valid')
        f_dn_online[neuron, AnalysisConstants.dff_win - 1:] = np.convolve(pre_f_online[neuron, :], smooth_filt, 'valid')
    f_tl = pp.create_time_locked_array(f_dn, indices, (2*fc, fc))
    f_tl_online = pp.create_time_locked_array(f_dn_online, np.where(bmi_online['data']['selfHits'])[0], (2*fc, fc))

    r2_E1 = np.full([len(indices), 3], np.nan)
    r2_E2 = np.full([len(indices), 3], np.nan)
    for ii, index in enumerate(indices):
        r2_E1[ii, 0] = np.corrcoef(np.nansum(f_tl[:2, ii, :fc],0), np.nansum(f_tl_online[:2, ii, :fc],0))[0, 1] ** 2
        r2_E2[ii, 0] = np.corrcoef(np.nansum(f_tl[2:, ii, :fc], 0), np.nansum(f_tl_online[2:, ii, :fc], 0))[0, 1] ** 2
        r2_E1[ii, 1] = np.corrcoef(np.nansum(f_tl[:2, ii, fc:2*fc],0), np.nansum(f_tl_online[:2, ii, fc:2*fc],0))[0, 1] ** 2
        r2_E2[ii, 1] = np.corrcoef(np.nansum(f_tl[2:, ii, fc:2*fc], 0), np.nansum(f_tl_online[2:, ii, fc:2*fc], 0))[0, 1] ** 2
        r2_E1[ii, 2] = np.corrcoef(np.nansum(f_tl[:2, ii, 2*fc:],0), np.nansum(f_tl_online[:2, ii, 2*fc:],0))[0, 1] ** 2
        r2_E2[ii, 2] = np.corrcoef(np.nansum(f_tl[2:, ii, 2*fc:], 0), np.nansum(f_tl_online[2:, ii, 2*fc:], 0))[0, 1] ** 2

    return np.nanmean(r2_E1, 0), np.nanmean(r2_E2, 0)


def online_comparison(f_E1: np.array, f_E2: np.array) -> pd.DataFrame:
    """ Function to study differences of E1 and E2 """

    # Initialize an empty DataFrame with the appropriate columns
    df_results = pd.DataFrame(columns=['std', 'pks', 'ampl', 'type'])

    # Combined loop for both E1 and E2
    for nn in range(f_E1.shape[0]):
        # Process E1
        aux_E1 = signal.savgol_filter(f_E1[nn, ~np.isnan(f_E1[nn, :])], window_length=101, polyorder=2)
        peaks_E1, _ = signal.find_peaks(aux_E1, height=None, distance=100, prominence=50)

        std_E1 = np.nanstd(f_E1[nn, :])
        E1_pks = peaks_E1.shape[0]
        f_E1_ampl = aux_E1[peaks_E1].mean() / aux_E1.mean() if peaks_E1.size > 0 else np.nan

        # Append the result to the DataFrame
        df_results.loc[nn] = [std_E1, E1_pks, f_E1_ampl, 'E1']

    for nn in range(f_E2.shape[0]):
        # Process E2
        aux_E2 = signal.savgol_filter(f_E2[nn, ~np.isnan(f_E2[nn, :])], window_length=101, polyorder=2)
        peaks_E2, _ = signal.find_peaks(aux_E2, height=None, distance=100, prominence=50)

        std_E2 = np.nanstd(f_E2[nn, :])
        E2_pks = peaks_E2.shape[0]
        f_E2_ampl = aux_E2[peaks_E2].mean() / aux_E2.mean() if peaks_E2.size > 0 else np.nan

        # Append the result to the DataFrame
        df_results.loc[nn + f_E1.shape[0]] = [std_E2, E2_pks, f_E2_ampl, 'E2']

    return df_results


def comparison_neurons_trials(folder_suite2p: Path, fc:int = 150) -> pd.DataFrame:
    """ Function to study differences of E1 and E2 """
    dff = np.load(Path(folder_suite2p) / "dff.npy")
    index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
    index_dict = index_aux.take(0)
    indices = index_dict['target_index']
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    dff_E1 = dff[direct_neurons['E1'], :]
    dff_E2 = dff[direct_neurons['E2'], :]
    dff_E1_tl = pp.create_time_locked_array(dff_E1, indices, (fc, 0))
    dff_E2_tl = pp.create_time_locked_array(dff_E2, indices, (fc, 0))


    # Initialize an empty DataFrame with the appropriate columns
    df_results = pd.DataFrame(columns=['std', 'pks', 'ampl', 'type', 'trial'])

    # Combined loop for both E1 and E2
    ii=0
    for tt, index in enumerate(indices):
        for nn in range(dff_E1.shape[0]):
            # Process E1
            aux_E1 = signal.savgol_filter(dff_E1_tl[nn, tt, ~np.isnan(dff_E1_tl[nn, tt, :])], window_length=11, polyorder=2)
            peaks_E1, _ = signal.find_peaks(aux_E1, height=None, distance=10, prominence=0.1)

            std_E1 = np.nanstd(dff_E1_tl[nn, tt, :])
            E1_pks = peaks_E1.shape[0]
            E1_amp = aux_E1[peaks_E1].mean() / np.abs(aux_E1.mean()) if peaks_E1.size > 0 else np.nan

            # Append the result to the DataFrame
            df_results.loc[ii] = [std_E1, E1_pks, E1_amp, 'E1', tt]
            ii += 1

        for nn in range(dff_E2.shape[0]):
            # Process E2
            aux_E2 = signal.savgol_filter(dff_E2_tl[nn, tt, ~np.isnan(dff_E2_tl[nn, tt, :])], window_length=11, polyorder=2)
            peaks_E2, _ = signal.find_peaks(aux_E2, height=None, distance=10, prominence=0.1)

            std_E2 = np.nanstd(dff_E2_tl[nn, tt, :])
            E2_pks = peaks_E2.shape[0]
            E2_amp = aux_E2[peaks_E2].mean() / np.abs(aux_E2.mean()) if peaks_E2.size > 0 else np.nan

            # Append the result to the DataFrame
            df_results.loc[ii] = [std_E2, E2_pks, E2_amp, 'E2', tt]
            ii += 1

    return df_results















