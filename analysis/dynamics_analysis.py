__author__ = 'Nuria'

import copy
import collections
import numpy as np
import pandas as pd
from pathlib import Path

from typing import Tuple, Optional
from sklearn.linear_model import RidgeCV

import utils.utils_analysis as ut
import analysis.dynamics_mat as dm
import preprocess.prepare_data as pp
from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants


def obtain_manifold(folder_suite2p: Path, expected_length: int = 54000):
    """ function to obtain the measures for manifold info """
    spks = np.load(Path(folder_suite2p) / "spks.npy")
    if spks.shape[1] != expected_length:
        raise ValueError(f'The length of the experiment should be {expected_length} and is {spks.shape[1]}')
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    spks_av = ut.sum_array_samples(spks[is_cell[:, 0].astype(bool), :], 1, AnalysisConfiguration.FA_spks_av_win)
    if spks_av.shape[0] > AnalysisConfiguration.FA_n_neu:
        n_neu = int(AnalysisConfiguration.FA_n_neu)
        n_iter = AnalysisConfiguration.FA_n_iter
    else:
        n_neu = int(spks_av.shape[0])
        n_iter = 1
    len_mat = int(spks_av.shape[1] / 2)

    dim_array = np.full((n_iter, 3), np.nan)
    SOT_array = np.full((n_iter, 3), np.nan)
    VAF_array = np.full((n_iter, 3), np.nan)
    for iter in np.arange(n_iter):
        selected_neurons = np.random.choice(np.arange(spks_av.shape[0]), size=n_neu, replace=False)
        spks_a = spks_av[selected_neurons, :len_mat]
        spks_b = spks_av[selected_neurons, len_mat:]
        dim_all, _, _ = dm.obtain_FA(spks_av[selected_neurons, :], VAF=AnalysisConfiguration.FA_VAF)
        dim_a, _, _ = dm.obtain_FA(spks_a, VAF=AnalysisConfiguration.FA_VAF)
        dim_b, _, _ = dm.obtain_FA(spks_b, VAF=AnalysisConfiguration.FA_VAF)
        _, SOT_all, VAF_all = dm.obtain_FA(spks_av[selected_neurons, :],
                                           n_components=AnalysisConfiguration.FA_components)
        _, SOT_a, VAF_a = dm.obtain_FA(spks_a, n_components=AnalysisConfiguration.FA_components)
        _, SOT_b, VAF_b = dm.obtain_FA(spks_b, n_components=AnalysisConfiguration.FA_components)
        dim_array[iter, :] = [dim_all, dim_a, dim_b]
        SOT_array[iter, :] = [SOT_all, SOT_a, SOT_b]
        VAF_array[iter, :] = [VAF_all, VAF_a, VAF_b]
    return dim_array.mean(0), SOT_array.mean(0), VAF_array.mean(0)


def obtain_manifold_time(folder_suite2p: Path, expected_length: int = 54000, time_points: int = 30):
    """ function to obtain the measures for manifold info """
    spks = np.load(Path(folder_suite2p) / "spks.npy")
    if spks.shape[1] != expected_length:
        raise ValueError(f'The length of the experiment should be {expected_length} and is {spks.shape[1]}')
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    spks_av = spks[is_cell[:, 0].astype(bool), :]
    if spks_av.shape[0] > AnalysisConfiguration.FA_n_neu:
        n_neu = int(AnalysisConfiguration.FA_n_neu)
        n_iter = AnalysisConfiguration.FA_n_iter
    else:
        n_neu = int(spks_av.shape[0])
        n_iter = 1
    frame_array = np.linspace(0, spks_av.shape[1], time_points + 1, dtype=int)

    dim_array = np.full((n_iter, len(frame_array)), np.nan)
    SOT_array = np.full((n_iter, len(frame_array)), np.nan)
    VAF_array = np.full((n_iter, len(frame_array)), np.nan)

    for iter in np.arange(n_iter):
        for tt, frame in enumerate(frame_array[:-1]):
            selected_neurons = np.random.choice(np.arange(spks_av.shape[0]), size=n_neu, replace=False)
            spks = spks_av[selected_neurons, frame:frame_array[tt + 1]]
            dim, _, _ = dm.obtain_FA(spks, VAF=AnalysisConfiguration.FA_VAF)
            _, SOT, VAF = dm.obtain_FA(spks, n_components=AnalysisConfiguration.FA_components)
            dim_array[iter, tt] = dim
            SOT_array[iter, tt] = SOT
            VAF_array[iter, tt] = VAF
    return dim_array.mean(0), SOT_array.mean(0), VAF_array.mean(0)


def obtain_SOT_over_time(folder_suite2p: Path, tos: str = 'stim') \
        -> Tuple[np.array, np.array, np.array, np.array]:
    """ function to obtain the SOT over time for direct and indirect neurons with neurons x time """
    if tos == 'stim':
        index_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['stim_index']
    elif tos == 'target':
        index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['target_index']
    else:
        indices = np.sort(np.random.randint(AnalysisConfiguration.FA_event_frames + 1,
                                            AnalysisConstants.calibration_frames - AnalysisConfiguration.FA_rew_frames - 1,
                                            size=AnalysisConfiguration.FA_len_SOT))
    spks_dff = np.load(Path(folder_suite2p) / "spks_dff.npy")
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    indirect_neurons = copy.deepcopy(is_cell)
    indirect_neurons[ensemble, :] = [0, 0]
    indirect_neurons[direct_neurons['exclude'], :] = [0, 0]
    indices = indices[np.where(np.logical_and(np.logical_and(indices +
                                                             AnalysisConfiguration.FA_rew_frames < spks_dff.shape[1],
                                                             indices > AnalysisConfiguration.FA_event_frames),
                                              np.isin(indices, np.where(~np.isnan(spks_dff.mean(0)))[0])))[0]]
    SOT_stim_dn, SOT_stim_in, SOT_stim_all, DIM_stim_all = obtain_SOT_event(indices, ensemble, indirect_neurons,
                                                                            spks_dff)
    return SOT_stim_dn, SOT_stim_in, SOT_stim_all, DIM_stim_all


def obtain_SOT_over_all_lines(folder_suite2p: Path, tos: str = 'stim') -> Tuple[np.array, np.array]:
    """ function to obtain the SOT over time for direct and indirect neurons with stimxtime"""
    if tos == 'stim':
        index_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['stim_index']
    elif tos == 'target':
        index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['target_index']
    else:
        indices = np.sort(np.random.randint(AnalysisConfiguration.FA_event_frames + 1,
                                            AnalysisConstants.calibration_frames - AnalysisConfiguration.FA_rew_frames - 1,
                                            size=AnalysisConfiguration.FA_len_SOT))
    spks_dff = np.load(Path(folder_suite2p) / "spks_dff.npy")
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    indirect_neurons = copy.deepcopy(is_cell)
    indirect_neurons[ensemble, :] = [0, 0]
    indirect_neurons[direct_neurons['exclude'], :] = [0, 0]
    indices = indices[np.where(np.logical_and(np.logical_and(indices +
                                                             AnalysisConfiguration.FA_rew_frames < spks_dff.shape[1],
                                                             indices > AnalysisConfiguration.FA_event_frames),
                                              np.isin(indices, np.where(~np.isnan(spks_dff.mean(0)))[0])))[0]]
    SOT_ln_dn, SOT_ln_in = obtain_SOT_line(indices, ensemble, indirect_neurons, spks_dff)
    return SOT_ln_dn, SOT_ln_in


def obtain_SOT_windows(folder_suite2p: Path, win: Tuple, remove_target: bool = True) -> Tuple[float, float]:

    """
    Function that obtains the SOT for a given window
    :param folder_suite2p: folder where the data is stored
    :param win: the window of frames where to calculate SOT
    :return:
    """

    spks_dff = np.load(Path(folder_suite2p) / "spks_dff.npy")
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    indirect_neurons = copy.deepcopy(is_cell)
    indirect_neurons[ensemble, :] = [0, 0]
    indirect_neurons[direct_neurons['exclude'], :] = [0, 0]

    frames = np.arange(np.max([win[0], 0]), np.min([win[1], spks_dff.shape[1]]), dtype=int)

    if remove_target:
        index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['target_index']
        frames = ut.remove_matching_index(frames, indices, AnalysisConfiguration.FA_event_frames)

    SOT_dn, SOT_in, DIM_dn, DIM_in = obtain_SOT_all(ensemble, indirect_neurons, spks_dff[:, frames])
    return SOT_dn, SOT_in, DIM_dn, DIM_in


def obtain_SOT_all(ensemble: np.array, indirect_neurons: np.array, spks_dff: np.array) \
        -> Tuple[float, float]:
    """ Function to obtain the SOT over the whole spks_dff array for direct and indirect neurons

    :param ensemble: ensemble neurons
    :param indirect_neurons: indirect neurons
    :param spks_dff: array of spikes to analyze
    :return: the SOT of the direct neurons, the SOT of the indirect neurons and the dimension of the manifold
    """

    spks_dn = spks_dff[ensemble, :]
    spks_dn = spks_dn[:, ~ np.isnan(np.sum(spks_dn, 0))]
    spks_in = spks_dff[indirect_neurons[:, 0].astype(bool), :]
    spks_in = spks_in[:, ~ np.isnan(np.sum(spks_in, 0))]
    SOT_in = np.full(AnalysisConfiguration.FA_n_iter, np.nan)
    if spks_dn.shape[0] > 0:
        DIM_dn, _, _ = dm.obtain_FA(spks_dn, VAF=AnalysisConfiguration.FA_VAF)
        _, SOT_dn, _ = dm.obtain_FA(spks_dn, 4)
    else:
        DIM_dn = np.nan
        SOT_dn = np.nan
    if spks_in.shape[0] > 0:
        DIM_in, _, _ = dm.obtain_FA(spks_in, VAF=AnalysisConfiguration.FA_VAF)
        _, SOT_in, _ = dm.obtain_FA(spks_in, AnalysisConfiguration.FA_components)
    else:
        DIM_in = np.nan
        SOT_in = np.nan

    return SOT_dn, SOT_in, DIM_dn, DIM_in


def obtain_SOT_event(indices: np.array, ensemble: np.array, indirect_neurons: np.array, spks_dff: np.array) \
        -> Tuple[np.array, np.array, np.array, np.array]:
    """ function to obtain the SOT for direct and indirect neurons """

    if len(indices) == 0:
        return np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan)

    spks_tl = pp.create_time_locked_array(spks_dff, indices, (AnalysisConfiguration.FA_event_frames,
                                                              AnalysisConfiguration.FA_rew_frames))
    spks_dn = spks_tl[ensemble, :, :]
    spks_in = spks_tl[indirect_neurons[:, 0].astype(bool), :, :]

    SOT_stim_dn = np.full(spks_tl.shape[1], np.nan)
    SOT_stim_in = np.full((spks_tl.shape[1], AnalysisConfiguration.FA_n_iter), np.nan)
    SOT_stim_all = np.full(spks_tl.shape[1], np.nan)
    DIM_stim_all = np.full(spks_tl.shape[1], np.nan)
    for stim in np.arange(spks_tl.shape[1]):
        if np.sum(np.isnan(np.sum(spks_dn[:, stim, :], 0))) > 0:
            continue
        _, SOT_dn, _ = dm.obtain_FA(spks_dn[:, stim, :], 2)
        SOT_stim_dn[stim] = SOT_dn
        _, SOT_all, _ = dm.obtain_FA(spks_tl[:, stim, :], 4)
        DIM_all, _, _ = dm.obtain_FA(spks_tl[:, stim, :], VAF=0.9)
        SOT_stim_all[stim] = SOT_all
        DIM_stim_all[stim] = DIM_all
        if spks_in.shape[0] > len(ensemble):
            for iter in np.arange(AnalysisConfiguration.FA_n_iter):
                selected_neurons = np.random.choice(np.arange(spks_in.shape[0]), size=len(ensemble), replace=False)
                if np.sum(np.isnan(np.sum(spks_in[selected_neurons, stim, :], 0))) > 0:
                    continue
                _, SOT_in, _ = dm.obtain_FA(spks_in[selected_neurons, stim, :], 2)
                SOT_stim_in[stim, iter] = SOT_in

    return SOT_stim_dn, SOT_stim_in.mean(1), SOT_stim_all, DIM_stim_all


def obtain_SOT_line(indices: np.array, ensemble: np.array, indirect_neurons: np.array, spks_dff: np.array) \
        -> Tuple[np.array, np.array]:
    """ function to obtain the SOT for direct and indirect neurons """

    if len(indices) < AnalysisConfiguration.FA_stim_win:
        return np.full(1, np.nan), np.full(1, np.nan)
    spks_tl = pp.create_time_locked_array(spks_dff, indices, (AnalysisConfiguration.FA_event_frames,
                                                              AnalysisConfiguration.FA_rew_frames))
    spks_dn = spks_tl[ensemble, :, :]
    spks_in = spks_tl[indirect_neurons[:, 0].astype(bool), :, :]
    win = int(np.ceil(AnalysisConfiguration.FA_stim_win / 2))
    SOT_stim_dn = np.full(spks_tl.shape[1]-2*win, np.nan)
    SOT_stim_in = np.full((spks_tl.shape[1]-2*win, AnalysisConfiguration.FA_n_iter), np.nan)
    spks_aux = spks_dn.transpose(1, 0, 2).reshape((spks_dn.shape[1], -1))
    for stim in np.arange(win, spks_tl.shape[1] - win):
        if np.sum(np.isnan(spks_aux[stim - win: stim + win, :])) > 0:
            continue
        _, SOT_dn, _ = dm.obtain_FA(spks_aux[stim - win: stim + win, :], 2)
        SOT_stim_dn[stim - win] = SOT_dn
        if spks_in.shape[0] > len(ensemble):
            for iter in np.arange(AnalysisConfiguration.FA_n_iter):
                selected_neurons = np.random.choice(np.arange(spks_in.shape[0]), size=len(ensemble), replace=False)
                spks_in_aux = spks_in[selected_neurons, stim - win: stim + win, :]
                if np.sum(np.isnan(spks_in_aux)) > 0:
                    continue
                _, SOT_in, _ = dm.obtain_FA(spks_in_aux.transpose(1, 0, 2).reshape((spks_in_aux.shape[1], -1)), 2)
                SOT_stim_in[stim - win, iter] = SOT_in

    return SOT_stim_dn, SOT_stim_in.mean(1)


def obtain_engagement_event(folder_suite2p: Path, tos: str = 'stim'):
    """ function to obtain the engagement of indirect neurons to the cursor"""
    if tos == 'stim':
        index_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['stim_index']
    elif tos == 'target':
        index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['target_index']
    else:
        indices = np.sort(np.random.randint(AnalysisConfiguration.FA_event_frames + 1,
                                            AnalysisConstants.calibration_frames - AnalysisConfiguration.FA_event_frames - 1,
                                            size=AnalysisConfiguration.FA_len_SOT))
    dff = pp.obtain_dffs(folder_suite2p, smooth=True)
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    indirect_neurons = copy.deepcopy(is_cell)
    indirect_neurons[ensemble, :] = [0, 0]
    indirect_neurons[direct_neurons['exclude'], :] = [0, 0]
    indices = indices[np.where(np.logical_and(np.logical_and(indices +
                                                             AnalysisConfiguration.FA_rew_frames < dff.shape[1],
                                                             indices > AnalysisConfiguration.eng_event_frames),
                                              np.isin(indices, np.where(~np.isnan(dff.mean(0)))[0])))[0]]

    if len(indices) == 0:
        return np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan)
    dff_tl = pp.create_time_locked_array(dff, indices, (AnalysisConfiguration.eng_event_frames,
                                                        AnalysisConfiguration.FA_rew_frames))
    dff_dn_cursor = dff[ensemble, :]
    dff_dn = dff_tl[ensemble, :, :]
    dff_in = dff_tl[indirect_neurons[:, 0].astype(bool), :, :]
    cursor = - np.nanmean(dff_dn_cursor[:2, :], 0) + np.nanmean(dff_dn_cursor[2:, :], 0)
    cursor_tl = pp.create_time_locked_array(cursor, indices, (AnalysisConfiguration.eng_event_frames,
                                                              AnalysisConfiguration.FA_rew_frames))
    r2_l = np.full(dff_tl.shape[1], np.nan)
    r2_l2 = np.full(dff_tl.shape[1], np.nan)
    r2_rcv = np.full(dff_tl.shape[1], np.nan)
    r2_dff_rcv = np.full(dff_tl.shape[1], np.nan)
    if dff_in.shape[0] >= 2:
        for stim in np.arange(dff_tl.shape[1]):
            if np.sum(np.isnan(dff_in[:, stim, :])) > 0:
                continue
            latents = dm.obtain_latent(dff_dn[:, stim, :])
            r = RidgeCV(5).fit(dff_in[:, stim, :].T, latents)
            r2_l[stim] = r.score(dff_in[:, stim, :].T, latents)
            latents = dm.obtain_latent(dff_dn[:, stim, :], 2)
            r = RidgeCV(5).fit(dff_in[:, stim, :].T, latents)
            r2_l2[stim] = r.score(dff_in[:, stim, :].T, latents)
            latents = dm.obtain_latent(dff_in[:, stim, :], 2)
            r = RidgeCV(5).fit(latents, cursor_tl[stim, :])
            r2_rcv[stim] = r.score(latents, cursor_tl[stim, :])
            r = RidgeCV(5).fit(dff_in[:, stim, :].T, cursor_tl[stim, :])
            r2_dff_rcv[stim] = r.score(dff_in[:, stim, :].T, cursor_tl[stim, :])
    return r2_l, r2_l2, r2_rcv, r2_dff_rcv


def obtain_engagement_line(folder_suite2p: Path, tos: str = 'stim'):
    """ function to obtain the engagement of indirect neurons to the cursor"""
    if tos == 'stim':
        index_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['stim_index']
    elif tos == 'target':
        index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['target_index']
    else:
        indices = np.sort(np.random.randint(AnalysisConfiguration.FA_event_frames + 1,
                                            AnalysisConstants.calibration_frames - AnalysisConfiguration.FA_event_frames - 1,
                                            size=AnalysisConfiguration.FA_len_SOT))
    dff = pp.obtain_dffs(folder_suite2p, smooth=True)
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    indirect_neurons = copy.deepcopy(is_cell)
    indirect_neurons[ensemble, :] = [0, 0]
    indirect_neurons[direct_neurons['exclude'], :] = [0, 0]
    indices = indices[np.where(np.logical_and(np.logical_and(indices +
                                                             AnalysisConfiguration.FA_rew_frames < dff.shape[1],
                                                             indices > AnalysisConfiguration.eng_event_frames),
                                              np.isin(indices, np.where(~np.isnan(dff.mean(0)))[0])))[0]]

    if len(indices) == 0:
        return np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan)
    dff_tl = pp.create_time_locked_array(dff, indices, (AnalysisConfiguration.eng_event_frames,
                                                        AnalysisConfiguration.FA_rew_frames))
    dff_dn_cursor = dff[ensemble, :]
    dff_dn = dff_tl[ensemble, :, :]
    dff_in = dff_tl[indirect_neurons[:, 0].astype(bool), :, :]
    cursor = - np.nanmean(dff_dn_cursor[:2, :], 0) + np.nanmean(dff_dn_cursor[2:, :], 0)
    cursor_tl = pp.create_time_locked_array(cursor, indices, (AnalysisConfiguration.eng_event_frames,
                                                              AnalysisConfiguration.FA_rew_frames))
    win = int(np.ceil(AnalysisConfiguration.FA_stim_win / 2))
    if dff_tl.shape[1] > 2*win:
        r2_l = np.full(dff_tl.shape[1]-2*win, np.nan)
        r2_l2 = np.full(dff_tl.shape[1]-2*win, np.nan)
        r2_rcv = np.full(dff_tl.shape[1]-2*win, np.nan)
        r2_dff_rcv = np.full(dff_tl.shape[1]-2*win, np.nan)
    else:
        r2_l = np.full(1, np.nan)
        r2_l2 = np.full(1, np.nan)
        r2_rcv = np.full(1, np.nan)
        r2_dff_rcv = np.full(1, np.nan)

    if dff_in.shape[0] >= 2:
        dff_dn_aux = dff_dn.reshape((dff_dn.shape[0], -1))
        dff_in_aux = dff_in.reshape((dff_in.shape[0], -1))
        cursor_aux = cursor_tl.reshape(-1)
        for stim in np.arange(win, dff_tl.shape[1] - win):
            if np.sum(np.isnan(dff_dn_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])) > 0 or \
                    np.sum(np.isnan(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])) > 0 or \
                    np.sum(np.isnan(cursor_aux[(stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])) > 0:
                continue
            latents = dm.obtain_latent(dff_dn_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])
            r = RidgeCV(5).fit(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]].T, latents)
            r2_l[stim-win] = r.score(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]].T, latents)
            latents = dm.obtain_latent(dff_dn_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]], 2)
            r = RidgeCV(5).fit(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]].T, latents)
            r2_l2[stim-win] = r.score(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]].T, latents)
            latents = dm.obtain_latent(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]], 2)
            r = RidgeCV(5).fit(latents, cursor_aux[(stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])
            r2_rcv[stim-win] = r.score(latents, cursor_aux[(stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])
            r = RidgeCV(5).fit(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]].T,
                               cursor_aux[(stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])
            r2_dff_rcv[stim-win] = r.score(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]].T,
                                       cursor_aux[(stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])
    return r2_l, r2_l2, r2_rcv, r2_dff_rcv


def obtain_engagement_trial(folder_suite2p: Path, indices_lag: np.array, tos: str = 'stim'):
    """ function to obtain the engagement of indirect neurons to the cursor"""
    if tos == 'stim':
        index_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['stim_index']
    elif tos == 'target':
        index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['target_index']
    else:
        indices = np.sort(np.random.randint(AnalysisConfiguration.FA_event_frames + 1,
                                            AnalysisConstants.calibration_frames - AnalysisConfiguration.FA_event_frames - 1,
                                            size=AnalysisConfiguration.FA_len_SOT))
    dff = pp.obtain_dffs(folder_suite2p, smooth=True)
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    indirect_neurons = copy.deepcopy(is_cell)
    indirect_neurons[ensemble, :] = [0, 0]
    indirect_neurons[direct_neurons['exclude'], :] = [0, 0]

    win = int(np.ceil(AnalysisConfiguration.FA_stim_win / 2))
    if len(indices) >= 2*win:
        r2_l = np.full((len(indices)-2*win, len(indices_lag)), np.nan)
        r2_l2 = np.full((len(indices)-2*win, len(indices_lag)), np.nan)
        r2_rcv = np.full((len(indices)-2*win, len(indices_lag)), np.nan)
        r2_dff_rcv = np.full((len(indices)-2*win, len(indices_lag)), np.nan)
    else:
        return np.full((1, len(indices_lag)), np.nan), np.full((1, len(indices_lag)), np.nan), \
               np.full((1, len(indices_lag)), np.nan), np.full((1, len(indices_lag)), np.nan)

    for ii, il in enumerate(indices_lag):
        indices_aux = indices + il
        indices_aux = indices_aux[np.where(np.logical_and(np.logical_and(indices_aux +
                                                                 AnalysisConfiguration.FA_rew_frames < dff.shape[1],
                                                                 indices_aux > AnalysisConfiguration.eng_event_frames),
                                                  np.isin(indices_aux, np.where(~np.isnan(dff.mean(0)))[0])))[0]]

        dff_tl = pp.create_time_locked_array(dff, indices_aux, (AnalysisConfiguration.eng_event_frames,
                                                            AnalysisConfiguration.FA_rew_frames))
        dff_dn_cursor = dff[ensemble, :]
        dff_dn = dff_tl[ensemble, :, :]
        dff_in = dff_tl[indirect_neurons[:, 0].astype(bool), :, :]
        cursor = - np.nanmean(dff_dn_cursor[:2, :], 0) + np.nanmean(dff_dn_cursor[2:, :], 0)
        cursor_tl = pp.create_time_locked_array(cursor, indices_aux, (AnalysisConfiguration.eng_event_frames,
                                                                  AnalysisConfiguration.FA_rew_frames))

        if dff_in.shape[0] >= 2:
            dff_dn_aux = dff_dn.reshape((dff_dn.shape[0], -1))
            dff_in_aux = dff_in.reshape((dff_in.shape[0], -1))
            cursor_aux = cursor_tl.reshape(-1)
            for stim in np.arange(win, dff_tl.shape[1] - win):
                if np.sum(np.isnan(dff_dn_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])) > 0 or \
                        np.sum(np.isnan(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])) > 0 or \
                        np.sum(np.isnan(cursor_aux[(stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])) > 0:
                    continue
                latents = dm.obtain_latent(dff_dn_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])
                r = RidgeCV(5).fit(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]].T, latents)
                r2_l[stim-win, ii] = r.score(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]].T, latents)
                latents = dm.obtain_latent(dff_dn_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]], 2)
                r = RidgeCV(5).fit(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]].T, latents)
                r2_l2[stim-win, ii] = r.score(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]].T, latents)
                latents = dm.obtain_latent(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]], 2)
                r = RidgeCV(5).fit(latents, cursor_aux[(stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])
                r2_rcv[stim-win, ii] = r.score(latents, cursor_aux[(stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])
                r = RidgeCV(5).fit(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]].T,
                                   cursor_aux[(stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])
                r2_dff_rcv[stim-win, ii] = r.score(dff_in_aux[:, (stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]].T,
                                           cursor_aux[(stim - win)*dff_dn.shape[2]: (stim + win)*dff_dn.shape[2]])
    return r2_l, r2_l2, r2_rcv, r2_dff_rcv


def obtain_SOT_over_trial(folder_suite2p: Path, indices_lag: np.array, tos: str = 'stim') \
        -> Tuple[np.array, np.array]:
    """ function to obtain the SOT over time for direct and indirect neurons with neurons x time """
    if tos == 'stim':
        index_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['stim_index']
    elif tos == 'target':
        index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['target_index']
    else:
        indices = np.sort(np.random.randint(AnalysisConfiguration.FA_event_frames + 1,
                                            AnalysisConstants.calibration_frames - AnalysisConfiguration.FA_rew_frames - 1,
                                            size=AnalysisConfiguration.FA_len_SOT))
    spks_dff = np.load(Path(folder_suite2p) / "spks_dff.npy")
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    indirect_neurons = copy.deepcopy(is_cell)
    indirect_neurons[ensemble, :] = [0, 0]
    indirect_neurons[direct_neurons['exclude'], :] = [0, 0]
    SOT_t_dn = np.full((len(indices), len(indices_lag)), np.nan)
    SOT_t_in = np.full((len(indices), len(indices_lag)), np.nan)

    for ii, il in enumerate(indices_lag):
        indices_aux = indices + il
        indices_aux = indices_aux[np.where(np.logical_and(np.logical_and(indices_aux +
                                                                 AnalysisConfiguration.FA_rew_frames < spks_dff.shape[1],
                                                                 indices_aux > AnalysisConfiguration.FA_event_frames),
                                                  np.isin(indices_aux, np.where(~np.isnan(spks_dff.mean(0)))[0])))[0]]

        SOT_stim_dn, SOT_stim_in, _, _ = obtain_SOT_event(indices_aux, ensemble, indirect_neurons,
                                                                                spks_dff)
        min_x = np.min([SOT_stim_dn.shape[0], SOT_t_dn.shape[0]])
        SOT_t_dn[:min_x, ii] = SOT_stim_dn[:min_x]
        SOT_t_in[:min_x, ii] = SOT_stim_in[:min_x]
    return SOT_t_dn, SOT_t_in


def obtain_SOT_over_all_trials(folder_suite2p: Path, indices_lag: np.array, tos: str = 'stim') -> Tuple[np.array, np.array]:
    """ function to obtain the SOT over time for direct and indirect neurons with stimxtime"""
    if tos == 'stim':
        index_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['stim_index']
    elif tos == 'target':
        index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['target_index']
    else:
        indices = np.sort(np.random.randint(AnalysisConfiguration.FA_event_frames + 1,
                                            AnalysisConstants.calibration_frames - AnalysisConfiguration.FA_rew_frames - 1,
                                            size=AnalysisConfiguration.FA_len_SOT))
    spks_dff = np.load(Path(folder_suite2p) / "spks_dff.npy")
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    indirect_neurons = copy.deepcopy(is_cell)
    indirect_neurons[ensemble, :] = [0, 0]
    indirect_neurons[direct_neurons['exclude'], :] = [0, 0]
    SOT_t_dn = np.full((len(indices), len(indices_lag)), np.nan)
    SOT_t_in = np.full((len(indices), len(indices_lag)), np.nan)
    for ii, il in enumerate(indices_lag):
        indices_aux = indices + il
        indices_aux = indices_aux[np.where(np.logical_and(np.logical_and(indices_aux +
                                                                 AnalysisConfiguration.FA_rew_frames < spks_dff.shape[1],
                                                                 indices_aux > AnalysisConfiguration.FA_event_frames),
                                                  np.isin(indices_aux, np.where(~np.isnan(spks_dff.mean(0)))[0])))[0]]

        SOT_ln_dn, SOT_ln_in = obtain_SOT_line(indices_aux, ensemble, indirect_neurons, spks_dff)
        min_x = np.min([SOT_ln_dn.shape[0], SOT_t_dn.shape[0]])
        SOT_t_dn[:min_x, ii] = SOT_ln_dn[:min_x]
        SOT_t_in[:min_x, ii] = SOT_ln_in[:min_x]
    return SOT_t_dn, SOT_t_in

def obtain_activation_vs_reinforcement(folder_suite2p: Path, indices: np.array) -> Tuple[np.array, np.array]:
    """ function to obtain the relationship between activation of neurons and reinforcement """


def obtain_rcv_stim(array_stim: np.array) -> np.array:
    """ Function to obtain the linear regression RidgeCV to an array of data
    with dimensions (neurons, stims, time)"""
    neurons, stims, time = array_stim.shape

    # Initialize the output matrix with NaNs
    rcv = np.full((neurons, stims), np.nan)

    # Iterate over each neuron
    for neuron in range(neurons):
        # Iterate over stims, starting from the 6th one
        for stim in range(5, stims):
            # Extract the target data for the current stim
            y = array_stim[neuron, stim, :]
            # Extract the data for the previous 5 stims
            X = array_stim[neuron, stim - 5:stim, :].reshape(-1, time).T

            # Initialize the RidgeCV model with a fixed alpha of 5
            model = RidgeCV()

            # Fit the model
            try:
                model.fit(X, y)
                # Store the score, change this line if you want to store something else like coefficients
                rcv[neuron, stim] = model.score(X, y)
            except Exception as e:
                # In case the model cannot be fitted, the corresponding entry remains NaN
                print(f"Error fitting model for neuron {neuron}, stim {stim}: {e}")
                continue
    return rcv


def find_events_with_preceding_avg(zscore_arr:np.array, thres:float, win:int) -> list:
    """
    Identify events based on the average of a window of size 'win' before the index
    being greater than a specified threshold 'thres', disregarding events closer
    than the size of 'win'.

    Parameters:
    - zscore_arr: 2D numpy array with dimensions [neurons, time], containing z-scored values.
    - thres: Float, the threshold for the average z-score over the window to identify events.
    - win: Integer, the size of the window to average over before the index.

    Returns:
    - A list of numpy arrays, where each array contains the indices of events for each neuron,
      based on the average z-score of the preceding window being above the threshold.
    """
    events = []
    for neuron_data in zscore_arr:
        neuron_events = []
        last_event_index = -win  # Initialize to avoid selecting the first event if too early
        for i in range(win, len(neuron_data)):
            # Calculate average of the window before the current index
            avg_preceding_window = np.mean(neuron_data[i - win:i])
            # Check if the average of the preceding window is above the threshold
            # and ensure the event is sufficiently far from the last event
            if avg_preceding_window > thres and i - last_event_index >= win:
                neuron_events.append(i)
                last_event_index = i  # Update the last event index to enforce the minimum gap
        events.append(np.array(neuron_events))
    return events


def obtain_dp_events_per_event(folder_suite2p: Path, thres:float, win:int) -> (
        Tuple[pd.DataFrame, pd.DataFrame]):
    """
    function to obtain a dp with the events where the zscore was bigger than thres
    :param folder_suite2p: path where the data is stored
    :param thres: zscore minimim to consider it an event
    :param win: size of the window to calculate the zscore event
    :return
        df: Dataframe with all the events as index
            'Calibration': Was the event during calibration,
            'Baseline': Was the event during baseline,
            'neuron': neuron_idx,
            'Type': neuron_type,
            'Index_ev': event_index,
            'stim': simultaneous stim?,
            'next': how long until the next event for that neuron
            'next_5':  how long until 5 events happen
            'prev': how much time from the previous event
            'prev_5': how much time from the previous event
            'zscore': the zscore of this event
            'zscore_next': the zscore of the next event
            'zscore_next_5': the average zscore of the next 5 event
            'zscore_prev': the zscore of the next event
            'zscore_prev_5': the average zscore of the next 5 event
            'rate_min': the rate of events per min in the next min
            'rate_5min': the rate of events per min in the next 5 min
            'rate_all': the rate of events per min for the rest of the time
            'false_pos_last': number of false positives (there was an event without stim) since last stim
            'false_neg_last': number of false negatives (there was an stim without event) since last event

    """
    # obtain values
    index_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
    index_dict = index_aux.take(0)
    indices = index_dict['stim_index']

    spks_dff = np.load(Path(folder_suite2p) / "spks_dff.npy")
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)

    zscore_arr = ut.calculate_zscore_neuron_wise(spks_dff)
    events = find_events_with_preceding_avg(zscore_arr, thres, win)
    frame_min = int(AnalysisConstants.framerate*60)  # frames per minute
    is_cell[direct_neurons['E1'] + direct_neurons['E2'], 0] = 1


    ret = collections.defaultdict(list)
    ret_stim = collections.defaultdict(list)

    for neuron_idx, neuron_events in enumerate(events):
        if is_cell[neuron_idx, 0]:
            # Determine neuron type
            neuron_type = 'Indirect'
            for key, values in direct_neurons.items():
                if key in ['E1', 'E2']:
                    if neuron_idx in values:
                        neuron_type = key
                        break
                    else:
                        neuron_type = 'na'

            for ev, event_index in enumerate(neuron_events):
                stim_distance = np.where(indices > event_index)[0]
                if len(stim_distance) > 0:
                    ret['neuron'].append(neuron_idx)
                    ret['neuron_type'].append(neuron_type)
                    ret['event_index'].append(event_index)
                    ret['calibration'].append(event_index < AnalysisConstants.calibration_frames)
                    ret['baseline'].append(event_index < indices[0])
                    ret['stim_dist'].append(indices[stim_distance[0]] - event_index)
                    ret['zscore'].append(np.nanmean(zscore_arr[neuron_idx, event_index - win:event_index]))
                    if ev < len(neuron_events) - 1:
                        ret['zscore_next'].append(np.nanmean(zscore_arr[neuron_idx, neuron_events[ev + 1] - win:
                                                                                    neuron_events[ev + 1]]))
                        ret['next'].append(neuron_events[ev + 1] - event_index)
                    else:
                        ret['zscore_next'].append(np.nan)
                        ret['next'].append(np.nan)
                    if ev < len(neuron_events) - 5:
                        aux_zscore = np.full(5, np.nan)
                        for ii in np.arange(1, 6):
                            aux_zscore[ii-1] = np.nanmean(zscore_arr[neuron_idx, neuron_events[ev + ii] - win:
                                                                                    neuron_events[ev + ii]])
                        ret['zscore_next_5'].append(np.nanmean(aux_zscore))
                        ret['next_5'].append(neuron_events[ev + 5] - event_index)
                    else:
                        ret['zscore_next_5'].append(np.nan)
                        ret['next_5'].append(np.nan)
                    if ev >= 1 and ((neuron_events[ev - 1] - win) > 0):
                        ret['zscore_prev'].append(np.nanmean(zscore_arr[neuron_idx, neuron_events[ev - 1] - win:
                                                                                    neuron_events[ev - 1]]))
                        ret['prev'].append(event_index - neuron_events[ev-1])
                    else:
                        ret['zscore_prev'].append(np.nan)
                        ret['prev'].append(np.nan)
                    if ev >= 5 and ((neuron_events[ev - 5] - win) > 0):
                        aux_zscore = np.full(5, np.nan)
                        for ii in np.arange(1 , 6):
                            aux_zscore[ii-1] =  np.nanmean(zscore_arr[neuron_idx, neuron_events[ev - ii] - win:
                                                                                    neuron_events[ev - ii]])
                        ret['zscore_prev_5'].append(np.nanmean(aux_zscore))
                        ret['prev_5'].append(event_index - neuron_events[ev-5])
                    else:
                        ret['zscore_prev_5'].append(np.nan)
                        ret['prev_5'].append(np.nan)
                    if event_index < zscore_arr.shape[1] - frame_min:
                        ret['rate_min'].append(np.sum(np.logical_and(neuron_events > event_index,
                                                                     neuron_events < event_index + frame_min)))
                    else:
                        ret['rate_min'].append(np.nan)
                    if event_index < zscore_arr.shape[1] - 5 * frame_min:
                        ret['rate_5min'].append(np.sum(np.logical_and(neuron_events > event_index,
                                                                     neuron_events < event_index + 5 * frame_min))/5)
                    else:
                        ret['rate_5min'].append(np.nan)
                    frames_left = zscore_arr.shape[1] - event_index
                    time_left = frames_left / AnalysisConstants.framerate / 60
                    ret['rate_all'].append(np.sum(neuron_events > event_index)/ time_left)

                    # obtain the closest stim (in the past)
                    ret['false_pos_last'].append(ut.count_since_last(indices, neuron_events, ev, False))
                    ret['false_neg_last'].append(sum(1 for i in indices if neuron_events[ev-1] < i < event_index))

    return pd.DataFrame(ret)



def obtain_dp_events_per_stim(folder_suite2p: Path, thres:float, win:int) -> (
        Tuple[pd.DataFrame, pd.DataFrame]):
    """
    function to obtain a dp with the events where the zscore was bigger than thres
    :param folder_suite2p: path where the data is stored
    :param thres: zscore minimim to consider it an event
    :param win: size of the window to calculate the zscore event
    :return
        df: Dataframe with all the events as index
            'Calibration': Was the event during calibration,
            'Baseline': Was the event during baseline,
            'neuron': neuron_idx,
            'Type': neuron_type,
            'Index_ev': event_index,
            'stim': simultaneous stim?,
            'next': how long until the next event for that neuron
            'next_5':  how long until 5 events happen
            'prev': how much time from the previous event
            'prev_5': how much time from the previous event
            'zscore': the zscore of this event
            'zscore_next': the zscore of the next event
            'zscore_next_5': the average zscore of the next 5 event
            'zscore_prev': the zscore of the next event
            'zscore_prev_5': the average zscore of the next 5 event
            'rate_min': the rate of events per min in the next min
            'rate_5min': the rate of events per min in the next 5 min
            'rate_all': the rate of events per min for the rest of the time
            'false_pos_last': number of false positives (there was an event without stim) since last stim
            'false_neg_last': number of false negatives (there was an stim without event) since last event

    """
    # obtain values
    index_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
    index_dict = index_aux.take(0)
    indices = index_dict['stim_index']

    spks_dff = np.load(Path(folder_suite2p) / "spks_dff.npy")
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)

    zscore_arr = ut.calculate_zscore_neuron_wise(spks_dff)
    events = find_events_with_preceding_avg(zscore_arr, thres, win)
    frame_min = int(AnalysisConstants.framerate*60)  # frames per minute
    is_cell[direct_neurons['E1'] + direct_neurons['E2'], 0] = 1

    ret = collections.defaultdict(list)
    ret_stim = collections.defaultdict(list)

    for neuron_idx, neuron_events in enumerate(events):
        if is_cell[neuron_idx, 0]:
            # Determine neuron type
            neuron_type = 'Indirect'
            for key, values in direct_neurons.items():
                if key in ['E1', 'E2']:
                    if neuron_idx in values:
                        neuron_type = key
                        break
                    else:
                        neuron_type = 'na'

            for ind, index in enumerate(indices):
                event_distance = np.where(neuron_events < index)[0]
                if len(event_distance) > 0:
                    ret_stim['neuron'].append(neuron_idx)
                    ret_stim['neuron_type'].append(neuron_type)
                    ret_stim['stim_index'].append(index)
                    ret_stim['event_dist'].append(index - neuron_events[event_distance[-1]])
                    next_event = np.where(neuron_events>index)[0]
                    if len(next_event) > 0:
                        ret_stim['next'].append(neuron_events[next_event[0]] - index)
                        ret_stim['zscore_next'].append(np.nanmean(zscore_arr[neuron_idx,
                                                                  neuron_events[next_event[0]] - win:
                                                                  neuron_events[next_event[0]]]))
                    else:
                        ret_stim['next'].append(np.nan)
                        ret_stim['zscore_next'].append(np.nan)
                    if index < zscore_arr.shape[1] - frame_min:
                        ret_stim['rate_min'].append(np.sum(np.logical_and(neuron_events > index,
                                                                     neuron_events < index + frame_min)))
                    else:
                        ret_stim['rate_min'].append(np.nan)

    return pd.DataFrame(ret_stim)






