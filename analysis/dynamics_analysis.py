__author__ = 'Nuria'

import numpy as np
import copy
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

    SOT_dn, SOT_in = obtain_SOT_all(ensemble, indirect_neurons, spks_dff[:, frames])
    return SOT_dn, SOT_in


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
    _, SOT_dn, _ = dm.obtain_FA(spks_dn, 2)
    if spks_in.shape[0] > len(ensemble):
        for iter in np.arange(AnalysisConfiguration.FA_n_iter):
            selected_neurons = np.random.choice(np.arange(spks_in.shape[0]), size=len(ensemble), replace=False)
            _, SOT_aux, _ = dm.obtain_FA(spks_in[selected_neurons, :], 2)
            SOT_in[iter] = SOT_aux

    return SOT_dn, SOT_in.mean()


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