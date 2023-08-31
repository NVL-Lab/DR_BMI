__author__ = 'Nuria'

import numpy as np
import copy
from pathlib import Path

from typing import Tuple

import utils.utils_analysis as ut
import analysis.dynamics_mat as dm
import preprocess.prepare_data as pp
from utils.analysis_command import AnalysisConfiguration


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
    len_mat = int(spks_av.shape[1]/2)

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
        _, SOT_all, VAF_all = dm.obtain_FA(spks_av[selected_neurons, :], n_components=AnalysisConfiguration.FA_components)
        _, SOT_a, VAF_a = dm.obtain_FA(spks_a, n_components=AnalysisConfiguration.FA_components)
        _, SOT_b, VAF_b = dm.obtain_FA(spks_b, n_components=AnalysisConfiguration.FA_components)
        dim_array[iter, :] = [dim_all, dim_a, dim_b]
        SOT_array[iter, :] = [SOT_all, SOT_a, SOT_b]
        VAF_array[iter, :] = [VAF_all, VAF_a, VAF_b]
    return dim_array.mean(0), SOT_array.mean(0), VAF_array.mean(0)


def obtain_SOT_over_time(folder_suite2p: Path, spks_win: int = 3) -> Tuple[np.array, np.array, np.array, np.array]:
    """ function to obtain the SOT over time for direct and indirect neurons """
    stim_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
    stim_time_dict = stim_aux.take(0)
    stim_index = stim_time_dict['stim_time']
    spks_dff = np.load(Path(folder_suite2p) / "spks_dff.npy")
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    indirect_neurons = copy.deepcopy(is_cell)
    indirect_neurons[ensemble, :] = [0, 0]

    spks_tl = pp.create_time_locked_array(spks_dff, stim_index, (60, 60))
    spks_dn = spks_tl[ensemble, :, :]
    spks_in = spks_tl[indirect_neurons[:, 0].astype(bool), :, :]
    SOT_stim_dn = np.full(spks_tl.shape[1], np.nan)
    SOT_stim_in = np.full((spks_tl.shape[1], AnalysisConfiguration.FA_n_iter), np.nan)
    SOT_stim_all = np.full(spks_tl.shape[1], np.nan)
    DIM_stim_all = np.full(spks_tl.shape[1], np.nan)
    for stim in np.arange(spks_tl.shape[1]):
        _, SOT_dn, _ = dm.obtain_FA(ut.sum_array_samples(spks_dn[:, stim, :], 1, spks_win), 2)
        SOT_stim_dn[stim] = SOT_dn
        _, SOT_all, _ = dm.obtain_FA(ut.sum_array_samples(spks_tl[:, stim, :], 1, spks_win), 4)
        DIM_all, _, _ = dm.obtain_FA(ut.sum_array_samples(spks_tl[:, stim, :], 1, spks_win), VAF=0.9)
        SOT_stim_all[stim] = SOT_all
        DIM_stim_all[stim] = DIM_all
        for iter in np.arange(AnalysisConfiguration.FA_n_iter):
            selected_neurons = np.random.choice(np.arange(spks_in.shape[0]), size=len(ensemble), replace=False)
            _, SOT_in, _ = dm.obtain_FA(ut.sum_array_samples(spks_in[selected_neurons, stim, :], 1, spks_win), 2)
            SOT_stim_in[stim, iter] = SOT_in

    return SOT_stim_dn, SOT_stim_in, SOT_stim_all, DIM_stim_all


def obtain_SOT_av(folder_suite2p: Path) -> Tuple[float, float]:
    """ function to obtain the SOT of the average spks data on direct and indirect neurons """
    stim_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
    stim_time_dict = stim_aux.take(0)
    stim_index = stim_time_dict['stim_time']
    spks_dff = np.load(Path(folder_suite2p) / "spks_dff.npy")
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    indirect_neurons = copy.deepcopy(is_cell)
    indirect_neurons[ensemble, :] = [0, 0]
    spks_av = np.nanmean(pp.create_time_locked_array(spks_dff, stim_index, (60, 60)), 1)
    spks_dn_av = spks_av[ensemble, :]
    spks_in_av = spks_av[indirect_neurons[:, 0].astype(bool), :]

    _, SOT_dn_all, _ = dm.obtain_FA(spks_dn_av, 2)
    SOT_in_all = np.full(AnalysisConfiguration.FA_n_iter, np.nan)
    for iter in np.arange(AnalysisConfiguration.FA_n_iter):
        selected_4 = np.random.choice(np.arange(spks_in_av.shape[0]), size=len(ensemble), replace=False)
        _, SOT_in, _ = dm.obtain_FA(spks_in_av[selected_4, :], 2)
        SOT_in_all[iter] = SOT_in

    return SOT_dn_all, SOT_in_all.mean()

