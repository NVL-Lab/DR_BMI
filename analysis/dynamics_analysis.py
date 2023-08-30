__author__ = 'Nuria'

import numpy as np
from pathlib import Path

import utils.utils_analysis as ut
import analysis.dynamics_mat as dm
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

