__author__ = 'Nuria'

import numpy as np
from typing import Optional, Tuple
from sklearn.decomposition import FactorAnalysis


def obtain_FA(mat_a: np.array, n_components: Optional[int] = None, VAF: Optional[float] = None) \
        -> Tuple[int, float, float]:
    """ function to calculate FA and obtain the number of dimensions required to acount for VAF and SOT
    with mat_a of N(neurons) * T(time)"""
    fa = FactorAnalysis(n_components=n_components)
    fa.fit(mat_a.T)

    # obtain SOT
    L = fa.components_
    sharedCov_mat = np.dot(L.T, L)
    sharedCov = np.diag(sharedCov_mat)
    privateCov = fa.noise_variance_

    totalCov = sharedCov + privateCov
    SOT = np.sum(sharedCov) / np.sum(totalCov)

    # obtain dimensionality
    cum_VAF = np.cumsum(np.sum(L**2, axis=1)/mat_a.var(axis=1).sum())
    if VAF is not None:
        fa_dim = np.where(cum_VAF > VAF)[0]
        if len(fa_dim)>0:
            fa_dim = fa_dim[0]
        else:
            fa_dim = mat_a.shape[0]
    else:
        fa_dim = n_components
    return fa_dim, SOT, cum_VAF[-1]
