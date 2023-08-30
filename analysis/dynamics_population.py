__author__ = 'Nuria'

import collections

import pandas as pd
from pathlib import Path

from preprocess import sessions as ss
from analysis import dynamics_analysis as da


def obtain_manifold_spontaneous(folder_list: list) -> pd.DataFrame:
    """ function to obtain gain for all experiments """
    ret = collections.defaultdict(list)
    df_sessions = ss.get_sessions_df(folder_list, 'BEHAVIOR')
    mice = df_sessions.mice_name.unique()
    for aa, mouse in enumerate(mice):
        df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
        for index, row in df_sessions_mouse.iterrows():
            folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
            folder_processed_experiment = Path(folder_process) / row['session_path'] / 'behavior'
            folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
            ret['mice'].append(mouse)
            dim_array, SOT_array, VAF_array = da.obtain_manifold(folder_suite2p)
            ret['dim_sa'].append(dim_array[2])
            ret['dim_all'].append(dim_array[0])
            ret['dim_d1r'].append(dim_array[1])
            ret['SOT_sa'].append(SOT_array[2])
            ret['SOT_all'].append(SOT_array[0])
            ret['SOT_d1r'].append(SOT_array[1])
            ret['VAF_sa'].append(VAF_array[2])
            ret['VAF_all'].append(VAF_array[0])
            ret['VAF_d1r'].append(VAF_array[1])
    return pd.DataFrame(ret)