__author__ = 'Nuria'

import scipy
import pandas as pd
import numpy as np
import sklearn as sk
from pathlib import Path

from preprocess import sessions as ss
from utils.analysis_command import AnalysisConfiguration


def compare_bmi_baseline(folder_experiments: Path, experiment_type: str):
    """ function to compare motion characteristics between baseline and experiment """
    df_sessions = ss.get_sessions_df(folder_experiments, experiment_type)
    mice = df_sessions.mice_name.unique()
    for aa, mouse in enumerate(mice):
        df_sessions_mouse = df_sessions[df_sessions.mice_name == mouse]
        for index, row in df_sessions_mouse.iterrows():
            session_name = row['session_name']
