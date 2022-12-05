
__author__ = 'Nuria'

import collections
import pandas as pd
import numpy as np
import sklearn as sk
import scipy.io as sio

from pathlib import Path
from typing import Optional


from preprocess import sessions as ss


def obtain_target_info (file_path: Path) -> pd.DataFrame:
    """ Function to retrieve the info inside target_info"""
    # Todo check what is wrong with scipy.io and other ways to load mat in python


def obtain_online_data(file_path: Path) -> dict:
    """ Function to retrieve the info inside BMI online """
    bmi_online = sio.loadmat(file_path, simplify_cells=True)
    return bmi_online
