
__author__ = 'Nuria'

import scipy
import collections
import pandas as pd
import numpy as np
import sklearn as sk

from pathlib import Path
from typing import Optional

from preprocess import sessions as ss


def obtain_target_info (file_path: Path) -> pd.DataFrame:
    """ Function to retrieve the info inside target_info"""
    # Todo check what is wrong with scipy.io and other ways to load mat in python