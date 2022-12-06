__author__ = 'Nuria'

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
from matplotlib import interactive


def plot_movement_features(motion_data: pd.DataFrame, motion_behavior: pd.DataFrame):
    """ Function to plot all features of movement """
    for feature in motion_behavior.columns[4:]:
        
