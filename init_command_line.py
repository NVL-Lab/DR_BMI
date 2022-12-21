
__author__ = 'Nuria'

import traja
from motion import motion_analysis as ma
from motion import motion_population as mp

import scipy
import os
import collections
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns

from pathlib import Path
from matplotlib import interactive

from preprocess import sessions as ss
from preprocess.prepare_data import prepare_ops_file_2nd_pass
from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants
from utils import util_plots as ut_plots
from analysis import learning

interactive(True)

folder_experiments = Path("F:/data/raw")
folder_save = Path("F:/data/process")

motion_data = pd.read_parquet(folder_save / 'motion_data.parquet')
motion_behavior = pd.read_parquet(folder_save / 'motion_behavior.parquet')
folder_plots = Path('F:/data/process/plots/learning')


###########################################################
# suite2p
file_path = 'F:/data/process/ago13/221112/D01'
file_origin = 'F:/data/raw/ago13/221112/D01/im/baseline/baseline_221112T092905-237'
prepare_ops_file_2nd_pass(file_path, file_origin)

# TODO OTHERS
# simulate BMI after doing the motion correction (For HOLOBMI)

# TODO ANALYSIS
# check if the chance of silence at the beginning of the experiment is not by chance. the increase is real. MEaning
# there are not trains/bouts
# plot the hits/min average per mice
# obtain occupancy as per Vivek's paper
# check if the whole brain is just more active (cursor is uniform instead of pseudo gaussian)
# XGBoost? on what?

# TODO CONTROLS
# check the time of the stim regarding the hit (Before/after, does it matter)
# regress hits by stim closeness (before/after)
# check success vs time of the experiment was it before or after another experiment, first of the day, first of the set
# check general neuronal activity differences in all experiment types?

# TODO MOTION
# can we synchronize well the motion sensor with the 2p data online and images?
# check increase decrease of motion and other charactericts of motion during baseline / BMI / periods of BMI/ stim
# relationship between motion and reward rate?
# relationship between motion and neuronal activity? regress motion?

# TODO minor:
# change the name of the repository to stimBMIPrairie


# TODO FUTURE CONTROLS:
# is no-audio an issue? is it volitional? is it only mechanistic? --> first day 3 sessions of random_stim no audio
# second day 3 sessions of bmi_atim_ago no audio (Difference? yes? total 5 sessions).
# Third day random_stim with audio. 4th day normal
# new control extinction...
