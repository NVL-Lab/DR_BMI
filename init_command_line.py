
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
from preprocess.prepare_data import prepare_ops_file_2nd_pass, prepare_ops_1st_pass
from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants
from utils import util_plots as ut_plots
from analysis import learning

interactive(True)

folder_raw = Path("F:/data/raw")
folder_save = Path("F:/data/process")
default_path = folder_save / "default_var"


###########################################################
# suite2p
import pandas as pd
from pathlib import Path
from preprocess.prepare_data import prepare_ops_file_2nd_pass, prepare_ops_1st_pass

folder_raw = Path("F:/data/raw")
folder_save = Path("F:/data/process")
default_path = folder_save / "default_var"

folder_experiment = Path('F:/data/raw/ago13/221112/D01')
folder_processed_experiment = Path('F:/data/process/ago13/221112/D01')
file_path = folder_processed_experiment / 'suite2p' / 'plane0'
if not Path(file_path).exists():
    Path(file_path).mkdir(parents=True, exist_ok=True)
file_origin = 'F:/data/raw/ago13/221112/D01/im/baseline/baseline_221112T092905-237'

data_path = [str(folder_experiment / 'im/baseline/baseline_221112T092905-237'),
              str(folder_experiment / 'im/BMI_stim/BMI_stim_221112T095524-239')]

db = {
    'data_path': data_path,
    'save_path0': str(folder_processed_experiment),
    'ops_path': str(file_path / 'ops.npy'),
    'fast_disk': str(Path('C:/Users/Nuria/Documents/DATA')),
      }

ops = prepare_ops_1st_pass(default_path, db['ops_path'])

from suite2p.run_s2p import run_s2p
ops1 = run_s2p(ops, db)


# AFTER IT RUNS
classfile = suite2p.classification.builtin_classfile
ops2, stat = suite2p.detection_wrapper(f_reg=ops['reg_file'], ops=ops1, classfile=classfile)



####
from preprocess.process_data import run_all_experiments

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

###########################################################
# MOTION
motion_data = pd.read_parquet(folder_save / 'motion_data.parquet')
motion_behavior = pd.read_parquet(folder_save / 'motion_behavior.parquet')
folder_plots = Path('F:/data/process/plots/learning')