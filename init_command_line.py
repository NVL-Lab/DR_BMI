__author__ = 'Nuria'

import traja
from motion import motion_analysis as ma
from motion import motion_population as mp

import scipy
import os
import shutil
import collections
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns

import copy
from pathlib import Path
from matplotlib import interactive

import preprocess.prepare_data as pp
import utils.utils_analysis as ut
from preprocess import sessions as ss
from preprocess.prepare_data import prepare_ops_1st_pass
from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants
from utils import util_plots as ut_plots
from analysis import learning_analysis
import analysis.dynamics_mat as dm
from analysis import dynamics_analysis as da
from analysis import dynamics_population as dp


interactive(True)

folder_list = {'FA': 'D:/data', 'FB': 'F:/data', 'FC': 'G:/data'}
folder_data = Path("C:/Users/Nuria/Documents/DATA/D1exp/df_data")
folder_plots = Path("C:/Users/Nuria/Documents/DATA/D1exp/plots")
default_path = Path("C:/Users/Nuria/Documents/DATA/D1exp/default_var")

###########################################################
# suite2p
import pandas as pd
import numpy as np
from pathlib import Path
from preprocess.prepare_data import prepare_ops_1st_pass, obtain_bad_frames_from_fneu
from preprocess import sessions as ss

folder_raw = Path("F:/data/raw")
folder_save = Path("F:/data/process")
default_path = folder_save / "default_var"

folder_experiment = Path('F:/data/raw/ago16/221116/D05-2')
folder_processed_experiment = Path('F:/data/process/ago16/221115/D04-2')
folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
if not Path(folder_suite2p).exists():
    Path(folder_suite2p).mkdir(parents=True, exist_ok=True)
file_origin = 'F:/data/raw/ago13/221112/D01/im/baseline/baseline_221112T092905-237'

data_path = [str(folder_experiment / 'im/baseline/baseline_221112T092905-237'),
             str(folder_experiment / 'im/BMI_stim/BMI_stim_221112T095524-239')]

db = {
    'data_path': data_path,
    'save_path0': str(folder_processed_experiment),
    'ops_path': str(folder_suite2p / 'ops.npy'),
    'fast_disk': str(Path('C:/Users/Nuria/Documents/DATA')),
}

ops = prepare_ops_1st_pass(default_path, db['ops_path'])

from suite2p.run_s2p import run_s2p

ops1 = run_s2p(ops, db)

# AFTER IT RUNS
classfile = suite2p.classification.builtin_classfile
ops2, stat = suite2p.detection_wrapper(f_reg=ops['reg_file'], ops=ops1, classfile=classfile)

# try not anatomical

folder_experiment = Path('F:/data/raw/ago16/221115/D04-2')
folder_processed_experiment = Path('F:/data/process/ago16/221115/D04-2')
folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
if not Path(folder_suite2p).exists():
    Path(folder_suite2p).mkdir(parents=True, exist_ok=True)
file_origin = 'F:/data/raw/ago16/221115/D04-2/im/baseline/baseline_221115T171903-239'

data_path = [str(folder_experiment / 'im/baseline/baseline_221115T171903-239'),
             str(folder_experiment / 'im/BMI_stim/BMI_stim_221115T173703-240')]

db = {
    'data_path': data_path,
    'save_path0': str(folder_processed_experiment),
    'ops_path': str(folder_suite2p / 'ops.npy'),
    'fast_disk': str(Path('C:/Users/Nuria/Documents/DATA')),
}

ops = prepare_ops_1st_pass(default_path, db['ops_path'])

#### some analyisis df_occupnacy
df_occupancy = pd.read_parquet(Path(folder_save) / "df_occupancy.parquet")
####


### obtain the direct neurons
from preprocess.prepare_data import obtain_bad_frames_from_fneu

experiment_type = "BMI_CONTROL_LIGHT"
df = ss.get_sessions_df(folder_raw, experiment_type)
exp_info = df.loc[0]
folder_path = folder_raw / exp_info['session_path']
folder_suite2p = folder_save / exp_info['session_path'] / 'suite2p' / 'plane0'

### obtain the pds for motion
df_aux2 = pd.read_parquet("C:/Users/Nuria/Documents/DATA/D1exp/df_data/motion_behavior.parquet")
df_aux1 = pd.read_parquet("C:/Users/Nuria/Documents/DATA/D1exp/df_data/motion_data.parquet")

df_control = df_aux2[df_aux2.BB == "BMI"]
df_control = df_control.drop(columns="BB")

df_aux1["Laser"] = "ON"
df_aux1 = df_aux1[df_aux1["experiment"] == "Behavior_before"]
df_aux1 = df_aux1.drop(columns="experiment")

df_aux2["Laser"] = "OFF"
df_aux2.loc[df_aux2.BB == "BMI", "Laser"] = "BMI"
df_aux2 = df_aux2.drop(columns="BB")
df_aux2 = df_aux2[df_aux2.experiment.isin(['D1act', 'RANDOM', 'NO_AUDIO', 'DELAY'])]
df_aux2 = df_aux2.drop(columns="experiment")
df_motion = pd.concat((df_aux1, df_aux2))

df_control.to_parquet("C:/Users/Nuria/Documents/DATA/D1exp/df_data/df_motion_controls.parquet")
df_motion.to_parquet("C:/Users/Nuria/Documents/DATA/D1exp/df_data/df_motion.parquet")

from preprocess.process_data import run_all_experiments

# TODO OTHERS
# simulate BMI after doing the motion correction (For HOLOBMI)

# TODO ANALYSIS
# check if the chance of silence at the beginning of the experiment is not by chance. the increase is real. MEaning
# there are not trains/bouts
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

# run the behavior before turning the uv light to see if there is lickage of room light
# brief summary for the increase in activity in cortex
# removing tone before or after high performance
# is it worth it the randomstim no audio
# how many animals we need for this.

###########################################################
# MOTION
motion_data = pd.read_parquet(folder_save / 'motion_data.parquet')
motion_behavior = pd.read_parquet(folder_save / 'motion_behavior.parquet')
folder_plots = Path('F:/data/process/plots/learning')



###
folder_temp_save: str = 'C:/Users/Nuria/Documents/DATA/D1exp'
folder_temp_save = Path(folder_temp_save)
default_path = folder_temp_save / "default_var"


###
folder_processed_experiment = Path('D:/data/process/m16/221113/D02')
folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
stim_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
stim_time_dict = stim_aux.take(0)
stim_index = stim_time_dict['stim_time']
spks = np.load(Path(folder_suite2p) / "spks.npy")
is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
dff = pp.obtain_dffs(folder_suite2p)
aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
direct_neurons = aux_dn.take(0)
ensemble = direct_neurons['E1'] + direct_neurons['E2']
indirect_neurons = copy.deepcopy(is_cell)
indirect_neurons[ensemble, :] = [0, 0]

dff_tl = pp.create_time_locked_array(dff, stim_index, (60, 60))
spks_tl = pp.create_time_locked_array(spks, stim_index, (60, 60))
spks_dn = spks_tl[ensemble, :, :]
spks_in = spks_tl[indirect_neurons[:, 0].astype(bool), :, :]
spks_dn_av = ut.sum_array_samples(np.reshape(spks_dn, (spks_dn.shape[0], np.prod(spks_dn.shape[1:]))), 1, 3)
spks_in_av = ut.sum_array_samples(np.reshape(spks_in, (spks_in.shape[0], np.prod(spks_in.shape[1:]))), 1, 3)



folder_suite2p = Path('D:/data/process/m16/221113/D02') / 'suite2p' / 'plane0'


AnalysisConfiguration.FA_event_frames = 30
AnalysisConfiguration.FA_rew_frames = 0
df_SOT_line = dp.obtain_population_SOT_line(folder_list)
df_SOT = dp.obtain_population_SOT(folder_list)
AnalysisConfiguration.FA_time_win = 5
df5 = dp.obtain_population_SOT_windows(folder_list)

AnalysisConfiguration.eng_event_frames = 15
AnalysisConfiguration.FA_rew_frames = 0
df_rcv_line = dp.obtain_population_engagement(folder_list, line_flag=True)
df_rcv = dp.obtain_population_engagement(folder_list)

AnalysisConfiguration.eng_event_frames = 60
AnalysisConfiguration.FA_rew_frames = 60
df_rcv60 = dp.obtain_population_engagement(folder_list)

