
__author__ = 'Nuria'

import pandas as pd
import numpy as np
import scipy
import sklearn as sk
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib import interactive

interactive(True)


# TODO ONACID
# Learn about ONACID, can it deal with stim artifacts some other way?
# can it deal with such noisy images? or should I average across multiple files?
# Find a way to remove STIM artifacts from the data by looking at the voltage rec
# function to deal with voltage recording in general not one by one
# create a way to go through all the files for onacid and start the preprocess


# TODO OTHERS
# simulate BMI after doing the motion correction (For HOLOBMI)

# TODO ANALYSIS
# obtain the increase of hits as done before but adjusting by time of start
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
# find a package for the motion
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
