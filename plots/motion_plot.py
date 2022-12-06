__author__ = 'Nuria'

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
from matplotlib import interactive

from utils import util_plots as ut_plots
interactive(True)


def plot_movement_features(motion_data: pd.DataFrame, motion_behavior: pd.DataFrame):
    """ Function to plot all features of movement """
    motion_BMI = motion_data[motion_data.BB == 'BMI'].drop(columns='BB')
    motion_baseline = motion_data[motion_data.BB == 'baseline'].drop(columns='BB')
    motion_baseline['experiment'] = 'baselines'
    motion_bases_behavior = pd.concat([motion_baseline, motion_behavior[motion_behavior.experiment=='Behavior_before']])
    mice = motion_bases_behavior.mice.unique()
    experiments = motion_data.experiment.unique()
    for feature in motion_behavior.columns[4:]:
        fig1, ax1 = ut_plots.open_plot()
        sns.boxplot(data=motion_bases_behavior, x='experiment', y=feature, ax=ax1)
        a = motion_bases_behavior[motion_bases_behavior.experiment == 'baselines'][feature]
        b = motion_bases_behavior[motion_bases_behavior.experiment == 'Behavior_before'][feature]
        ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
        fig2, ax2 = ut_plots.open_plot()
        sns.boxplot(data=motion_bases_behavior, x='mice', y=feature, hue='experiment', ax=ax2)
        for mm, mouse in enumerate(mice):
            mpm = motion_bases_behavior[motion_bases_behavior.mice == mouse]
            ut_plots.get_pvalues(mpm[mpm.experiment == 'baselines'][feature],
                                 mpm[mpm.experiment == 'Behavior_before'][feature],
                                 ax2, pos=mm, height=mpm[feature].dropna().mean(), ind=True)
        # Take into account that total values are dependent on size of experiment, so only features per min should be
        # use for this part of the analysis
        fig3, ax3 = ut_plots.open_plot()
        sns.boxplot(data=motion_data, x='experiment', y=feature, hue='BB', ax=ax3)
        for ee, experiment in enumerate(experiments):
            mde = motion_data[motion_data.experiment == experiment]
            ut_plots.get_pvalues(mde[mde.BB == 'baseline'][feature],
                                 mde[mde.BB == 'BMI'][feature],
                                 ax3, pos=ee, height=mde[feature].dropna().mean(), ind=True)

