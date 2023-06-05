__author__ = 'Nuria'

import pandas as pd
import seaborn as sns
import numpy as np

from pathlib import Path
from matplotlib import interactive

from utils import util_plots as ut_plots
interactive(True)


def plot_movement_features(df_motion: pd.DataFrame, folder_plots: Path):
    """ Function to plot all features of movement """

    mice = df_motion.mice.unique()
    for feature in df_motion.columns[4:len(df_motion.columns) - 1]:
        fig1, ax1 = ut_plots.open_plot()
        sns.boxplot(data=df_motion, x='Laser', y=feature, ax=ax1)
        a = df_motion[df_motion.Laser == 'ON'][feature]
        b = df_motion[df_motion.Laser == 'OFF'][feature]
        c = df_motion[df_motion.Laser == 'BMI'][feature]
        ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(a, c, ax1, pos=1, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(b, c, ax1, pos=1.5, height=a[~np.isnan(a)].max()*1.2, ind=True)
        ut_plots.save_plot(fig1, ax1, folder_plots, 'behavior_experiment', feature, False)
        fig2, ax2 = ut_plots.open_plot()
        sns.boxplot(data=df_motion, x='mice', y=feature, hue='Laser', ax=ax2)
        for mm, mouse in enumerate(mice):
            mpm = df_motion[df_motion.mice == mouse]
            ut_plots.get_pvalues(mpm[mpm.Laser == 'ON'][feature],
                                 mpm[mpm.Laser == 'OFF'][feature],
                                 ax2, pos=mm, height=mpm[feature].dropna().mean(), ind=True)
        ut_plots.save_plot(fig2, ax2, folder_plots, 'mice', feature, False)
        # Take into account that total values are dependent on size of experiment, so only features per min should be
        # use for this part of the analysis


def plot_controls(df_motion_controls: pd.DataFrame, folder_plots: Path):
    mice = df_motion_controls.mice.unique()

    for feature in df_motion_controls.columns[4:]:
        fig1, ax1 = ut_plots.open_plot()
        sns.boxplot(data=df_motion_controls, x='experiment', y=feature, ax=ax1)
        a = df_motion_controls[df_motion_controls.experiment == 'D1act'][feature]
        b = df_motion_controls[df_motion_controls.experiment == 'RANDOM'][feature]
        c = df_motion_controls[df_motion_controls.experiment == 'CONTROL'][feature]
        ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(a, c, ax1, pos=1, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(b, c, ax1, pos=1.5, height=a[~np.isnan(a)].max()*1.2, ind=True)
        ut_plots.save_plot(fig1, ax1, folder_plots, 'behavior_experiment', feature, False)
        fig2, ax2 = ut_plots.open_plot()
        sns.boxplot(data=df_motion_controls, x='mice', y=feature, hue='experiment', ax=ax2)
        for mm, mouse in enumerate(mice):
            mpm = df_motion_controls[df_motion_controls.mice == mouse]
            ut_plots.get_pvalues(mpm[mpm.experiment == 'D1act'][feature],
                                 mpm[mpm.experiment == 'RANDOM'][feature],
                                 ax2, pos=mm, height=mpm[feature].dropna().mean(), ind=True)
        ut_plots.save_plot(fig2, ax2, folder_plots, 'mice', feature, False)
        # Take into account that total values are dependent on size of experiment, so only features per min should be
        # use for this part of the analysis
