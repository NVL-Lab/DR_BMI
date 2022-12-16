__author__ = 'Nuria'

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from pathlib import Path
from matplotlib import interactive

from utils import util_plots as ut_plots
from utils.analysis_constants import AnalysisConstants

interactive(True)


def plot_learning(df: pd.DataFrame, folder_plots: Path):
    """ function to plot learning stats """
    mice = df.mice.unique()
    experiments = df.experiment.unique()
    for metric in df.columns[3:]:
        fig1, ax1 = ut_plots.open_plot()
        df_group = df.groupby(["mice", "experiment"]).mean().sort_values('experiment').reset_index()
        sns.boxplot(data=df_group, x='experiment', y=metric, ax=ax1)
        a = df_group[df_group.experiment == 'BMI_STIM_AGO'][metric]
        b = df_group[df_group.experiment == 'BMI_CONTROL_RANDOM'][metric]
        c = df_group[df_group.experiment == 'BMI_CONTROL_LIGHT'][metric]
        d = df_group[df_group.experiment == 'BMI_CONTROL_AGO'][metric]
        ut_plots.get_pvalues(a, d, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=False)
        ut_plots.get_pvalues(a, c, ax1, pos=1.5, height=a[~np.isnan(a)].max(), ind=False)
        ut_plots.get_pvalues(a, b, ax1, pos=2.5, height=a[~np.isnan(a)].max(), ind=False)
        ut_plots.save_plot(fig1, ax1, folder_plots, metric, 'av_mice', False)

        for mm, mouse in enumerate(mice):
            dfm = df[df.mice == mouse].sort_values('experiment').reset_index()
            fig2, ax2 = ut_plots.open_plot()
            sns.boxplot(data=dfm, x='experiment', y=metric, ax=ax2)
            a = dfm[dfm.experiment == 'BMI_STIM_AGO'][metric]
            b = dfm[dfm.experiment == 'BMI_CONTROL_RANDOM'][metric]
            c = dfm[dfm.experiment == 'BMI_CONTROL_LIGHT'][metric]
            d = dfm[dfm.experiment == 'BMI_CONTROL_AGO'][metric]
            ut_plots.save_plot(fig2, ax2, folder_plots, metric, mouse, False)
        # Take into account that total values are dependent on size of experiment, so only features per min should be
        # use for this part of the analysis


def plot_performance_sessions(df_performance: pd.DataFrame, metric: str, folder_plots: Path):
    """ function to check if there is a difference in performance for different sessions on same day """
    df_performance['first_session'] = False
    df_performance.loc[df_performance.session_day == "1st", 'first_session'] = True
    for experiment_type in AnalysisConstants.experiment_types:
        fig1, ax1 = ut_plots.open_plot()
        df_experiment_type = df_performance[df_performance.experiment == experiment_type]
        sns.boxplot(data=df_experiment_type, x='first_session', y=metric, ax=ax1)
        ut_plots.get_pvalues(df_experiment_type[df_experiment_type.first_session][metric],
                             df_experiment_type[~df_experiment_type.first_session][metric],
                             ax1, pos=0.5, height=df_experiment_type[metric].mean(), ind=True)
        ut_plots.save_plot(fig1, ax1, folder_plots,
                           'Differences_same_day_session_' + experiment_type, metric, False)
        fig2, ax2 = ut_plots.open_plot()
        previous_experiments = df_experiment_type[df_experiment_type.previous_session!="None"].previous_session.unique()
        sns.boxplot(data=df_experiment_type, x='previous_session', y=metric, ax=ax2)
        for pe, pre_ses in enumerate(previous_experiments):
            ut_plots.get_pvalues(df_experiment_type[df_experiment_type.previous_session == pre_ses][metric],
                                 df_experiment_type[df_experiment_type.previous_session == 'None'][metric],
                                 ax2, pos=pe + 0.5, height=df_experiment_type[metric].mean(), ind=True)
        ut_plots.save_plot(fig2, ax2, folder_plots,
                           'Differences_same_day_session_experiment_' + experiment_type, metric, False)

