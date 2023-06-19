__author__ = 'Nuria'

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from pathlib import Path
from matplotlib import interactive

from utils.util_plots import generate_palette_all_figures
from utils.utils_analysis import harmonic_mean, geometric_mean, remove_bad_mice
from utils import util_plots as ut_plots
from utils.analysis_constants import AnalysisConstants
from analysis.learning_population import get_bad_mice

interactive(True)


def plot_learning(df: pd.DataFrame, folder_plots: Path):
    """ function to plot learning stats """
    bad_mice, average_control, _ = get_bad_mice(df)
    mice = df.mice.unique()
    good_mice = np.setdiff1d(mice, bad_mice)
    df = remove_bad_mice(df, bad_mice)
    df = df.dropna()
    color_mapping = generate_palette_all_figures()

    fig0, ax0 = ut_plots.open_plot()
    experiments = ['D1act', 'CONTROL']
    df_fig0 = df[df.experiment.isin(experiments)]
    df_group = df_fig0.groupby(["mice", "experiment"]).apply(geometric_mean, 'gain').sort_values('experiment').reset_index()
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=experiments, ax=ax0)
    sns.stripplot(data=df_group, x='experiment', y='gain', hue='mice', order=experiments, palette=color_mapping, ax=ax0)
    ax0.set_ylim([0.15, 3])
    a = df_group[df_group.experiment == 'D1act']['gain']
    b = df_group[df_group.experiment == 'CONTROL']['gain']
    ut_plots.get_pvalues(a, b, ax0, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
    ut_plots.save_plot(fig0, ax0, folder_plots, 'control_basic', 'av_mice', False)

    fig1, ax1 = ut_plots.open_plot()
    experiments = ['D1act', 'CONTROL_AGO', 'CONTROL_LIGHT']
    df_fig1 = df[df.experiment.isin(experiments)]
    df_group = df_fig1.groupby(["mice", "experiment"]).apply(geometric_mean, 'gain').reset_index()
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=experiments, ax=ax1)
    sns.stripplot(data=df_group, x='experiment', y='gain', hue='mice', order=experiments, palette=color_mapping, ax=ax1)
    plt.axhline(y=average_control, color='#990000', linestyle='--')
    ax1.set_ylim([0.15, 3])
    a = df_group[df_group.experiment == 'D1act']['gain']
    b = df_group[df_group.experiment == 'CONTROL_AGO']['gain']
    c = df_group[df_group.experiment == 'CONTROL_LIGHT']['gain']
    ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
    ut_plots.get_pvalues(a, c, ax1, pos=1.5, height=a[~np.isnan(a)].max(), ind=True)
    ut_plots.get_pvalues(b, c, ax1, pos=1.8, height=a[~np.isnan(a)].max(), ind=True)
    ut_plots.save_plot(fig1, ax1, folder_plots, 'controls', 'av_mice', False)

    fig2, ax2 = ut_plots.open_plot()
    experiments = ['D1act', 'DELAY', 'RANDOM']
    df_fig2 = df[df.experiment.isin(experiments)]
    df_group = df_fig2.groupby(["mice", "experiment"]).apply(geometric_mean, 'gain').reset_index()
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=experiments, ax=ax2)
    sns.stripplot(data=df_group, x='experiment', y='gain', hue='mice', order=experiments, palette=color_mapping, ax=ax2)
    plt.axhline(y=average_control, color='#990000', linestyle='--')
    ax2.set_ylim([0.15, 3])
    a = df_group[df_group.experiment == 'D1act']['gain']
    b = df_group[df_group.experiment == 'DELAY']['gain']
    c = df_group[df_group.experiment == 'RANDOM']['gain']
    ut_plots.get_pvalues(a, b, ax2, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
    ut_plots.get_pvalues(a, c, ax2, pos=1.5, height=a[~np.isnan(a)].max(), ind=True)
    ut_plots.get_pvalues(b, c, ax2, pos=1.8, height=a[~np.isnan(a)].max(), ind=True)
    ut_plots.save_plot(fig2, ax2, folder_plots, 'timing', 'av_mice', False)

    fig3, ax3 = ut_plots.open_plot()
    experiments = ['D1act', 'NO_AUDIO']
    df_fig3 = df[df.experiment.isin(experiments)]
    df_group = df_fig3.groupby(["mice", "experiment"]).apply(geometric_mean, 'gain').reset_index()
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=experiments, ax=ax3)
    sns.stripplot(data=df_group, x='experiment', y='gain', hue='mice', order=experiments, palette=color_mapping, ax=ax3)
    plt.axhline(y=average_control, color='#990000', linestyle='--')
    ax3.set_ylim([0.15, 3])
    a = df_group[df_group.experiment == 'D1act']['gain']
    b = df_group[df_group.experiment == 'NO_AUDIO']['gain']
    ut_plots.get_pvalues(a, b, ax3, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
    ut_plots.save_plot(fig3, ax3, folder_plots, 'feedback', 'av_mice', False)

    # only same mice
    fig6, ax6 = ut_plots.open_plot()
    experiment = ['D1act', 'CONTROL_AGO', 'CONTROL_LIGHT']
    df_fig6 = df[df.experiment.isin(experiment)]
    all_experiment = set(df_fig6['experiment'])
    grouped = df_fig6.groupby('mice')['experiment'].nunique()
    selected_mice = grouped[grouped == len(all_experiment)].index
    # Select the entries in the 'mice' column that match the selected mice
    sub_df = df_fig6[df_fig6['mice'].isin(selected_mice)]
    df_group = sub_df.groupby(["mice", "experiment"]).apply(geometric_mean, 'gain').reset_index()
    df_group = df_group.sort_values(by='experiment', key=lambda x: x.map({value: i for i, value in enumerate(experiment)}))
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=experiment, ax=ax6)
    sns.lineplot(data=df_group, x='experiment', y='gain', hue='mice', palette=color_mapping, ax=ax6)
    plt.axhline(y=average_control, color='#990000', linestyle='--')
    ax6.set_ylim([0.15, 3])
    a = df_group[df_group.experiment == 'D1act']['gain']
    b = df_group[df_group.experiment == 'CONTROL_AGO']['gain']
    c = df_group[df_group.experiment == 'CONTROL_LIGHT']['gain']
    ut_plots.get_pvalues(a, b, ax6, pos=0.5, height=a[~np.isnan(a)].max())
    ut_plots.get_pvalues(a, c, ax6, pos=1.5, height=a[~np.isnan(a)].max())
    ut_plots.get_pvalues(b, c, ax6, pos=1.8, height=a[~np.isnan(a)].max())
    ut_plots.save_plot(fig6, ax6, folder_plots, 'control_same_mice', 'av_mice', False)

    fig4, ax4 = ut_plots.open_plot()
    experiment = ['D1act', 'RANDOM', 'DELAY']
    df_fig4 = df[df.experiment.isin(experiment)]
    all_experiment = set(df_fig4['experiment'])
    grouped = df_fig4.groupby('mice')['experiment'].nunique()
    selected_mice = grouped[grouped == len(all_experiment)].index
    # Select the entries in the 'mice' column that match the selected mice
    sub_df = df_fig4[df_fig4['mice'].isin(selected_mice)]
    df_group = sub_df.groupby(["mice", "experiment"]).apply(geometric_mean, 'gain').reset_index()
    order_fig4 = ['D1act', 'DELAY', 'RANDOM']
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=order_fig4, ax=ax4)
    sns.lineplot(data=df_group, x='experiment', y='gain', hue='mice', palette=color_mapping, ax=ax4)
    plt.axhline(y=average_control, color='#990000', linestyle='--')
    ax4.set_ylim([0.15, 3])
    a = df_group[df_group.experiment == 'D1act']['gain']
    b = df_group[df_group.experiment == 'DELAY']['gain']
    c = df_group[df_group.experiment == 'RANDOM']['gain']
    ut_plots.get_pvalues(a, b, ax4, pos=0.5, height=a[~np.isnan(a)].max())
    ut_plots.get_pvalues(a, c, ax4, pos=1.5, height=a[~np.isnan(a)].max())
    ut_plots.get_pvalues(b, c, ax4, pos=1.8, height=a[~np.isnan(a)].max())
    ut_plots.save_plot(fig4, ax4, folder_plots, 'timing_same_mice', 'av_mice', False)

    fig5, ax5 = ut_plots.open_plot()
    experiment = ['D1act', 'NO_AUDIO']
    df_fig5 = df[df.experiment.isin(experiment)]
    all_experiment = set(df_fig3['experiment'])
    grouped = df_fig5.groupby('mice')['experiment'].nunique()
    selected_mice = grouped[grouped == len(all_experiment)].index
    # Select the entries in the 'mice' column that match the selected mice
    sub_df = df_fig5[df_fig5['mice'].isin(selected_mice)]
    df_group = sub_df.groupby(["mice", "experiment"]).apply(geometric_mean, 'gain').reset_index()
    order_fig5 = ['D1act', 'NO_AUDIO']
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=order_fig5, ax=ax5)
    sns.lineplot(data=df_group, x='experiment', y='gain', hue='mice', palette=color_mapping, ax=ax5)
    plt.axhline(y=average_control, color='#990000', linestyle='--')
    ax5.set_ylim([0.15, 3])
    a = df_group[df_group.experiment == 'D1act']['gain']
    b = df_group[df_group.experiment == 'NO_AUDIO']['gain']
    ut_plots.get_pvalues(a, b, ax5, pos=0.5, height=a[~np.isnan(a)].max())
    ut_plots.save_plot(fig5, ax5, folder_plots, 'feedback_same_mice', 'av_mice', False)

    mice = df.mice.unique()
    for mm, mouse in enumerate(mice):
        dfm = df[df.mice == mouse].sort_values('experiment').reset_index()
        fig4, ax4 = ut_plots.open_plot()
        sns.boxplot(data=dfm, x='experiment', y='gain', ax=ax4)
        ut_plots.save_plot(fig4, ax4, folder_plots, 'gain', mouse, False)
        # Take into account that total values are dependent on size of experiment, so only features per min should be
        # use for this part of the analysis

    for signal in ['hit_array', 'time_to_hit']:
        max_length = df[signal].apply(len).max()
        df[signal] = df[signal].apply(lambda arr: np.pad(arr, (0, max_length- len(arr)), constant_values=np.nan))
        df_group = df.groupby(["mice", "experiment"])[signal].mean().reset_index()
        df_exp = df_group[df_group.experiment=='D1act']
        expanded_values = df_exp[signal].apply(pd.Series)
        df_cleaned = expanded_values.dropna(axis=1, how='all')
        array_data = df_cleaned.to_numpy()
        arr_1d = array_data.flatten()
        rows, cols = array_data.shape
        x = np.tile(np.arange(cols), rows)
        fig7, ax7 = ut_plots.open_plot()
        sns.regplot(x=x, y=arr_1d, x_estimator=np.nanmean, color='gray', x_bins=20, ax=ax7)
        ut_plots.get_reg_pvalues(arr_1d, x, ax7, np.nanmean(x), np.nanmean(arr_1d))
        ut_plots.save_plot(fig7, ax7, folder_plots, 'array', signal, False)


def plot_performance_sessions(df: pd.DataFrame, folder_plots: Path):
    """ function to check if there is a difference in performance for different sessions on same day """
    df_performance = df[df.session_day.isin(['1st', '2nd'])]
    df_performance['first_session'] = False
    df_performance.loc[df_performance.session_day == "1st", 'first_session'] = True
    for experiment_type in AnalysisConstants.experiment_types:
        fig1, ax1 = ut_plots.open_plot()
        df_experiment_type = df_performance[df_performance.experiment == experiment_type]
        sns.boxplot(data=df_experiment_type, x='first_session', y='gain', ax=ax1)
        ax1.set_title(experiment_type)
        ut_plots.get_pvalues(df_experiment_type[df_experiment_type.first_session]['gain'],
                             df_experiment_type[~df_experiment_type.first_session]['gain'],
                             ax1, pos=0.5, height=df_experiment_type['gain'].mean(), ind=True)
        ut_plots.save_plot(fig1, ax1, folder_plots,
                           'Differences_same_day_session_' + experiment_type, 'gain', False)
        fig2, ax2 = ut_plots.open_plot()
        previous_experiments = df_experiment_type[df_experiment_type.previous_session!="None"].previous_session.unique()
        sns.boxplot(data=df_experiment_type, x='previous_session', y='gain', ax=ax2)
        for pe, pre_ses in enumerate(previous_experiments):
            ut_plots.get_pvalues(df_experiment_type[df_experiment_type.previous_session == pre_ses]['gain'],
                                 df_experiment_type[df_experiment_type.previous_session == 'None']['gain'],
                                 ax2, pos=pe + 0.5, height=df_experiment_type['gain'].mean(), ind=True)
        ax2.set_title(experiment_type)
        ut_plots.save_plot(fig2, ax2, folder_plots,
                           'Differences_same_day_session_experiment_' + experiment_type, 'gain', False)


def plot_across_day_learning(df: pd.DataFrame, folder_plots: Path):
    """ Function to plot learning over days """
    df_group = df[df.experiment == 'D1act']
    color_mapping = generate_palette_all_figures()

    fig1, ax1 = ut_plots.open_plot()
    sns.stripplot(data=df_group, x="day_index", y='gain', hue='mice', palette=color_mapping, ax=ax1)
    a = df_group[df_group.day_index == 0]['gain']
    b = df_group[df_group.day_index == 1]['gain']
    c = df_group[df_group.day_index == 2]['gain']
    d = df_group[df_group.day_index == 3]['gain']
    ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max())
    ut_plots.get_pvalues(a, c, ax1, pos=1.5, height=a[~np.isnan(a)].max())
    ut_plots.get_pvalues(a, d, ax1, pos=2.5, height=a[~np.isnan(a)].max())
    ut_plots.save_plot(fig1, ax1, folder_plots, 'across_days_1st', 'gain', False)


def plot_extinction(df_ext: pd.DataFrame, folder_plots: Path, bad_mice: list):
    df_fig = df_ext[~df_ext.mice.isin(bad_mice)]
    df_fig.replace("None", np.nan, inplace=True)
    df_fig.dropna()
    color_mapping = generate_palette_all_figures()

    # Iterate over each row in df_selected
    for measure in ['hpm', 'gain']:
        df_new = pd.DataFrame(columns=['experiments', 'values'])
        for index, row in df_fig.iterrows():
            mice = row['mice']
            values = row[['BMI_' + measure, 'ext_' + measure, 'ext2_' + measure]]

            # Append the values as new rows in df_new
            for experiment, value in values.iteritems():
                df_new = df_new.append({'experiments': experiment, 'values': value, 'mice': mice}, ignore_index=True)

        fig1, ax1 = ut_plots.open_plot()
        sns.lineplot(data=df_new, x="experiments", y="values", hue='mice', palette=color_mapping, ax=ax1)
        sns.stripplot(data=df_new, x='experiments', y='values', hue='mice', palette=color_mapping, jitter=False, s=10,
                      marker="D", ax=ax1)
        ut_plots.save_plot(fig1, ax1, folder_plots, 'extinction', measure, False)








