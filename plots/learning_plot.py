__author__ = 'Nuria'

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from pathlib import Path
from matplotlib import interactive

from utils.utils_analysis import geometric_mean, harmonic_mean
from utils import util_plots as ut_plots
from utils.analysis_constants import AnalysisConstants

interactive(True)


def plot_learning(df: pd.DataFrame, folder_plots: Path):
    """ function to plot learning stats """
    mice = df.mice.unique()
    df = df.dropna()
    df_control = df[df.experiment == "CONTROL"]
    df_no_control = df[df.experiment != "CONTROL"]

    # remove the bad animals
    average_control = harmonic_mean(df_control, 'gain').values[0][0]
    df_group_control = df_control.groupby(["mice", "experiment"]).apply(harmonic_mean, "gain")
    df_group = df.groupby(["mice", "experiment"]).apply(harmonic_mean, "gain").sort_values('experiment').reset_index()
    df_group_d1act = df_group[df_group.experiment=='D1act']
    deviations = df_group_control.gain - average_control
    squared_deviations = deviations ** 2
    mean_squared_deviations = np.mean(squared_deviations)
    std_harmonic = np.sqrt(mean_squared_deviations)
    bad_mice = df_group_d1act[df_group_d1act.gain < (std_harmonic + average_control)].mice.unique()
    df_good = df_no_control[~df_no_control.mice.isin(bad_mice)]
    df = pd.concat([df_good, df_control])

    fig0, ax0 = ut_plots.open_plot()
    experiments = ['D1act', 'CONTROL']
    df_fig0 = df[df.experiment.isin(experiments)]
    df_group = df_fig0.groupby(["mice", "experiment"]).apply(harmonic_mean, 'gain').sort_values('experiment').reset_index()
    order_fig0 = ['D1act', 'CONTROL']
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=order_fig0, ax=ax0)
    ax0.set_ylim([0.15, 3])
    a = df_group[df_group.experiment == 'D1act']['gain']
    b = df_group[df_group.experiment == 'CONTROL']['gain']
    ut_plots.get_pvalues(a, b, ax0, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
    ut_plots.save_plot(fig0, ax0, folder_plots, 'control_basic', 'av_mice', False)

    fig1, ax1 = ut_plots.open_plot()
    experiments = ['D1act', 'CONTROL_AGO', 'CONTROL_LIGHT']
    df_fig1 = df[df.experiment.isin(experiments)]
    df_group = df_fig1.groupby(["mice", "experiment"]).apply(harmonic_mean, 'gain').sort_values('experiment').reset_index()
    copper_palette = sns.color_palette("copper", n_colors=len(df_group.mice.unique()))
    order_fig1 = ['D1act', 'CONTROL_AGO', 'CONTROL_LIGHT']
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=order_fig1, ax=ax1)
    sns.stripplot(data=df_group, x='experiment', y='gain', hue='mice', order=order_fig1, palette=copper_palette, ax=ax1)
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
    experiments = ['D1act', 'RANDOM', 'DELAY']
    df_fig2 = df[df.experiment.isin(experiments)]
    df_group = df_fig2.groupby(["mice", "experiment"]).apply(harmonic_mean, 'gain').sort_values('experiment').reset_index()
    copper_palette = sns.color_palette("copper", n_colors=len(df_group.mice.unique()))
    order_fig2 = ['D1act', 'DELAY', 'RANDOM']
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=order_fig2, ax=ax2)
    sns.stripplot(data=df_group, x='experiment', y='gain', hue='mice', order=order_fig2, palette=copper_palette, ax=ax2)
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
    df_group = df_fig3.groupby(["mice", "experiment"]).apply(harmonic_mean, 'gain').sort_values('experiment').reset_index()
    copper_palette = sns.color_palette("copper", n_colors=len(df_group.mice.unique()))
    order_fig3 = ['D1act', 'NO_AUDIO']
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=order_fig3, ax=ax3)
    sns.stripplot(data=df_group, x='experiment', y='gain', hue='mice', order=order_fig3, palette=copper_palette, ax=ax3)
    plt.axhline(y=average_control, color='#990000', linestyle='--')
    ax3.set_ylim([0.15, 3])
    a = df_group[df_group.experiment == 'D1act']['gain']
    b = df_group[df_group.experiment == 'NO_AUDIO']['gain']
    ut_plots.get_pvalues(a, b, ax3, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
    ut_plots.save_plot(fig3, ax3, folder_plots, 'feedback', 'av_mice', False)

    # only same mice
    fig4, ax4 = ut_plots.open_plot()
    experiment = ['D1act', 'RANDOM', 'DELAY']
    df_fig4 = df[df.experiment.isin(experiment)]
    all_experiment = set(df_fig4['experiment'])
    grouped = df_fig4.groupby('mice')['experiment'].nunique()
    selected_mice = grouped[grouped == len(all_experiment)].index
    # Select the entries in the 'mice' column that match the selected mice
    sub_df = df_fig4[df_fig4['mice'].isin(selected_mice)]
    df_group = sub_df.groupby(["mice", "experiment"]).apply(harmonic_mean, 'gain').sort_values('experiment').reset_index()
    copper_palette = sns.color_palette("copper", n_colors=len(df_group.mice.unique()))
    order_fig4 = ['D1act', 'DELAY', 'RANDOM']
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=order_fig4, ax=ax4)
    sns.lineplot(data=df_group, x='experiment', y='gain', hue='mice', palette=copper_palette, ax=ax4)
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
    df_group = sub_df.groupby(["mice", "experiment"]).apply(harmonic_mean, 'gain').sort_values('experiment').reset_index()
    copper_palette = sns.color_palette("copper", n_colors=len(df_group.mice.unique()))
    order_fig5 = ['D1act', 'NO_AUDIO']
    sns.boxplot(data=df_group, x='experiment', y='gain', color='gray', order=order_fig5, ax=ax5)
    sns.lineplot(data=df_group, x='experiment', y='gain', hue='mice', palette=copper_palette, ax=ax5)
    plt.axhline(y=average_control, color='#990000', linestyle='--')
    ax5.set_ylim([0.15, 3])
    a = df_group[df_group.experiment == 'D1act']['gain']
    b = df_group[df_group.experiment == 'NO_AUDIO']['gain']
    ut_plots.get_pvalues(a, b, ax5, pos=0.5, height=a[~np.isnan(a)].max())
    ut_plots.save_plot(fig5, ax5, folder_plots, 'feedback_same_mice', 'av_mice', False)

    for mm, mouse in enumerate(mice):
        dfm = df[df.mice == mouse].sort_values('experiment').reset_index()
        fig4, ax4 = ut_plots.open_plot()
        sns.boxplot(data=dfm, x='experiment', y='gain', ax=ax4)
        ut_plots.save_plot(fig4, ax4, folder_plots, 'gain', mouse, False)
        # Take into account that total values are dependent on size of experiment, so only features per min should be
        # use for this part of the analysis


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
    df_fig = df[df.experiment == 'D1act']
    df_fig['session_prefix'] = df_fig['session_path'].str[:11]
    df_fig["average_day"] = df_fig.groupby('session_prefix')['gain'].transform('mean')
    copper_palette = sns.color_palette("copper", n_colors=len(df_fig.mice.unique()))

    fig1, ax1 = ut_plots.open_plot()
    df_fig1 = df_fig[df_fig.session_day == '1st']
    df_group = df_fig1.groupby(["mice", "day_index"]).mean().reset_index()
    df_group['day'] = df_group.groupby('mice').cumcount()
    sns.lineplot(data=df_group, x="day", y="gain", ax=ax1)
    sns.stripplot(data=df_group, x="day", y='gain', hue='mice', palette=copper_palette, ax=ax1)
    ut_plots.save_plot(fig1, ax1, folder_plots, 'across_days_1st', 'gain', False)

    fig2, ax2 = ut_plots.open_plot()
    sns.lineplot(data=df_group, x="day", y="average_day", ax=ax2)
    sns.stripplot(data=df_group, x='day', y='average_day', hue='mice', palette=copper_palette, ax=ax2)
    ut_plots.save_plot(fig2, ax2, folder_plots, 'across_average_days_1st', 'gain', False)

    fig3, ax3 = ut_plots.open_plot()
    df_group = df_fig.groupby(["mice", "day_index"]).mean().reset_index()
    sns.lineplot(data=df_group, x="day_index", y="gain", ax=ax3)
    sns.stripplot(data=df_group, x='day_index', y='gain', hue='mice', palette=copper_palette, ax=ax3)
    ut_plots.save_plot(fig3, ax3, folder_plots, 'across_sessions', 'gain', False)


def plot_extinction(df_ext: pd.DataFrame, folder_plots: Path, bad_mice: list):
    df_ext = df_ext[~df_ext.mice.isin(bad_mice)]
    df_fig = df_ext[["mice", "BMI_hpm", "ext_hpm", "ext2_hpm"]].copy()
    df_fig.replace("None", np.nan, inplace=True)
    df_fig.dropna()
    copper_palette = sns.color_palette("copper", n_colors=len(df_fig.mice.unique()))
    df_new = pd.DataFrame(columns=['experiments', 'values'])

    # Iterate over each row in df_selected
    for index, row in df_fig.iterrows():
        mice = row['mice']
        values = row[['BMI_hpm', 'ext_hpm', 'ext2_hpm']]

        # Append the values as new rows in df_new
        for experiment, value in values.iteritems():
            df_new = df_new.append({'experiments': experiment, 'values': value, 'mice': mice}, ignore_index=True)

    fig1, ax1 = ut_plots.open_plot()
    sns.lineplot(data=df_new, x="experiments", y="values", hue='mice', palette=copper_palette, ax=ax1)
    sns.stripplot(data=df_new, x='experiments', y='values', hue='mice', palette=copper_palette, jitter=False, s=10,
                  marker="D", ax=ax1)
    ax1.set_ylim([0, 2.5])
    ut_plots.save_plot(fig1, ax1, folder_plots, 'extinction', 'gain', False)








