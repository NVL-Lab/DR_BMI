__author__ = 'Nuria'

import pandas as pd
import seaborn as sns
import numpy as np

from pathlib import Path

from analysis.learning_population import get_bad_mice
from utils import util_plots as ut_plots
from utils.utils_analysis import harmonic_mean, geometric_mean, remove_bad_mice
from utils.analysis_constants import AnalysisConstants


def plot_occupancy(df_occupancy, folder_plots: Path, bad_mice: list):
    """ Function to plot occupancy and hits from simulated BMI """
    df_occupancy = df_occupancy.replace([np.inf, -np.inf], np.nan).dropna()
    df_occupancy = remove_bad_mice(df_occupancy, bad_mice)
    color_mapping = ut_plots.generate_palette_all_figures()
    basic_columns = ["mice", "session_date", "session_path", "experiment"]
    occupancy_columns = [col for col in df_occupancy.columns if "occupancy" in col and "gain" not in col]
    hit_columns = [col for col in df_occupancy.columns if "hits" in col and "gain" not in col]
    for experiment_type in AnalysisConstants.experiment_types:
        for measure in ['occupancy', 'hits']:
            if measure == 'occupancy':
                measure_columns = occupancy_columns
                ylim = 10
            else:
                measure_columns = hit_columns
                ylim = 3
            df_subset = df_occupancy[basic_columns + measure_columns].copy()
            df_subset = df_subset[df_subset.experiment == experiment_type]
            df_fig1 = df_subset.melt(id_vars=basic_columns, var_name=measure, value_name='value')
            fig1, ax1 = ut_plots.open_plot((12, 6))
            df_group = df_fig1.groupby(["mice", measure]).apply(geometric_mean, 'value').sort_values(measure).reset_index()
            sns.boxplot(data=df_group, x=measure, y='value', color='gray', order=measure_columns, ax=ax1)
            sns.stripplot(data=df_group, x=measure, y='value', hue='mice', order=measure_columns,
                          palette=color_mapping, ax=ax1)
            ax1.set_xlabel(measure + '_' + experiment_type)
            ax1.set_ylim([0, ylim])
            a = df_group[df_group[measure] == measure_columns[0]]['value']
            b = df_group[df_group[measure] == measure_columns[1]]['value']
            c = df_group[df_group[measure] == measure_columns[2]]['value']
            d = df_group[df_group[measure] == measure_columns[3]]['value']
            e = df_group[df_group[measure] == measure_columns[4]]['value']
            f = df_group[df_group[measure] == measure_columns[5]]['value']
            g = df_group[df_group[measure] == measure_columns[6]]['value']
            h = df_group[df_group[measure] == measure_columns[7]]['value']
            ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].mean(), ind=False)
            ut_plots.get_pvalues(c, d, ax1, pos=2.5, height=c[~np.isnan(c)].mean(), ind=False)
            ut_plots.get_pvalues(e, f, ax1, pos=4.5, height=e[~np.isnan(e)].mean(), ind=False)
            ut_plots.get_pvalues(g, h, ax1, pos=6.5, height=g[~np.isnan(g)].mean(), ind=False)
            ut_plots.save_plot(fig1, ax1, folder_plots, measure, experiment_type, False)

    for experiment_type in AnalysisConstants.experiment_types:
        for ratio in ['hits_gain', 'occupancy_gain']:
            ratio_columns = [col for col in df_occupancy.columns if ratio in col]
            df_ratio = df_occupancy[basic_columns + ratio_columns].copy()
            df_ratio = df_ratio[df_ratio.experiment == experiment_type]
            df_ratio = df_ratio.replace([np.inf, -np.inf], np.nan).dropna()
            df_fig2 = df_ratio.melt(id_vars=basic_columns, var_name=ratio, value_name='value')
            df_group = df_fig2.groupby(["mice", ratio]).apply(geometric_mean, 'value').reset_index()
            fig2, ax2 = ut_plots.open_plot()
            sns.lineplot(data=df_group, x=ratio, y='value', hue='mice', palette=color_mapping, ax=ax2)
            sns.stripplot(data=df_group, x=ratio, y='value', hue='mice', palette=color_mapping, ax=ax2, s=10,
                          marker="D", jitter=False)
            ax2.set_xlabel(ratio + '_' + experiment_type)
            if ratio == 'hits_gain':
                ylim = 4.5
            else:
                ylim = 7
            ax2.set_ylim([0, ylim])
            a = df_group[df_group[ratio] == ratio_columns[0]]['value']
            b = df_group[df_group[ratio] == ratio_columns[1]]['value']
            ut_plots.get_pvalues(a, b, ax2, pos=0.5, height=a[~np.isnan(a)].mean(), ind=False)
            ut_plots.save_plot(fig2, ax2, folder_plots, ratio, experiment_type, False)

    for ratio in ['gain_occupancy', 'gain_hits']:
        ratio_columns = [col for col in df_occupancy.columns if ratio in col and "cal" not in col]
        df_ratio = df_occupancy[basic_columns + ratio_columns].copy()
        experiments = ['D1act', 'CONTROL']
        df_ratio = df_ratio[df_ratio.experiment.isin(experiments)]
        df_fig3 = df_ratio.melt(id_vars=basic_columns, var_name=ratio, value_name='value')
        df_group = df_fig3.groupby(["mice", "experiment"]).apply(geometric_mean, 'value').reset_index()
        fig3, ax3 = ut_plots.open_plot((14, 6))
        sns.boxplot(data=df_group, x='experiment', y='value', color='gray', order=experiments, ax=ax3)
        sns.stripplot(data=df_group, x='experiment', y='value', hue='mice', order=experiments, palette=color_mapping,
                      ax=ax3)
        ax3.set_ylim([0, 4.5])
        ax3.set_xlabel(ratio)
        a = df_group[df_group['experiment'] == 'D1act']['value']
        b = df_group[df_group['experiment'] == 'CONTROL']['value']
        ut_plots.get_pvalues(a, b, ax3, pos=0.5, height=a[~np.isnan(a)].mean())
        ut_plots.save_plot(fig3, ax3, folder_plots, ratio, 'T1T2', False)


def plot_occupancy_good_sessions(df: pd.DataFrame, df_occupancy: pd.DataFrame):
    """ function to check the occupancy of sessions that had a good outcome (besides D1act)"""
    bad_mice, average_control, std_harmonic = get_bad_mice(df)
    color_mapping = ut_plots.generate_palette_all_figures()
    basic_columns = ["mice", "session_date", "session_path", "experiment"]
    df_gs = df[df.gain > (std_harmonic + average_control)]
    gs_occupancy = df_occupancy.merge(df_gs[["session_path"]], on="session_path", how="inner")
    gs_occupancy['experiment'] = gs_occupancy['experiment'].str.replace('.*CONTROL.*', 'CONTROL', regex=True)
    for experiment_type in gs_occupancy.experiment.unique():
        for ratio in ['hits_gain', 'occupancy_gain']:
            ratio_columns = [col for col in df_occupancy.columns if ratio in col]
            df_ratio = gs_occupancy[basic_columns + ratio_columns].copy()
            df_ratio = df_ratio[df_ratio.experiment == experiment_type]
            df_fig2 = df_ratio.melt(id_vars=basic_columns, var_name=ratio, value_name='value')
            df_group = df_fig2.groupby(["mice", ratio]).apply(geometric_mean, 'value').reset_index()
            fig2, ax2 = ut_plots.open_plot()
            sns.lineplot(data=df_group, x=ratio, y='value', hue='mice', palette=color_mapping, ax=ax2)
            sns.stripplot(data=df_group, x=ratio, y='value', hue='mice', palette=color_mapping, ax=ax2, s=10,
                          marker="D", jitter=False)
            ax2.set_xlabel(ratio + '_' + experiment_type)

            a = df_group[df_group[ratio] == ratio_columns[0]]['value']
            b = df_group[df_group[ratio] == ratio_columns[1]]['value']
            ut_plots.get_pvalues(a, b, ax2, pos=0.5, height=a[~np.isnan(a)].mean(), ind=False)





