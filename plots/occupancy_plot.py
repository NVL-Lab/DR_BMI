__author__ = 'Nuria'

import pandas as pd
import seaborn as sns
import numpy as np

from pathlib import Path
from utils import util_plots as ut_plots
from utils.utils_analysis import geometric_mean, harmonic_mean
from utils.analysis_constants import AnalysisConstants

def plot_occupancy(df_occupancy, folder_plots: Path):


    """ Function to plot occupancy and hits from simulated BMI """
    basic_columns = ["mice", "session_date", "session_path", "experiment"]
    occupancy_columns = [col for col in df_occupancy.columns if "occupancy" in col]
    for experiment_type in AnalysisConstants.experiment_types:
        df_subset = df_occupancy[basic_columns + occupancy_columns].copy()
        df_subset = df_subset[df_subset.experiment == experiment_type]
        df_fig1 = df_subset.melt(id_vars=basic_columns, var_name='occupancy', value_name='value')
        fig1, ax1 = ut_plots.open_plot()
        df_group = df_fig1.groupby(["mice", "occupancy"]).apply(harmonic_mean, 'value').reset_index()
        sns.boxplot(data=df_group, x='occupancy', y='value', color='gray', ax=ax1)
        ax1.set_ylim([0.15, 3])
        ax1.set_xlabel(experiment_type)
        a = df_group[df_group.experiment == 'D1act']['gain']
        b = df_group[df_group.experiment == 'CONTROL']['gain']
        ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.save_plot(fig1, ax1, folder_plots, 'control_basic', 'av_mice', False)





    df_occupancy = pd.read_parquet(Path(folder_save) / "df_occupancy.parquet")
    df_occupancy = df_occupancy[df_occupancy.did_simulation_worked]
    df_occupancy = df_occupancy.dropna()
    df_occupancy['T1_gain'] = df_occupancy.hits_T1 / df_occupancy.hits_baseline_T1
    df_occupancy['T1_rel_gain'] = df_occupancy.rel_hits_T1 / df_occupancy.rel_hits_baseline_T1
    df_occupancy['T1_occupancy'] = df_occupancy.occupancy_T1 / df_occupancy.occupancy_baseline_T1
    df_occupancy['T1_rel_occupancy'] = df_occupancy.rel_occupancy_T1 / df_occupancy.rel_occupancy_baseline_T1
    df_occupancy['T2_gain'] = df_occupancy.hits_T2 / df_occupancy.hits_baseline_T2
    df_occupancy['T2_rel_gain'] = df_occupancy.rel_hits_T2 / df_occupancy.rel_hits_baseline_T2
    df_occupancy['T2_occupancy'] = df_occupancy.occupancy_T2 / df_occupancy.occupancy_baseline_T2
    df_occupancy['T2_rel_occupancy'] = df_occupancy.rel_occupancy_T2 / df_occupancy.rel_occupancy_baseline_T2
    df_occupancy['rel_occupancy'] = df_occupancy.T1_rel_occupancy / df_occupancy.T2_rel_occupancy
    df_occupancy['rel_gain'] = df_occupancy.T1_rel_gain / df_occupancy.T2_rel_gain
    df_aux_BMI_T1 = df_occupancy[
        df_occupancy.columns[np.concatenate((np.arange(0, 5),
                                             np.where(df_occupancy.columns.isin(["hits_T1",
                                                                                 "rel_hits_T1",
                                                                                 "occupancy_T1",
                                                                                 "rel_occupancy_T1"]))[0]))]]
    df_aux_BMI_T2 = df_occupancy[
        df_occupancy.columns[np.concatenate((np.arange(0, 5),
                                             np.where(df_occupancy.columns.isin(["hits_T2",
                                                                                 "rel_hits_T2",
                                                                                 "occupancy_T2",
                                                                                 "rel_occupancy_T2"]))[0]))]]
    df_aux_baseline_T1 = df_occupancy[
        df_occupancy.columns[np.concatenate((np.arange(0, 5),
                                             np.where(df_occupancy.columns.isin(["hits_baseline_T1",
                                                                                 "rel_hits_baseline_T1",
                                                                                 "occupancy_baseline_T1",
                                                                                 "rel_occupancy_baseline_T1"]))[0]))]]
    df_aux_baseline_T2 = df_occupancy[
        df_occupancy.columns[np.concatenate((np.arange(0, 5),
                                             np.where(df_occupancy.columns.isin(["hits_baseline_T2",
                                                                                 "rel_hits_baseline_T2",
                                                                                 "occupancy_baseline_T2",
                                                                                 "rel_occupancy_baseline_T2"]))[0]))]]
    df_aux_T1 = df_occupancy[
        df_occupancy.columns[np.concatenate((np.arange(0, 5),
                                             np.where(df_occupancy.columns.isin(["hits_T1",
                                                                                 "rel_hits_T1",
                                                                                 "occupancy_T1",
                                                                                 "rel_occupancy_T1",
                                                                                 "hits_baseline_T1",
                                                                                 "rel_hits_baseline_T1",
                                                                                 "occupancy_baseline_T1",
                                                                                 "rel_occupancy_baseline_T1"]))[0]))]]
    df_aux_T2 = df_occupancy[
        df_occupancy.columns[np.concatenate((np.arange(0, 5),
                                             np.where(df_occupancy.columns.isin(["hits_T2",
                                                                                 "rel_hits_T2",
                                                                                 "occupancy_T2",
                                                                                 "rel_occupancy_T2",
                                                                                 "hits_baseline_T2",
                                                                                 "rel_hits_baseline_T2",
                                                                                 "occupancy_baseline_T2",
                                                                                 "rel_occupancy_baseline_T2"]))[0]))]]
    df_aux_BMI_T1['Target'] = 'T1'
    df_aux_BMI_T1['Part'] = 'BMI'
    df_aux_baseline_T1['Target'] = 'T1'
    df_aux_baseline_T1['Part'] = 'baseline'
    df_aux_BMI_T2['Target'] = 'T2'
    df_aux_BMI_T2['Part'] = 'BMI'
    df_aux_baseline_T2['Target'] = 'T2'
    df_aux_baseline_T2['Part'] = 'baseline'
    df_aux_T1['Target'] = 'T1'
    df_aux_T2['Target'] = 'T2'
    df_aux_BMI_T1 = df_aux_BMI_T1.rename(
        columns={"hits_T1": "hits", "rel_hits_T1": "rel_hits", "occupancy_T1": "occupancy",
                 "rel_occupancy_T1": "rel_occupancy"})
    df_aux_BMI_T2 = df_aux_BMI_T2.rename(columns={"hits_T2": "hits", "rel_hits_T2": "rel_hits",
                                                  "occupancy_T2": "occupancy", "rel_occupancy_T2": "rel_occupancy"})
    df_aux_baseline_T1 = df_aux_baseline_T1.rename(columns={"hits_baseline_T1": "hits",
                                                            "rel_hits_baseline_T1": "rel_hits",
                                                            "occupancy_baseline_T1": "occupancy",
                                                            "rel_occupancy_baseline_T1": "rel_occupancy"})
    df_aux_baseline_T2 = df_aux_baseline_T2.rename(columns={"hits_baseline_T2": "hits",
                                                            "rel_hits_baseline_T2": "rel_hits",
                                                            "occupancy_baseline_T2": "occupancy",
                                                            "rel_occupancy_baseline_T2": "rel_occupancy"})
    df_aux_T1 = df_aux_T1.rename(
        columns={"hits_T1": "hits", "rel_hits_T1": "rel_hits", "occupancy_T1": "occupancy_BMI",
                 "rel_occupancy_T1": "rel_occupancy_BMI", "hits_baseline_T1": "hits_baseline",
                 "rel_hits_baseline_T1": "rel_hits_baseline", "occupancy_baseline_T1": "occupancy_baseline",
                 "rel_occupancy_baseline_T1": "rel_occupancy_baseline"})
    df_aux_T2 = df_aux_T2.rename(columns={"hits_T2": "hits", "rel_hits_T2": "rel_hits",
                                          "occupancy_T2": "occupancy_BMI", "rel_occupancy_T2": "rel_occupancy_BMI",
                                          "hits_baseline_T2": "hits_baseline",
                                          "rel_hits_baseline_T2": "rel_hits_baseline",
                                          "occupancy_baseline_T2": "occupancy_baseline",
                                          "rel_occupancy_baseline_T2": "rel_occupancy_baseline"})
    df_original = pd.concat((df_aux_BMI_T1, df_aux_BMI_T2, df_aux_baseline_T1, df_aux_baseline_T2))
    df_original = df_original.dropna()
    df_target = pd.concat((df_aux_T1, df_aux_T2))
    df_target['gain'] = df_target.hits / df_target.hits_baseline
    df_target['rel_gain'] = df_target.rel_hits / df_target.rel_hits_baseline
    df_target['occupancy'] = df_target.occupancy_BMI / df_target.occupancy_baseline
    df_target['rel_occupancy'] = df_target.rel_occupancy_BMI / df_target.rel_occupancy_baseline

    ## plot original values
    df_original_group = df_original.groupby(
        ["mice_name", "experiment_type", 'Target', 'Part']).mean().sort_values('experiment_type').reset_index()
    for metric in ['rel_occupancy', 'rel_hits']:
        fig1, ax1 = ut_plots.open_plot()
        sns.boxplot(data=df_original_group[df_original_group.Target == 'T1'], x='experiment_type', y=metric, hue='Part',
                    ax=ax1)
        for ee, experiment_type in enumerate(df_original_group.experiment_type.unique()):
            a1 = df_original_group[np.logical_and(df_original_group.experiment_type == experiment_type,
                                                  df_original_group.Part == 'baseline')][metric]
            a2 = df_original_group[np.logical_and(df_original_group.experiment_type == experiment_type,
                                                  df_original_group.Part == 'BMI')][metric]

            ut_plots.get_pvalues(a1, a2, ax1, pos=0.1 + ee, height=a1[~np.isnan(a1)].max(), ind=False)
        ut_plots.save_plot(fig1, ax1, folder_plots, metric, 'av_mice_original', False)
        fig2, ax2 = ut_plots.open_plot()
        sns.boxplot(data=df_original[df_original.Target == 'T1'], x='experiment_type', y=metric, hue='Part', ax=ax2)
        for ee, experiment_type in enumerate(df_original_group.experiment_type.unique()):
            a1 = df_original[np.logical_and(df_original.experiment_type == experiment_type,
                                            df_original.Part == 'baseline')][metric]
            a2 = df_original[np.logical_and(df_original.experiment_type == experiment_type,
                                            df_original.Part == 'BMI')][metric]
            ut_plots.get_pvalues(a1, a2, ax2, pos=0.1 + ee, height=a1[~np.isnan(a1)].max(), ind=False)
        ut_plots.save_plot(fig2, ax2, folder_plots, metric, 'original', False)

    ## plot div
    df_group = df_occupancy.groupby(["mice_name", "experiment_type"]).mean().sort_values(
        'experiment_type').reset_index()

    for metric in df_occupancy.columns[22:]:
        fig1, ax1 = ut_plots.open_plot()
        sns.boxplot(data=df_group, x='experiment_type', y=metric, ax=ax1)
        a = df_group[df_group.experiment_type == 'BMI_STIM_AGO'][metric]
        for ee, experiment_type in enumerate(df_occupancy.experiment_type.unique()):
            b = df_group[df_group.experiment_type == experiment_type][metric]
            ut_plots.get_pvalues(a, b, ax1, pos=0.1 + ee, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.save_plot(fig1, ax1, folder_plots, metric, 'av_mice', False)

        fig2, ax2 = ut_plots.open_plot()
        df_aux = df_occupancy.dropna()
        sns.boxplot(data=df_aux, x='experiment_type', y=metric, ax=ax2)
        a = df_aux[df_aux.experiment_type == 'BMI_STIM_AGO'][metric]
        for ee, experiment_type in enumerate(df_occupancy.experiment_type.unique()):
            b = df_aux[df_aux.experiment_type == experiment_type][metric]
            ut_plots.get_pvalues(a, b, ax2, pos=0.1 + ee, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.save_plot(fig2, ax2, folder_plots, metric, '', False)

    ## plot T1 vs T2
    df_target = df_target.dropna()
    df_group_T1T2 = df_target.groupby(["mice_name", "experiment_type", 'Target']).mean().sort_values(
        'experiment_type').reset_index()

    for metric in df_group_T1T2.columns[3:]:
        fig1, ax1 = ut_plots.open_plot()
        sns.boxplot(data=df_group_T1T2, x='experiment_type', y=metric, hue='Target', ax=ax1)
        for ee, experiment_type in enumerate(df_group_T1T2.experiment_type.unique()):
            a1 = df_group_T1T2[np.logical_and(df_group_T1T2.experiment_type == experiment_type,
                                              df_group_T1T2.Target == 'T1')][metric]
            a2 = df_group_T1T2[np.logical_and(df_group_T1T2.experiment_type == experiment_type,
                                              df_group_T1T2.Target == 'T2')][metric]
            ut_plots.get_pvalues(a1, a2, ax1, pos=0.1 + ee, height=a1[~np.isnan(a1)].max(), ind=True)
        ut_plots.save_plot(fig1, ax1, folder_plots, metric, 'target_av_mice', False)

        fig2, ax2 = ut_plots.open_plot()
        df_aux = df_target.dropna()
        sns.boxplot(data=df_aux, x='experiment_type', y=metric, hue='Target', ax=ax2)
        for ee, experiment_type in enumerate(df_group_T1T2.experiment_type.unique()):
            a1 = df_aux[np.logical_and(df_aux.experiment_type == experiment_type,
                                       df_aux.Target == 'T1')][metric]
            a2 = df_aux[np.logical_and(df_aux.experiment_type == experiment_type,
                                       df_aux.Target == 'T2')][metric]
            ut_plots.get_pvalues(a1, a2, ax2, pos=0.1 + ee, height=a1[~np.isnan(a1)].max(), ind=True)
        ut_plots.save_plot(fig2, ax2, folder_plots, metric, 'target', False)

    # all goes
    df_aux = df_occupancy[df_occupancy.experiment_type.isin(['BMI_STIM_AGO', 'BMI_CONTROL_RANDOM'])]
    df_aux = df_aux[df_aux.columns[[0, 1, 3, 30, 31]]]
    df_aux = df_aux.dropna()
    df_aux = df_aux.groupby(["mice_name", "experiment_type"], dropna=True).mean().sort_values('experiment_type').reset_index()

    for metric in ['rel_gain', 'rel_occupancy']:
        fig1, ax1 = ut_plots.open_plot()
        sns.boxplot(data=df_aux, x='experiment_type', y=metric, ax=ax1)
        a1 = df_aux[df_aux.experiment_type == 'BMI_STIM_AGO'][metric]
        a2 = df_aux[df_aux.experiment_type == 'BMI_CONTROL_RANDOM'][metric]
        ut_plots.get_pvalues(a1, a2, ax1, pos=0.5, height=a1[~np.isnan(a1)].max(), ind=True)
        ut_plots.save_plot(fig2, ax2, folder_plots, metric, 'T1T2', False)



