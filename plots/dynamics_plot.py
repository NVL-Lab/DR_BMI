__author__ = 'Nuria'

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from pathlib import Path
from matplotlib import interactive
from numpy.polynomial import Polynomial as P

import utils.util_plots as ut_plots
from utils.analysis_command import AnalysisConfiguration
from utils.analysis_constants import AnalysisConstants

interactive(True)


def plot_FA_spontaneous_activity(df: pd.DataFrame, folder_plots: Path):
    """ function to plot the results of the spontaenous activity"""
    color_mapping = ut_plots.generate_palette_all_figures()
    df_dim = df[df.columns[[0,1,3]]]
    df_dim = df_dim.groupby('mice').mean().reset_index()
    df_dim_melt = df_dim.melt(id_vars='mice', var_name='col', value_name='dim_value')

    # Extract 'dim' from the original column names
    df_dim_melt['dim'] = df_dim_melt['col'].str.replace('dim_', '')

    # Drop the 'col' column
    df_dim_melt.drop('col', axis=1, inplace=True)
    fig1, ax1 = ut_plots.open_plot()
    sns.lineplot(data=df_dim_melt, x='dim', y='dim_value', hue='mice', palette=color_mapping, ax=ax1)
    sns.stripplot(data=df_dim_melt, x='dim', y='dim_value', hue='mice', palette=color_mapping, s=10,
                  marker="D", jitter=False, ax=ax1)
    a = df_dim_melt[df_dim_melt.dim == 'sa']['dim_value']
    b = df_dim_melt[df_dim_melt.dim == 'd1r']['dim_value']
    ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=False)
    ut_plots.save_plot(fig1, ax1, folder_plots, 'FA_SA', 'dim', False)

    df_SOT = df[df.columns[[0,4,6]]]
    df_SOT = df_SOT.groupby('mice').mean().reset_index()
    df_SOT_melt = df_SOT.melt(id_vars='mice', var_name='col', value_name='SOT_value')

    # Extract 'dim' from the original column names
    df_SOT_melt['SOT'] = df_SOT_melt['col'].str.replace('SOT_', '')

    # Drop the 'col' column
    df_SOT_melt.drop('col', axis=1, inplace=True)
    fig2, ax2 = ut_plots.open_plot()
    sns.lineplot(data=df_SOT_melt, x='SOT', y='SOT_value', hue='mice', palette=color_mapping, ax=ax2)
    sns.stripplot(data=df_SOT_melt, x='SOT', y='SOT_value', hue='mice', palette=color_mapping, s=10,
                  marker="D", jitter=False, ax=ax2)
    a = df_SOT_melt[df_SOT_melt.SOT == 'sa']['SOT_value']
    b = df_SOT_melt[df_SOT_melt.SOT == 'd1r']['SOT_value']
    ut_plots.get_pvalues(a, b, ax2, pos=0.5, height=a[~np.isnan(a)].max(), ind=False)
    ut_plots.save_plot(fig2, ax2, folder_plots, 'FA_SA', 'SOT', False)


def plot_SOT(df: pd.DataFrame, df_learning: pd.DataFrame, folder_plots: Path):
    color_mapping = ut_plots.generate_palette_all_figures()
    # plot all over times
    for cc in df.columns[3:]:
        fig1, ax1 = ut_plots.open_plot()
        for experiment_type in df.experiment.unique():
            # ax1.plot(np.vstack(df[df.experiment == experiment_type][cc].values).T, 'gray')
            ax1.plot(np.nanmean(np.vstack(df[df.experiment == experiment_type][cc].values), 0), label=experiment_type)
        plt.xlabel(cc)
        plt.legend()

    len_array = 30
    df_d1act = df[df.experiment == 'D1act']
    col_aux = [col for col in df_d1act.columns[3:] if 'calib' not in col]
    for cc in col_aux:
        fig1, ax1 = ut_plots.open_plot()
        sot_array = np.full((len(df_d1act.mice.unique()), len_array), np.nan)
        sot_calib = np.full((len(df_d1act.mice.unique()), len_array), np.nan)
        for mm, mouse in enumerate(df.mice.unique()):
            day_values = np.vstack(df_d1act[df_d1act.mice == mouse][cc].values)
            if 'stim' in cc:
                cc_calib = cc.replace("stim", "calib")
            elif 'target' in cc:
                cc_calib = cc.replace("target", "calib")
            day_calib = np.vstack(df_d1act[df_d1act.mice == mouse][cc_calib].values)
            min_x = np.min([len_array, day_values.shape[1]])
            sot_array[mm, :min_x] = np.nanmean(day_values, 0)[:min_x]
            sot_calib[mm, :min_x] = np.nanmean(day_calib, 0)[:min_x]
            ax1.plot(sot_array[mm, :], '.', color=color_mapping[mouse])
        ax1.set_xlabel(cc)
        sns.regplot(x=np.arange(len_array), y=np.nanmean(sot_array, 0), ax=ax1)
        sns.regplot(x=np.arange(len_array), y=np.nanmean(sot_calib, 0), ax=ax1, color='lightgray')
        ut_plots.get_reg_pvalues(np.arange(len_array), np.nanmean(sot_array, 0), ax1, 1, height=np.nanmean(sot_array))

    df_a = df[df.columns[:3]]
    df_e = df[df.columns[:3]]
    df_l = df[df.columns[:3]]
    for cc in df.columns[3:]:
        df_a[cc] = df[cc].apply(lambda x: np.nanmean(x))
        df_e[cc] = df[cc].apply(lambda x: np.nanmean(x[:5]))
        df_l[cc] = df[cc].apply(lambda x: np.nanmean(x[15:]))
    df_a['period'] = 'all'
    df_e['period'] = 'early'
    df_l['period'] = 'late'
    df_av = pd.concat((df_a, df_e, df_l))

    df_time = df_a[df_a.experiment.isin(['D1act', 'CONTROL_LIGHT', 'DELAY', 'RANDOM'])]
    df_time = df_time.drop(['session_path', 'period'], axis=1)

    for col in df_time.columns:
        if 'stim' in col:
            col_divide = col.replace("stim", "calib")
        elif 'target' in col:
            col_divide = col.replace("target", "calib")
        else:
            continue
        df_time[col + '_div'] = df_time[col] / df_time[col_divide]

    df_group = df_time.groupby(['mice', 'experiment']).mean().reset_index()

    for cc in df_time.filter(like='SOT'):
        fig2, ax2 = ut_plots.open_plot()
        sns.boxplot(data=df_group, x='experiment', y=cc, order=['D1act', 'CONTROL_LIGHT', 'DELAY', 'RANDOM'], ax=ax2)
        sns.stripplot(data=df_group, x="experiment", y=cc, hue='mice',
                      order=['D1act', 'CONTROL_LIGHT', 'DELAY', 'RANDOM'], palette=color_mapping, jitter=True, ax=ax2)
        a = df_group[df_group.experiment == 'D1act'][cc]
        b = df_group[df_group.experiment == 'CONTROL_LIGHT'][cc]
        c = df_group[df_group.experiment == 'DELAY'][cc]
        d = df_group[df_group.experiment == 'RANDOM'][cc]
        ax2.set_xlabel(cc)
        ut_plots.get_pvalues(a, b, ax2, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(a, c, ax2, pos=1.5, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(a, d, ax2, pos=2.5, height=a[~np.isnan(a)].max(), ind=True)
        if 'div' in cc:
            ut_plots.get_1s_pvalues(a, 1, ax2, pos=0, height=1)
            ut_plots.get_1s_pvalues(b, 1, ax2, pos=1, height=1)
            ut_plots.get_1s_pvalues(c, 1, ax2, pos=2, height=1)
            ut_plots.get_1s_pvalues(d, 1, ax2, pos=3, height=1)

    df_av = df_av.dropna()
    df_av = df_av.drop('session_path', axis=1)
    df_group = df_av.groupby(['mice', 'experiment', 'period']).mean().reset_index()
    for cc in df_av.columns[3:]:
        for experiment in ['D1act', 'CONTROL_LIGHT']:
            fig3, ax3 = ut_plots.open_plot()
            df_exp = df_group[df_group.experiment==experiment]
            sns.boxplot(data=df_exp, x='period', y=cc, order=['all', 'early', 'late'], ax=ax3)
            sns.stripplot(data=df_exp, x="period", y=cc, hue='mice',
                          order=['all', 'early', 'late'], palette=color_mapping, jitter=True, ax=ax3)
            a = df_exp[df_exp.period == 'all'][cc]
            b = df_exp[df_exp.period == 'early'][cc]
            c = df_exp[df_exp.period == 'late'][cc]
            ax3.set_xlabel(cc + '_' + experiment)
            ut_plots.get_pvalues(a, b, ax3, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
            ut_plots.get_pvalues(a, c, ax3, pos=1.5, height=a[~np.isnan(a)].max(), ind=True)
            ut_plots.get_pvalues(b, c, ax3, pos=2.5, height=a[~np.isnan(a)].max(), ind=True)

    df_late = df_group[df_group.period == 'late'].drop('period', axis=1)
    for col in df_late.columns:
        if 'stim' in col:
            col_divide = col.replace("stim", "calib")
            df_late[col + '_div'] = df_time[col] / df_time[col_divide]

    for cc in df_late.columns[[2,3,4,5,14,15,16,17]]:
        fig4, ax4 = ut_plots.open_plot()
        sns.boxplot(data=df_late, x='experiment', y=cc, order=['D1act', 'CONTROL_LIGHT', 'DELAY', 'RANDOM'], ax=ax4)
        sns.stripplot(data=df_late, x="experiment", y=cc, hue='mice',
                      order=['D1act', 'CONTROL_LIGHT', 'DELAY', 'RANDOM'], palette=color_mapping, jitter=True, ax=ax4)
        a = df_late[df_late.experiment == 'D1act'][cc]
        b = df_late[df_late.experiment == 'CONTROL_LIGHT'][cc]
        c = df_late[df_late.experiment == 'DELAY'][cc]
        d = df_late[df_late.experiment == 'RANDOM'][cc]
        ax4.set_xlabel(cc)
        ut_plots.get_pvalues(a, b, ax4, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(a, c, ax4, pos=1.5, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(a, d, ax4, pos=2.5, height=a[~np.isnan(a)].max(), ind=True)
        if 'div' in cc:
            ut_plots.get_1s_pvalues(a, 1, ax4, pos=0, height=1)
            ut_plots.get_1s_pvalues(b, 1, ax4, pos=1, height=1)
            ut_plots.get_1s_pvalues(c, 1, ax4, pos=2, height=1)
            ut_plots.get_1s_pvalues(d, 1, ax4, pos=3, height=1)



