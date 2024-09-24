__author__ = 'Nuria'

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm

from pathlib import Path
from matplotlib import interactive
from numpy.polynomial import Polynomial as P

import utils.utils_analysis as ut
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
    for cc in df.columns[4:]:
        fig1, ax1 = ut_plots.open_plot()
        for experiment_type in df.experiment.unique():
            # ax1.plot(np.vstack(df[df.experiment == experiment_type][cc].values).T, 'gray')
            ax1.plot(np.nanmean(np.vstack(df[df.experiment == experiment_type][cc].values), 0), label=experiment_type)
        plt.xlabel(cc)
        plt.legend()

    len_array = 20
    df_d1act = df[df.experiment == 'D1act']
    col_aux = [col for col in df_d1act.columns[3:] if 'calib' not in col]
    for cc in col_aux:
        df_d1act = ut.replace_cc_val_with_nan(df_d1act, cc, len_array)
        fig1, ax1 = ut_plots.open_plot()
        sot_array = np.full((len(df_d1act.mice.unique()), len_array), np.nan)
        sot_calib = np.full((len(df_d1act.mice.unique()), len_array), np.nan)
        for mm, mouse in enumerate(df_d1act.mice.unique()):
            day_values = np.vstack(df_d1act[df_d1act.mice == mouse][cc].values)
            if 'stim' in cc:
                cc_calib = cc.replace("stim", "calib")
            elif 'target' in cc:
                cc_calib = cc.replace("target", "calib")
            day_calib = np.vstack(df_d1act[df_d1act.mice == mouse][cc_calib].values)
            # fig2, ax2 = ut_plots.open_plot()
            # for n in np.arange(day_calib.shape[0]):
            #     sns.regplot(x=np.arange(100), y=day_values[n,:] / np.nanmean(day_calib[n,:]), ax=ax2)
            #     ax2.set_xlabel(cc)
            min_x = np.min([len_array, day_values.shape[1]])
            sot_array[mm, :min_x] = np.nanmean(day_values, 0)[:min_x]
            sot_calib[mm, :min_x] = np.nanmean(day_calib, 0)[:min_x]

            # sns.regplot(x=np.arange(len_array), y=sot_array[mm, :], ax=ax1, color=color_mapping[mouse])
            #print('mouse: ' + mouse + 'len: ' + str(np.sum(~np.isnan(sot_array[mm, :]))))
            # ax1.plot(sot_array[mm, :], '.', color=color_mapping[mouse])
        ax1.set_xlabel(cc)
        sns.regplot(x=np.arange(len_array), y=np.nanmean(sot_array, 0), ax=ax1)
        sns.regplot(x=np.arange(len_array), y=np.nanmean(sot_calib, 0), ax=ax1, color='lightgray')
        ut_plots.get_reg_pvalues(np.arange(len_array), np.nanmean(sot_array, 0), ax1, 1, height=np.nanmean(sot_array))

    df_a = df[df.columns[:3]]
    df_e = df[df.columns[:3]]
    df_l = df[df.columns[:3]]
    for cc in df.columns[3:]:
        df_a[cc] = df[cc].apply(lambda x: np.mean(x[~np.isnan(x)]))
        df_e[cc] = df[cc].apply(lambda x: np.mean(x[~np.isnan(x)][:10]))
        df_l[cc] = df[cc].apply(lambda x: np.mean(x[~np.isnan(x)][-10:]))
    df_a['period'] = 'all'
    df_e['period'] = 'early'
    df_l['period'] = 'late'
    df_el = pd.concat((df_l[df_l.columns[[0, 2]]], df_l[df_l.columns[3:-1]]/df_e[df_e.columns[3:-1]]), axis=1)
    df_av = pd.concat((df_a, df_e, df_l))
    df_group = df_el.groupby(['mice', 'experiment']).mean().reset_index()

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

    for cc in df_group.filter(like='SOT'):
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

    df_melt = df_group.melt(id_vars=df_group.columns[:2], var_name='SOT_Type', value_name='SOT_Value')
    df_dn = df_melt.loc[df_melt['SOT_Type'].str.contains('dn'), :].reset_index(drop=True)
    df_in = df_melt.loc[df_melt['SOT_Type'].str.contains('in'), :].reset_index(drop=True)
    df_stim = df_melt.loc[df_melt['SOT_Type'].str.contains('stim'), :].reset_index(drop=True)
    df_target = df_melt.loc[df_melt['SOT_Type'].str.contains('target'), :].reset_index(drop=True)
    df_calib = df_melt.loc[df_melt['SOT_Type'].str.contains('calib'), :].reset_index(drop=True)
    df_div = df_melt.loc[df_melt['SOT_Type'].str.contains('dn'), df_melt.columns[:3]].reset_index(drop=True)
    df_div['DIV_Value'] = df_dn.SOT_Value / df_in.SOT_Value

    for tt in df_div.SOT_Type.unique():
        fig2, ax2 = ut_plots.open_plot()
        df_aux = df_div[df_div.SOT_Type==tt]
        sns.boxplot(data=df_aux, x='experiment', y='DIV_Value', order=['D1act', 'DELAY', 'RANDOM'],
                    ax=ax2)
        sns.stripplot(data=df_aux, x='experiment', y='DIV_Value', hue='mice', order=['D1act', 'DELAY', 'RANDOM'],
                      palette=color_mapping, ax=ax2)
        ax2.set_xlabel(tt)
        a = df_aux[df_aux.experiment == 'D1act'].DIV_Value
        c = df_aux[df_aux.experiment == 'DELAY'].DIV_Value
        d = df_aux[df_aux.experiment == 'RANDOM'].DIV_Value
        ut_plots.get_pvalues(a, c, ax2, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(a, d, ax2, pos=1.5, height=a[~np.isnan(a)].max(), ind=True)

    df_auxc = df_dn[df_dn.SOT_Type=='SOT_calib_dn'].drop(['mice', 'SOT_Type'], axis=1)
    df_c = df_auxc.groupby('experiment').mean()
    for tt in df_dn.SOT_Type.unique():
        fig2, ax2 = ut_plots.open_plot()
        df_aux = df_dn[df_dn.SOT_Type==tt]
        sns.boxplot(data=df_aux, x='experiment', y='SOT_Value', order=['D1act', 'DELAY', 'RANDOM'],
                    ax=ax2)
        sns.stripplot(data=df_aux, x='experiment', y='SOT_Value', hue='mice', order=['D1act', 'DELAY', 'RANDOM'],
                      palette=color_mapping, ax=ax2)
        ax2.set_xlabel(tt)
        ax2.axhline(y=df_c.loc['D1act'].SOT_Value, xmin=0, xmax=0.2, color='red', linestyle='--')
        ax2.axhline(y=df_c.loc['DELAY'].SOT_Value, xmin=0.3, xmax=0.6, color='red', linestyle='--')
        ax2.axhline(y=df_c.loc['RANDOM'].SOT_Value, xmin=0.7, xmax=1, color='red', linestyle='--')
        a = df_aux[df_aux.experiment == 'D1act'].SOT_Value
        c = df_aux[df_aux.experiment == 'DELAY'].SOT_Value
        d = df_aux[df_aux.experiment == 'RANDOM'].SOT_Value
        ca = df_auxc[df_auxc.experiment == 'D1act'].SOT_Value
        cc = df_auxc[df_auxc.experiment == 'DELAY'].SOT_Value
        cd = df_auxc[df_auxc.experiment == 'RANDOM'].SOT_Value
        ut_plots.get_pvalues(a, ca, ax2, pos=0, height=df_c.loc['D1act'].SOT_Value, ind=False)
        ut_plots.get_pvalues(c, cc, ax2, pos=0.5, height=df_c.loc['D1act'].SOT_Value, ind=False)
        ut_plots.get_pvalues(d, cd, ax2, pos=1, height=df_c.loc['D1act'].SOT_Value, ind=False)
        ut_plots.get_pvalues(a, c, ax2, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(a, d, ax2, pos=1.5, height=a[~np.isnan(a)].max(), ind=True)


    for ee in df_div.experiment.unique():
        df_c = df_calib[df_calib.experiment==ee].reset_index()
        df_c['SOT_Type'] = df_c['SOT_Type'].str.replace('SOT_calib_', '')

        fig3, ax3 = ut_plots.open_plot()
        df_exp = df_stim[df_stim.experiment==ee]
        df_exp = df_exp.loc[~df_exp['SOT_Type'].str.contains('div'), :].reset_index(drop=True)
        df_exp['SOT_Type'] = df_exp['SOT_Type'].str.replace('SOT_stim_', '')
        df_exp['DIV_Value'] = df_exp.SOT_Value / df_c.SOT_Value
        sns.lineplot(data=df_exp, x='SOT_Type', y='DIV_Value', hue='mice', palette=color_mapping, ax=ax3)
        sns.stripplot(data=df_exp, x='SOT_Type', y='DIV_Value', hue='mice',
                      palette=color_mapping, s=10, marker="D", jitter=False, ax=ax3)
        ax3.set_xlabel(ee + ' stim')
        ax3.set_ylim([0.5, 3])
        a = df_exp[df_exp.SOT_Type == 'dn'].DIV_Value
        b = df_exp[df_exp.SOT_Type == 'in'].DIV_Value
        ut_plots.get_pvalues(a, b, ax3, pos=0.5, height=a[~np.isnan(a)].max(), ind=False)

        fig4, ax4 = ut_plots.open_plot()
        df_exp = df_target[df_target.experiment==ee]
        df_exp['SOT_Type'] = df_exp['SOT_Type'].str.replace('SOT_target_', '')
        df_exp = df_exp.loc[~df_exp['SOT_Type'].str.contains('div'), :].reset_index(drop=True)
        df_exp['DIV_Value'] = df_exp.SOT_Value / df_c.SOT_Value
        sns.lineplot(data=df_exp, x='SOT_Type', y='DIV_Value', hue='mice', palette=color_mapping, ax=ax4)
        sns.stripplot(data=df_exp, x='SOT_Type', y='DIV_Value', hue='mice',
                      palette=color_mapping, s=10, marker="D", jitter=False, ax=ax4)
        ax4.set_xlabel(ee + ' target')
        ax4.set_ylim([0.5, 3])
        a = df_exp[df_exp.SOT_Type == 'dn'].DIV_Value
        b = df_exp[df_exp.SOT_Type == 'in'].DIV_Value
        ut_plots.get_pvalues(a, b, ax4, pos=0.5, height=a[~np.isnan(a)].max(), ind=False)

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

    for cc in df_late.columns[[2,3,4,5,8,9]]:#[[2,3,4,5,14,15,16,17]]:
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


def plot_SOT_EL(df: pd.DataFrame, df_learning: pd.DataFrame, df_snr:pd.DataFrame):
    color_mapping = ut_plots.generate_palette_all_figures()
    df_good = df_learning[df_learning.gain>1.5]
    df_bad = df_snr[df_snr.snr_dn_min < 2]

    df = df[df.session_path.isin(df_good.session_path.unique())]
    df = df[~df.session_path.isin(df_bad.session_path.unique())]

    df_melt = df.melt(id_vars=df.columns[:5], var_name='SOT_Type', value_name='SOT_Value').drop('window', axis=1)
    df_melt = df_melt.groupby(['mice', 'session_path', 'experiment', 'win_type', 'SOT_Type']).mean().reset_index()

    for experiment in df.experiment.unique():
        df_exp = df[df.experiment==experiment]
        fig1, ax1 = ut_plots.open_plot()
        sns.lineplot(data=df_exp, x='window', y='SOT_dn', ax=ax1)
        sns.lineplot(data=df_exp, x='window', y='SOT_in', ax=ax1)
        ax1.set_xlabel(experiment)

        df_exp = df_melt[df_melt.experiment == experiment]
        fig2, ax2 = ut_plots.open_plot()
        sns.boxplot(data=df_exp, x='win_type', y='SOT_Value', hue='SOT_Type', ax=ax2)
        df_dn = df_exp[df_exp.SOT_Type=='SOT_dn']
        df_in = df_exp[df_exp.SOT_Type=='SOT_in']
        a = df_dn[df_dn.win_type=='calib'].SOT_Value
        b = df_dn[df_dn.win_type=='exp'].SOT_Value
        c = df_in[df_in.win_type=='calib'].SOT_Value
        d = df_in[df_in.win_type=='exp'].SOT_Value
        ut_plots.get_pvalues(a, c, ax2, pos=0.2, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(b, d, ax2, pos=0.8, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(a, b, ax2, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
        ax2.set_xlabel(experiment)

        fig3, ax3 = ut_plots.open_plot()
        sns.boxplot(data=df_exp, x='SOT_Type', y='SOT_Value', hue='win_type', ax=ax3)
        ut_plots.get_pvalues(a, b, ax3, pos=0.2, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(c, d, ax3, pos=0.8, height=a[~np.isnan(a)].max(), ind=True)
        ut_plots.get_pvalues(b, d, ax3, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
        ax3.set_xlabel(experiment)


    df_exp = df_melt[df_melt.experiment.isin(['D1act', 'CONTROL_LIGHT'])]
    df_calib = df_exp[df_exp.win_type == 'calib'].drop('win_type', axis=1).reset_index(drop=True)
    df_exper = df_exp[df_exp.win_type == 'exp'].drop('win_type', axis=1).reset_index(drop=True)
    df_div = df_calib.loc[:, df_calib.columns[[0,2,3]]]
    df_div['DIV_Value'] = df_exper.iloc[:, 4:] / df_calib.iloc[:, 4:]

    fig5, ax5 = ut_plots.open_plot()
    sns.boxplot(data=df_div, x='experiment', y='DIV_Value', hue='SOT_Type',
                order=['D1act', 'CONTROL_LIGHT'], ax=ax5)
    df_d1act = df_div[df_div.experiment == 'D1act']
    df_control = df_div[df_div.experiment == 'CONTROL_LIGHT']
    a = df_d1act[df_d1act.SOT_Type=='SOT_dn']['DIV_Value']
    b = df_d1act[df_d1act.SOT_Type=='SOT_in']['DIV_Value']
    c = df_control[df_control.SOT_Type=='SOT_dn']['DIV_Value']
    d = df_control[df_control.SOT_Type=='SOT_in']['DIV_Value']
    ut_plots.get_pvalues(a, b, ax5, pos=0.2, height=a[~np.isnan(a)].max(), ind=True)
    ut_plots.get_pvalues(c, d, ax5, pos=0.8, height=a[~np.isnan(a)].max(), ind=True)
    ut_plots.get_pvalues(a, c, ax5, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)


    df_melt = df.melt(id_vars=df.columns[:5], var_name='SOT_Type', value_name='SOT_Value')
    df_late = df_melt[df_melt.window==8].drop(['window', 'win_type', 'session_path'], axis=1).reset_index(drop=True)
    df_early = df_melt[df_melt.window == 3].drop(['window', 'win_type', 'session_path'], axis=1).reset_index(drop=True)
    df_calib = df_melt[df_melt.window == 2].drop(['window', 'win_type', 'session_path'], axis=1).reset_index(drop=True)
    df_late['period'] = 'late'
    df_early['period'] = 'early'
    df_calib['period'] = 'calib'
    df_period = pd.concat([pd.concat([df_late, df_early]), df_calib])

    for experiment in df.experiment.unique():
        df_exp = df_period[df_period.experiment==experiment]
        fig1, ax1 = ut_plots.open_plot()
        sns.boxplot(data=df_exp, x='period', y='SOT_Value', hue='SOT_Type', order=['calib', 'early', 'late'], ax=ax1)
        ax1.set_xlabel(experiment)

        fig2, ax2 = ut_plots.open_plot()
        sns.boxplot(data=df_exp, x='SOT_Type', y='SOT_Value', hue='period', hue_order=['calib', 'early', 'late'], ax=ax2)
        ax2.set_xlabel(experiment)

    fig3, ax3 = ut_plots.open_plot()
    sns.boxplot(data=df_late, x='experiment', y='SOT_Value', hue='SOT_Type', ax=ax3)
    fig4, ax4 = ut_plots.open_plot()
    sns.boxplot(data=df_late, x='SOT_Type', y='SOT_Value', hue='experiment', ax=ax4)

    df_div = df_late[df_late.columns[:3]]
    df_div['DIV_Value'] = df_late.reset_index().SOT_Value / df_calib.reset_index().SOT_Value
    fig5, ax5 = ut_plots.open_plot()
    sns.boxplot(data=df_div, x='SOT_Type', y='DIV_Value', hue='experiment', ax=ax5)
    fig6, ax6 = ut_plots.open_plot()
    sns.boxplot(data=df_div, x='experiment', y='DIV_Value', hue='SOT_Type', ax=ax6)


    ##

def plot_tim_manifold():
    len_array = 30
    for cc in ['dim', 'SOT', 'VAF']:
        fig1, ax1 = ut_plots.open_plot()
        sot_array = np.full((len(df.mice.unique()), len_array), np.nan)
        for mm, mouse in enumerate(df.mice.unique()):
            day_values = np.vstack(df[df.mice == mouse][cc].values)

            # fig2, ax2 = ut_plots.open_plot()
            # for n in np.arange(day_calib.shape[0]):
            #     sns.regplot(x=np.arange(100), y=day_values[n,:] / np.nanmean(day_calib[n,:]), ax=ax2)
            #     ax2.set_xlabel(cc)
            min_x = np.min([len_array, day_values.shape[1]])
            sot_array[mm, :min_x] = np.nanmean(day_values, 0)[:min_x]
        ax1.set_xlabel(cc)
        sns.regplot(x=np.arange(len_array), y=np.nanmean(sot_array, 0), ax=ax1)
        ut_plots.get_reg_pvalues(np.arange(len_array), np.nanmean(sot_array, 0), ax1, 1, height=np.nanmean(sot_array))


def plot_sot_trial(folder_list: list):

    # r2_l = np.nanmean(r2_l_e, 3)
    # r2_l2 = np.nanmean(r2_l2_e, 3)
    # r2_rcv = np.nanmean(r2_rcv_e, 3)
    # r2_dff_rcv = np.nanmean(r2_dff_rcv_e, 3)
    # aux = np.nanmean(r2_dff_rcv, 2)
    # plt.plot(aux[0,:]), plt.plot(aux[19,:])
    indices_lag = np.arange(-120, 90, 6)
    SOT_all_dnd, SOT_all_ind = dp.obtain_population_trial_SOT(folder_list, 'DELAY')
    SOT_all_dn, SOT_all_in = dp.obtain_population_trial_SOT(folder_list, 'D1act')
    df_d1act = ss.get_sessions_df(folder_list, 'D1act')
    mice_d1act = df_d1act.mice_name.unique()
    df_delay = ss.get_sessions_df(folder_list, 'DELAY')
    mice_delay = df_delay.mice_name.unique()
    df_random = ss.get_sessions_df(folder_list, 'RANDOM')
    mice_random = df_random.mice_name.unique()

    SOT_dn = np.nanmean(np.nanmean(SOT_all_dn, 3), 0)
    SOT_in = np.nanmean(np.nanmean(SOT_all_in, 3), 0)
    SOT_dnd = np.nanmean(np.nanmean(SOT_all_dnd, 3), 0)
    SOT_ind = np.nanmean(np.nanmean(SOT_all_ind, 3), 0)
    SOT_dnr = np.nanmean(np.nanmean(SOT_all_dnr, 3), 0)
    SOT_inr = np.nanmean(np.nanmean(SOT_all_inr, 3), 0)

    fig1, ax1 = ut_plots.open_plot()
    for ind in np.arange(SOT_dn.shape[1]):
        ax1.plot(indices_lag / 30, SOT_dn[:, ind], color=color_mapping[mice_d1act[ind]], lw=0.2)
    ax1.plot(indices_lag / 30, np.nanmean(SOT_dn,1), color='darkgray', lw=4)
    ax1.plot(indices_lag / 30, np.nanmean(SOT_in, 1), color='lightgray', lw=4)
    ax1.axvline(x=0, ymin=0.15, ymax=0.35, color='r', linestyle='--')
    ax1.set_xlabel('D1act')
    ax1.set_ylim([0.1, 0.75])

    fig2, ax2 = ut_plots.open_plot()
    for ind in np.arange(SOT_dnd.shape[1]):
        ax2.plot(indices_lag / 30, SOT_dnd[:, ind], color=color_mapping[mice_delay[ind]], lw=0.2)
    ax2.plot(indices_lag / 30, np.nanmean(SOT_dnd, 1), color='darkgray', lw=4)
    ax2.plot(indices_lag / 30, np.nanmean(SOT_ind, 1), color='lightgray', lw=4)
    ax2.axvline(x=0, ymin=0.15, ymax=0.35, color='r', linestyle='--')
    ax2.set_xlabel('Delay')
    ax2.set_ylim([0.1, 0.75])

    fig3, ax3 = ut_plots.open_plot()
    for ind in np.arange(SOT_dnr.shape[1]):
        ax3.plot(indices_lag / 30, SOT_dnr[:, ind], color=color_mapping[mice_random[ind]], lw=0.2)
    ax3.plot(indices_lag / 30, np.nanmean(SOT_dnr, 1), color='darkgray', lw=4)
    ax3.plot(indices_lag / 30, np.nanmean(SOT_inr, 1), color='lightgray', lw=4)
    ax3.axvline(x=0, ymin=0.15, ymax=0.35, color='r', linestyle='--')
    ax3.set_xlabel('RANDOM')
    ax3.set_ylim([0.1, 0.75])


def plot_events_stim(df_stim: pd.DataFrame, folder_plots: Path):
    """ plot  stimulus vs events """
    color_mapping = ut_plots.generate_palette_all_figures()
    df_stim['next_sec'] = (df_stim['next']/AnalysisConstants.framerate).round()
    # df_stim = df_stim.drop(columns=['zscore_next_5', 'next_5', 'rate_5min', 'rate_all'])
    df_stim_mean = (df_stim.drop(['session_path', 'neuron', 'stim_index'], axis=1).
                        groupby(['neuron_type', 'event_dist', 'mice']).mean().reset_index())

    for column in df_stim_mean.columns[3:]:
        fig1, ax1 = ut_plots.open_plot()
        data_to_plot = df_stim[df_stim['event_dist'] < 90].drop('neuron_type', axis=1)
        data_to_plot = data_to_plot[['event_dist', column]].dropna()
        # data_to_plot = df_stim_mean[(df_stim_mean['neuron_type'] == 'na') &
        #                             (df_stim_mean['event_dist'] < 60)]
        sns.regplot(data= data_to_plot, x='event_dist', y=column, x_bins=np.arange(0,90,2), fit_reg=True, ax=ax1)
        # sns.regplot(data=data_to_plot[data_to_plot.event_dist<=9], x='event_dist', y=column, fit_reg=True,
        #             x_bins=np.arange(0, 90, 2), ax=ax1)
        ax1.set_xticks(np.arange(0,90, AnalysisConstants.framerate/2))
        ax1.set_xticklabels(np.around(np.arange(0,3.1,0.5), 1).astype(str))
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(column)
        data_to_plot = data_to_plot.groupby('event_dist').mean().reset_index()
        # ax1.set_ylim([2, 3.7])
        y = data_to_plot[column]
        X = data_to_plot['event_dist']
        ut_plots.get_reg_pvalues(y, X, ax1, 5, np.nanmean(data_to_plot[column]))
        weights = data_to_plot.groupby('event_dist').size()
        X = sm.add_constant(X)
        model = sm.WLS(y, X).fit()
        ax1.text(2, np.nanmean(y), model.rsquared)
        ut_plots.save_plot(fig1, ax1, folder_plots, 'stim_all_' , column, False)

    event_loc = [0 , 14]
    df_stim['event'] = df_stim['event_dist'].apply(lambda x: event_loc[0] < x < event_loc[1])
    df_stim_mean = (df_stim.drop(['session_path', 'neuron', 'stim_index'], axis=1).
                        groupby(['neuron_type', 'event_dist', 'mice', 'event']).mean().reset_index())
    df_stim_mean_mice = df_stim_mean.groupby(['neuron_type', 'mice', 'event']).mean().reset_index()
    for column in df_stim_mean.columns[5:]:
        if column == 'next_sec':
            y_lim_var = [0, 130]
        elif column == 'rate_min':
            y_lim_var =[0,17]
        else:
            y_lim_var = [2,4.5]
        for neuron_type in df_stim.neuron_type.unique():
            fig2, ax2 = ut_plots.open_plot()
            sns.boxplot(data=df_stim_mean_mice[df_stim_mean_mice['neuron_type'] == neuron_type], x='event', y=column, ax=ax2)
            sns.stripplot(df_stim_mean_mice[df_stim_mean_mice['neuron_type'] == neuron_type], x='event',
                          y=column, hue='mice', palette=color_mapping, ax=ax2)
            a = df_stim_mean_mice[(df_stim_mean_mice['neuron_type'] == neuron_type) &
                                  (df_stim_mean_mice['event'] == False)][column].values
            b = df_stim_mean_mice[(df_stim_mean_mice['neuron_type'] == neuron_type) &
                                  (df_stim_mean_mice['event'] == True)][column].values
            ut_plots.get_pvalues(a, b, ax2, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)

            if neuron_type in ['na', 'E1']:
                c = df_stim_mean_mice[(df_stim_mean_mice['neuron_type'] == 'E2') &
                                      (df_stim_mean_mice['event'] == True)][column].values
                d = df_stim_mean_mice[(df_stim_mean_mice['neuron_type'] == neuron_type) &
                                      (df_stim_mean_mice['event'] == True)][column].values
                ut_plots.get_pvalues(c, d, ax2, pos=1, height=c[~np.isnan(c)].max(), ind=True)

            ax2.set_ylim(y_lim_var)
            ax2.set_title(neuron_type)
            ut_plots.save_plot(fig2, ax2, folder_plots, 'stim_box_' + neuron_type, column, False)




    scale_factor = 30
    bin_edges = np.arange(0.5, 3.5, 0.5) * scale_factor
    bin_edges = np.concatenate(([-float('inf')], bin_edges, [float('inf')]))

    # Create labels for each bin
    labels = [f'e<{bin_edges[1] / scale_factor}'] + \
             [f'{bin_edges[i] / scale_factor}<=e<{bin_edges[i + 1] / scale_factor}' for i in
              range(1, len(bin_edges) - 2)] + \
             [f'e>={bin_edges[-2] / scale_factor}']

    # Create a new column 'event' based on conditions in 'event_dist'
    df_stim['event'] = pd.cut(df_stim['event_dist'], bins=bin_edges, labels=labels, right=False)

    df_stim = df_stim.dropna()
    # df_stim_mean = (df_stim.drop(['event_dist', 'stim_index', 'mice'], axis=1).
    #                 groupby(['neuron', 'session_path', 'neuron_type', 'event']).mean().reset_index())
    df_stim_mean_mice = (df_stim.drop(['neuron', 'event_dist', 'stim_index', 'session_path'], axis=1).dropna().
                         groupby(['neuron_type', 'mice', 'event']).mean().reset_index())
    df_stim_mean_mice = df_stim_mean_mice.dropna()
    # df_stim_mean = df_stim_mean.dropna()
    for column in df_stim_mean_mice.columns[3:]:
        for neuron_type in df_sorted_mean.neuron_type.unique():
            fig2, ax2 = ut_plots.open_plot()
            sns.boxplot(data=df_stim_mean_mice[df_stim_mean_mice['neuron_type'] == neuron_type], x='event', y=column, ax=ax2)
            sns.stripplot(df_stim_mean_mice[df_stim_mean_mice['neuron_type'] == neuron_type], x='event',
                          y=column, hue='mice', palette=color_mapping, ax=ax2)
            a = df_stim_mean_mice[(df_stim_mean_mice['neuron_type'] == neuron_type) &
                                  (df_stim_mean_mice['event'] == 'e<0.5')][column].values
            for ee,ev in enumerate(df_stim_mean_mice.event.unique()[1:]):

                b = df_stim_mean_mice[(df_stim_mean_mice['neuron_type'] == neuron_type) &
                                  (df_stim_mean_mice['event'] == ev)][column].values
                ut_plots.get_pvalues(a, b, ax2, pos=0.5+ee, height=a[~np.isnan(a)].max(), ind=True)
            # ax2.set_ylim([0, 130])
            ax2.set_title(neuron_type)
            ut_plots.save_plot(fig2, ax2, folder_plots, 'stim_box_times_' + neuron_type, column, False)

    df_x = df_stim_mean_mice[df_stim_mean_mice['event'].isin(['e<0.5', '2.5<=e<3.0'])]
    df_x['event'] = df_x['event'].cat.remove_unused_categories()
    for column in df_stim_mean_mice.columns[3:]:
        for neuron_type in df_sorted_mean.neuron_type.unique():
            fig2, ax2 = ut_plots.open_plot()
            sns.boxplot(data=df_x[df_x['neuron_type'] == neuron_type], x='event', y=column, ax=ax2)
            sns.stripplot(df_x[df_x['neuron_type'] == neuron_type], x='event',
                          y=column, hue='mice', palette=color_mapping, ax=ax2)
            a = df_x[(df_x['neuron_type'] == neuron_type) &
                                  (df_x['event'] == 'e<0.5')][column].values
            b = df_x[(df_x['neuron_type'] == neuron_type) &
                                  (df_x['event'] == '2.5<=e<3.0')][column].values
            ut_plots.get_pvalues(a, b, ax2, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
            # ax2.set_ylim([0, 130])
            ax2.set_title(neuron_type)


    df_sorted = df_stim.drop(['stim_index', 'event_dist'], axis=1).sort_values(by=['session_path', 'neuron_type', 'event', 'neuron'])
    df_sorted['stim_count'] = df_sorted.groupby(['neuron', 'session_path', 'event']).cumcount()
    df_sorted_mean = (df_sorted.drop('session_path', axis=1).
                      groupby(['neuron', 'event', 'mice','neuron_type', 'stim_count']).mean().reset_index())
    na_max_count_aux = df_sorted[(df_sorted.neuron_type == 'na') & (df_sorted.event == True)].stim_count.value_counts()
    na_max_count = na_max_count_aux[na_max_count_aux >= 3].index.max()
    E2_max_count_aux = df_sorted[(df_sorted.neuron_type == 'E2') & (df_sorted.event == True)].stim_count.value_counts()
    E2_max_count = E2_max_count_aux[E2_max_count_aux >= 3].index.max()
    E1_max_count_aux = df_sorted[(df_sorted.neuron_type == 'E1') & (df_sorted.event == True)].stim_count.value_counts()
    E1_max_count = E1_max_count_aux[E1_max_count_aux >= 3].index.max()
    for column in df_sorted.columns[2:-4]:
        for neuron_type in df_sorted_mean.neuron_type.unique():
            if neuron_type== 'na':
                max_count = na_max_count
            elif neuron_type=='E1':
                max_count = E1_max_count
            elif neuron_type=='E2':
                max_count = E2_max_count
            data_to_plot = df_sorted[(df_sorted.neuron_type==neuron_type)&(df_sorted.stim_count<=max_count)]
            data_to_plot = data_to_plot[['stim_count', 'event', column]].dropna()
            weights = 1. / (data_to_plot[data_to_plot.event]['stim_count'] + 0.1)
            y = data_to_plot[data_to_plot.event][column]
            X = sm.add_constant(data_to_plot[data_to_plot.event]['stim_count'])
            model_wls = sm.WLS(y, X, weights=weights)
            results_wls = model_wls.fit()
            h = sns.lmplot(x='stim_count', y=column, data=data_to_plot, hue='event',
                       x_bins=np.arange(0, E2_max_count, 1))
            h.ax.text(1, data_to_plot[data_to_plot.event][column].mean(), ut_plots.calc_pvalue(results_wls.pvalues.stim_count))
            h.ax.text(1.1, data_to_plot[data_to_plot.event][column].mean(), "p = %0.1E" % results_wls.pvalues.stim_count)
            h.ax.set_title(neuron_type + '_' + column)
            # ut_plots.get_anova_pvalues(data_to_plot[data_to_plot.event == True][column],
            #                          data_to_plot[data_to_plot.event == False][column])
            # ut_plots.get_reg_pvalues(data_to_plot[data_to_plot.event==True][column],
            #                          data_to_plot[data_to_plot.event==True]['stim_count'],
            #                          h.ax, 0, np.nanmean(data_to_plot[column]))
            # ut_plots.get_reg_pvalues(data_to_plot[data_to_plot.event==False][column],
            #                          data_to_plot[data_to_plot.event==False]['stim_count'],
            #                          h.ax, 5, np.nanmean(data_to_plot[column]))

            # h.ax.set_ylim([2, 6.5])
            ut_plots.save_plot(h.fig, h.ax, folder_plots, 'stim_counts_' + neuron_type, column, False)

def plot_events(df: pd.DataFrame):
    """ plot events vs stimulus """
    df_nocab = df[~df['calibration']].drop(['calibration', 'baseline'], axis=1)
    df_nocab_mean = df_nocab.drop('session_path', axis=1).groupby(['neuron_type', 'stim_dist', 'mice']).mean().reset_index()
    df_mean = (df.drop(['calibration', 'baseline', 'session_path'], axis=1).
               groupby(['neuron_type', 'stim_dist', 'mice']).mean().reset_index())

    for column in df_nocab_mean.columns[5:-1]:
        for neuron_type in ['E2', 'na']:
            fig1, ax1 = ut_plots.open_plot()
            data_to_plot = df_nocab_mean[(df_nocab_mean['neuron_type'] == neuron_type) &
                                           (df_nocab_mean['stim_dist'] < 300)]
            sns.regplot(data=data_to_plot, x='stim_dist', y=column, x_bins=np.arange(0,300,5), ax=ax1)
            ax1.set_xticks(np.arange(0,300, AnalysisConstants.framerate))
            ax1.set_xticklabels(np.arange(0,11).astype(str))
            ax1.set_title(neuron_type)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel(column)
            ax1.set_title(neuron_type)
            ut_plots.get_reg_pvalues(data_to_plot[column], data_to_plot['stim_dist'],
                                     ax1, 5, np.nanmean(data_to_plot[column]))
            ut_plots.save_plot(fig1, ax1, folder_plots, 'event_' + neuron_type, column, False)


    stim_loc = [0 , 14]
    df_nocab_mean['stim'] = df_nocab_mean['stim_dist'].apply(lambda x: stim_loc[0] < x < stim_loc[1])
    df_mean['stim'] = df_mean['stim_dist'].apply(lambda x: stim_loc[0] < x < stim_loc[1])
    df_nocab['stim'] = df_nocab['stim_dist'].apply(lambda x: stim_loc[0] < x < stim_loc[1])

    df_nocab_mean_mice = df_nocab_mean.groupby(['neuron_type', 'mice', 'stim']).mean().reset_index()
    df_mean_mice = df_mean.groupby(['neuron_type', 'mice', 'stim']).mean().reset_index()

    for column in df_nocab_mean.columns[5:-1]:
        for neuron_type in df_nocab_mean.neuron_type.unique():
            fig1, ax1 = ut_plots.open_plot()
            sns.boxplot(data=df_nocab_mean_mice[df_nocab_mean_mice['neuron_type'] == neuron_type], x='stim', y=column, ax=ax1)
            sns.stripplot(df_nocab_mean_mice[df_nocab_mean_mice['neuron_type'] == neuron_type], x='stim',
                          y=column, hue='mice', palette=color_mapping, ax=ax1)
            a = df_nocab_mean_mice[(df_nocab_mean_mice['neuron_type'] == neuron_type) & (df_nocab_mean_mice['stim'] == False)][column].values
            b = df_nocab_mean_mice[(df_nocab_mean_mice['neuron_type'] == neuron_type) & (df_nocab_mean_mice['stim'] == True)][column].values
            ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
            ax1.set_title(neuron_type + "_nobas")

            fig2, ax2 = ut_plots.open_plot()
            sns.boxplot(data=df_mean_mice[df_mean_mice['neuron_type'] == neuron_type], x='stim', y=column, ax=ax2)
            sns.stripplot(df_mean_mice[df_mean_mice['neuron_type'] == neuron_type], x='stim',
                          y=column, hue='mice', palette=color_mapping, ax=ax2)
            a = df_mean_mice[(df_mean_mice['neuron_type'] == neuron_type) & (df_mean_mice['stim'] == False)][column].values
            b = df_mean_mice[(df_mean_mice['neuron_type'] == neuron_type) & (df_mean_mice['stim'] == True)][column].values
            ut_plots.get_pvalues(a, b, ax2, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
            ax2.set_title(neuron_type)

    for column in df_nocab.columns[5:-1]:
        for neuron_type in df_nocab.neuron_type.unique():
            fig3, ax3 = ut_plots.open_plot()
            sns.boxplot(data=df_nocab[df_nocab['neuron_type'] == neuron_type], x='stim', y=column, ax=ax3)
            # sns.stripplot(df_nocab[df_nocab['neuron_type'] == neuron_type], x='stim',
            #               y=column, hue='mice', palette=color_mapping, ax=ax1)
            a = df_nocab[(df_nocab['neuron_type'] == neuron_type) & (df_nocab['stim'] == False)][column].values
            b = df_nocab[(df_nocab['neuron_type'] == neuron_type) & (df_nocab['stim'] == True)][column].values
            ut_plots.get_pvalues(a, b, ax3, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
            ax3.set_title(neuron_type + "_nobas")

            fig4, ax4 = ut_plots.open_plot()
            sns.boxplot(data=df[df['neuron_type'] == neuron_type], x='stim', y=column, ax=ax4)
            # sns.stripplot(df[df['neuron_type'] == neuron_type], x='stim',
            #               y=column, hue='mice', palette=color_mapping, ax=ax2)
            a = df[(df['neuron_type'] == neuron_type) & (df['stim'] == False)][column].values
            b = df[(df['neuron_type'] == neuron_type) & (df['stim'] == True)][column].values
            ut_plots.get_pvalues(a, b, ax4, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)
            ax4.set_title(neuron_type)


    df_sorted = df_nocab.sort_values(by=['session_path', 'neuron_type', 'stim', 'neuron'])
    df_sorted['event_count'] = df_sorted.groupby(['neuron', 'session_path', 'stim']).cumcount()
    df_sorted_mean = (df_sorted.drop('session_path', axis=1).
                      groupby(['neuron', 'stim', 'mice','neuron_type', 'event_count']).mean().reset_index())
    na_max_count_aux = df_sorted[(df_sorted.neuron_type == 'na') & (df_sorted.stim == True)].event_count.value_counts()
    na_max_count = na_max_count_aux[na_max_count_aux >= 3].index.max()
    E2_max_count_aux = df_sorted[(df_sorted.neuron_type == 'E2') & (df_sorted.stim == True)].event_count.value_counts()
    E2_max_count = E2_max_count_aux[E2_max_count_aux >= 3].index.max()
    for column in df_sorted.columns[5:-4]:
        for neuron_type in df_nocab_mean.neuron_type.unique():
            if neuron_type== 'na':
                max_count = na_max_count
            elif neuron_type=='E2':
                max_count = E2_max_count
            data_to_plot = df_sorted[(df_sorted.neuron_type==neuron_type)&(df_sorted.event_count<=max_count)]
            h = sns.lmplot(x='event_count', y=column, data=data_to_plot, hue='stim',
                       x_bins=np.arange(0, E2_max_count, 1))
            ut_plots.get_reg_pvalues(data_to_plot[data_to_plot.stim==True][column],
                                     data_to_plot[data_to_plot.stim==True]['event_count'],
                                     h.ax, 0, np.nanmean(data_to_plot[column]))
            ut_plots.get_reg_pvalues(data_to_plot[data_to_plot.stim==False][column],
                                     data_to_plot[data_to_plot.stim==False]['event_count'],
                                     h.ax, 5, np.nanmean(data_to_plot[column]))
            h.ax.set_title(neuron_type + '_' + column)
            ut_plots.save_plot(h.fig, h.ax, folder_plots, 'event_counts' + neuron_type, column, False)


def plot_dim_sot_window(df):
    """ to plot dim and sot in window times """
    color_mapping = ut_plots.generate_palette_all_figures()
    df_group = df.drop(['session_path', 'win_type'], axis=1).groupby(['mice', 'window']).mean().reset_index()

    fig1, ax1 = ut_plots.open_plot()
    sns.stripplot(data=df_group, x='window', y='SOT_dn', hue='mice', palette=color_mapping, ax=ax1)
    sns.regplot(data=df_group, x='window', y='SOT_dn', scatter=False, ax=ax1)
    ax1.set_xticklabels(np.sort(df_group.window.unique())*AnalysisConfiguration.FA_time_win)

    fig2, ax2 = ut_plots.open_plot()
    sns.stripplot(data=df_group, x='window', y='SOT_in', hue='mice', palette=color_mapping, ax=ax2)
    sns.regplot(data=df_group, x='window', y='SOT_in', scatter=False, ax=ax2)
    ax2.set_xticklabels(np.sort(df_group.window.unique()) * AnalysisConfiguration.FA_time_win)

    fig3, ax3 = ut_plots.open_plot()
    sns.stripplot(data=df_group, x='window', y='DIM_dn', hue='mice', palette=color_mapping, ax=ax3, jitter=False)
    sns.regplot(data=df_group, x='window', y='DIM_dn', scatter=False, ax=ax3)
    ax1.set_xticklabels(np.sort(df_group.window.unique())*AnalysisConfiguration.FA_time_win)

    fig4, ax4 = ut_plots.open_plot()
    sns.stripplot(data=df_group, x='window', y='DIM_in', hue='mice', palette=color_mapping, ax=ax4, jitter=False)
    sns.regplot(data=df_group, x='window', y='DIM_in', scatter=False, ax=ax4)
    ax1.set_xticklabels(np.sort(df_group.window.unique())*AnalysisConfiguration.FA_time_win)


def plot_engagement(df):
    """ plot engagement of all"""
    color_mapping = ut_plots.generate_palette_all_figures()
    len_array = 20
    df_d1act = df[df.experiment == 'D1act']
    col_aux = [col for col in df_d1act.columns[3:] if 'calib' not in col]
    for cc in col_aux:
        df_d1act = ut.replace_cc_val_with_nan(df_d1act, cc, len_array)
        fig1, ax1 = ut_plots.open_plot()
        fig2, ax2 = ut_plots.open_plot()
        ax1.set_xlabel(cc)
        ax2.set_xlabel(cc)
        sot_array = np.full((len(df_d1act.mice.unique()), len_array), np.nan)
        sot_calib = np.full((len(df_d1act.mice.unique()), len_array), np.nan)
        for mm, mouse in enumerate(df_d1act.mice.unique()):
            day_values = np.vstack(df_d1act[df_d1act.mice == mouse][cc].values)
            if 'stim' in cc:
                cc_calib = cc.replace("stim", "calib")
            elif 'target' in cc:
                cc_calib = cc.replace("target", "calib")
            day_calib = np.vstack(df_d1act[df_d1act.mice == mouse][cc_calib].values)
            min_x = np.min([len_array, day_values.shape[1]])
            sot_array[mm, :min_x] = np.nanmean(day_values, 0)[:min_x]
            sot_calib[mm, :min_x] = np.nanmean(day_calib, 0)[:min_x]
            sns.stripplot(x=np.arange(len_array), y=sot_array[mm, :], color= color_mapping[mouse], ax=ax2)

        sns.regplot(x=np.arange(len_array), y=np.nanmean(sot_array, 0), ax=ax2, scatter=False)
        sns.regplot(x=np.arange(len_array), y=np.nanmean(sot_array, 0), ax=ax1)
        sns.regplot(x=np.arange(len_array), y=np.nanmean(sot_calib, 0), ax=ax1, color='lightgray')
        ut_plots.get_reg_pvalues(np.arange(len_array), np.nanmean(sot_array, 0), ax1, 1, height=np.nanmean(sot_array))






    # result_df = df.groupby(['neuron', 'stim', 'mice', 'session_path', 'neuron_type']).agg({
    #     'next': 'mean',  # Average of 'next'
    #     'zscore_next': 'mean',  # Average of 'score_next'
    #     'event_index': 'count'  # Count of 'event_index'}).reset_index()
    # }).reset_index()
    # sns.lmplot(x='event_index', y='next', data=result_df[result_df.neuron_type=='na'], hue='stim')