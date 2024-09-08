__author__ = 'Nuria'

import copy

import scipy.io
import imageio
import cv2
import pandas as pd
import seaborn as sns
import numpy as np

from PIL import Image
from pathlib import Path
from matplotlib import interactive
from scipy.ndimage import binary_dilation, binary_erosion

from preprocess.prepare_data import obtain_dffs, create_time_locked_array
from utils import util_plots as ut_plots
from utils.analysis_constants import AnalysisConstants
from utils.util_plots import generate_palette_all_figures

interactive(True)

def plot_bout_init():
    color_mapping = generate_palette_all_figures()
    df_motion = pd.read_parquet(Path("C:/Users/Nuria/Documents/DATA/D1exp/df_data") / "df_motion.parquet")
    feature = "initiations_per_min"
    df_group = df_motion.groupby(["mice", "Laser"]).mean().reset_index()
    df_group = df_group[df_group.Laser.isin(['ON', 'OFF'])]
    fig1, ax1 = ut_plots.open_plot()
    sns.lineplot(data=df_group, x='Laser', y=feature, hue='mice', palette=color_mapping, ax=ax1)
    sns.stripplot(data=df_group, x="Laser", y=feature, hue='mice', palette=color_mapping, s=10,
                  marker="D", jitter=False, ax=ax1)
    ax1.set_ylim([0.5, 3])
    a = df_group[df_group.Laser == 'ON'][feature]
    b = df_group[df_group.Laser == 'OFF'][feature]
    ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=False)

    df_group = df_motion.groupby(["mice", "Laser"]).mean().reset_index()
    df_group = df_group[df_group.Laser.isin(['BMI', 'OFF'])]
    fig1, ax1 = ut_plots.open_plot()
    sns.lineplot(data=df_group, x='Laser', y=feature, hue='mice', palette=color_mapping, ax=ax1)
    sns.stripplot(data=df_group, x="Laser", y=feature, hue='mice', palette=color_mapping, s=10,
                  marker="D", jitter=False, ax=ax1)
    ax1.set_ylim([0.5, 3])
    a = df_group[df_group.Laser == 'BMI'][feature]
    b = df_group[df_group.Laser == 'OFF'][feature]
    ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=False)


def plot_motor_days(df_motion: pd.DataFrame, folder_plots: Path):
    """ function to plot the changes on initial motion by day """
    feature = "initiations_per_min"
    color_mapping = ut_plots.generate_palette_all_figures()
    df_motion = df_motion[df_motion.Laser == 'OFF']
    df_motion['days'] = df_motion.groupby('mice')['session_date'].rank(method='first', ascending=True)
    min_count = df_motion.groupby('mice')['days'].max().min()
    df_min = df_motion[df_motion.days.isin([2, min_count])]
    df_min.loc[df_min.days == 2, 'days'] = 0
    df_min.loc[df_min.days > 2, 'days'] = 1
    fig1, ax1 = ut_plots.open_plot()
    sns.lineplot(data=df_min, x='days', y=feature, hue='mice', palette=color_mapping, ax=ax1)
    sns.stripplot(data=df_min, x='days', y=feature, hue='mice', palette=color_mapping, s=10,
                  marker="D", jitter=False, ax=ax1)
    a = df_min[df_min.days == 0][feature]
    b = df_min[df_min.days == 1][feature]
    ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=False)


def plot_example_image(folder_suite2p: Path, folder_plots: Path):
    ## I've used folder_suite2p ='D:/data/process/m16/221116/D05/suite2p/plane0'
    ops_after = np.load(Path(folder_suite2p) / "ops.npy", allow_pickle=True)
    ops_after = ops_after.take(0)
    neurons = np.load(Path(folder_suite2p) / "stat.npy", allow_pickle=True)
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    dff = obtain_dffs(folder_suite2p, True)
    aux_stim_time = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
    stim_time = aux_stim_time.take(0)
    stim_index = stim_time["stim_time"]

    G = copy.deepcopy(ops_after["meanImg"])
    B = np.zeros((512, 512))
    R = np.zeros((512, 512))
    for nn, neuron in enumerate(neurons):
        if bool(is_cell[nn, 0]):
            if nn in direct_neurons['E1']:
                for pix in np.arange(neuron["xpix"].shape[0]):
                    R[neuron["ypix"][pix], neuron["xpix"][pix]] = 1
            elif nn in direct_neurons['E2']:
                for pix in np.arange(neuron["xpix"].shape[0]):
                    B[neuron["ypix"][pix], neuron["xpix"][pix]] = 1
    dilated_image_R = binary_dilation(binary_erosion(R, iterations=1), iterations=3)
    border_image_R = np.logical_xor(dilated_image_R, binary_dilation(binary_erosion(R, iterations=1)))
    dilated_image_B = binary_dilation(binary_erosion(B, iterations=1), iterations=3)
    border_image_B = np.logical_xor(dilated_image_B, binary_dilation(binary_erosion(B, iterations=1)))
    R_def = binary_dilation(border_image_R)
    B_def = binary_dilation(border_image_B)
    G[R_def] = 0
    G[B_def] = 0
    RGB = np.stack((R_def * 255, G / G.max() * 255, B_def * 255),
                   axis=2)
    imageio.imwrite(folder_plots / "kk.tif", RGB.astype(np.uint8), format="tiff")

    dff_time_locked = create_time_locked_array(dff, stim_time['stim_time'])
    cursor_time_locked = np.nansum(dff_time_locked[direct_neurons["E2"], :, :], 0) - \
                         np.nansum(dff_time_locked[direct_neurons["E1"], :, :], 0)

    df_ensemble = pd.DataFrame()
    df_ensemble['E1'] = np.nansum(dff[direct_neurons['E1'], :], 0)
    df_ensemble['E2'] = np.nansum(dff[direct_neurons['E2'], :], 0)

    fig1, ax1 = ut_plots.open_plot()
    xx = np.arange(-2, 2, 1 / AnalysisConstants.framerate)
    ax1.plot(xx, np.nanmean(dff_time_locked[direct_neurons["E1"], :, 240:360], 1).T, 'green')
    ax1.plot(xx, np.nanmean(dff_time_locked[direct_neurons["E2"], :, 240:360], 1).T, 'red')
    ax1.axvline(x=0, color="#990000", linestyle="--")

    fig2, ax2 = ut_plots.open_plot()
    ax2.plot(xx, cursor_time_locked[:, 240:360].T, 'k')
    ax2.plot(xx, np.nanmean(cursor_time_locked[:, 240:360], 0), 'r')
    ax2.axvline(x=0, color="#990000", linestyle="--")

    smooth_filt = np.ones(2) / 2
    df_ensemble['E1_smooth'] = np.convolve(df_ensemble["E1"], smooth_filt, 'same')
    df_ensemble['E2_smooth'] = np.convolve(df_ensemble["E2"], smooth_filt, 'same')
    fig3, ax3 = ut_plots.open_plot((8, 8))
    frame_aux = int(AnalysisConstants.framerate * 3)
    # sns.scatterplot(data=df_ensemble.iloc[: 9000], x="E1", y="E2", color="lightgray", ax=ax3)
    sns.histplot(data=df_ensemble.iloc[: 27000], x="E1", y="E2", color="lightgray")
    ax3.plot(df_ensemble.loc[stim_index[-1] - frame_aux: stim_index[-1], "E1"],
             df_ensemble.loc[stim_index[-1] - frame_aux: stim_index[-1], "E2"], 'k')
    ax3.plot(df_ensemble.loc[stim_index[-8] - frame_aux: stim_index[-8], "E1"],
             df_ensemble.loc[stim_index[-8] - frame_aux: stim_index[-8], "E2"], 'orange')
    ax3.plot(df_ensemble.loc[stim_index[-12] - frame_aux: stim_index[-12], "E1"],
             df_ensemble.loc[stim_index[-12] - frame_aux: stim_index[-12], "E2"], 'red')
    x = np.linspace(df_ensemble["E1"].min(), df_ensemble["E1"].max(), 100)
    T1 = 3
    T2 = -0.04
    ax3.plot(x, x + T1)
    ax3.plot(x, x - T2)
    ax3.set_xlim([-0.1, 2.6])

    cursor = np.nansum(dff[direct_neurons["E2"], :], 0) - np.nansum(dff[direct_neurons["E1"], :], 0)
    fig4, ax4 = ut_plots.open_plot()
    sns.histplot(y=cursor[:27000], binrange=[-0.5, 4], bins=50, stat="probability")
    ax4.axhline(y=T1, color='#990000', linestyle='--')
    ax4.axhline(y=T2,  color='#990000', linestyle='--')
    ax4.set_ylim([-0.6, 4])

    fig5, ax5 = ut_plots.open_plot()
    ax5.plot(cursor[stim_index[-12] - frame_aux: stim_index[-12]])
    ax5.axhline(y=T1, color='#990000', linestyle='--')
    ax5.axhline(y=T2,  color='#990000', linestyle='--')
    ax5.set_ylim([-0.6, 4])

    fig6, ax6 = ut_plots.open_plot()
    plot_frames = np.arange(stim_index[-12] - 400, stim_index[-12])
    ax6.plot(np.arange(len(plot_frames)), df_ensemble.loc[plot_frames, "E1"] -
              df_ensemble.loc[plot_frames, "E1"].mean(), 'blue')
    ax6.plot(np.arange(len(plot_frames)), df_ensemble.loc[plot_frames, "E2"] -
              df_ensemble.loc[plot_frames, "E2"].mean(), 'gray')

    cursor_trunc = copy.deepcopy(-cursor)
    cursor_trunc[cursor_trunc < -T1] = -T1
    cursor_trunc[cursor_trunc > -T2] = -T2
    cursor_trunc -= T2
    fb_cal_a = 6000
    fb_cal_cursor_range = T1 - T2
    fb_cal_b = (np.log(19000) - np.log(fb_cal_a)) / fb_cal_cursor_range
    freq = fb_cal_a * np.exp(fb_cal_b * (np.arange(cursor_trunc.max(), cursor_trunc.min(), -0.1) + T1))
    fig7, ax7 = ut_plots.open_plot()
    ax7.plot(freq, np.arange(cursor_trunc.max(), cursor_trunc.min(), -0.1))


def plot_snr(df_snr, df_learning):
    """ function to plot the snr for supp fig 2"""
    df_snr = df_snr.drop('session_path', axis=1)
    for snr in ['snr_dn', 'snr_all']:
        fig1, ax1 = ut_plots.open_plot()
        sns.boxplot(data=df_snr, y=snr, x='mice_name', ax=ax1)
        ax1.set_xlabel('mice')
        ax1.set_ylabel(snr)

        fig2, ax2 = ut_plots.open_plot()
        experiments = AnalysisConstants.experiment_types
        df_group = df_snr.groupby(["mice_name", "experiment_type"]).mean().reset_index()
        sns.boxplot(data=df_group, y=snr, x='experiment_type', order=AnalysisConstants.experiment_types, ax=ax2)
        a = df_group[df_group.experiment_type == 'D1act'][snr]
        for ee, exp in enumerate(experiments[1:]):
            b = df_group[df_group.experiment_type == exp][snr]
            ut_plots.get_pvalues(a, b, ax2, pos=ee+1, height=a[~np.isnan(a)].max(), ind=True)
        ax2.set_xlabel('experiment_type')
        ax2.set_ylabel(snr)

        fig3, ax3 = ut_plots.open_plot()
        experiments = AnalysisConstants.experiment_types
        sns.boxplot(data=df_snr, y=snr, x='experiment_type', order=AnalysisConstants.experiment_types, ax=ax3)
        a = df_snr[df_snr.experiment_type == 'D1act'][snr]
        for ee, exp in enumerate(experiments[1:]):
            b = df_snr[df_snr.experiment_type == exp][snr]
            ut_plots.get_pvalues(a, b, ax3, pos=ee + 1, height=a[~np.isnan(a)].max(), ind=True)
        ax2.set_xlabel('experiment_type')
        ax2.set_ylabel(snr)

        fig5, ax5 = ut_plots.open_plot()
        sns.regplot(data=df_snr, y=snr, x='day_index', ax=ax5)
        ut_plots.get_reg_pvalues(df_snr[snr], df_snr['day_index'], ax5, 1, 3)
        ax5.set_xlabel('day_index')
        ax5.set_ylabel(snr)

    fig6, ax6 = ut_plots.open_plot()
    merged_df = df_snr.merge(df_learning, on='session_path', how='inner')
    pearson_corr = merged_df['snr_dn'].corr(merged_df['gain'], method='pearson')
    merged_df['color'] = merged_df['mice_name'].map(color_mapping)
    for idx, row in merged_df.iterrows():
        plt.scatter(row['snr_dn'], row['gain'], color=row['color'], edgecolor='w')
    sns.regplot(data=merged_df, x='snr_dn', y='gain', scatter=False)
    ax6.text(5, 5, 'r2= ' + str(pearson_corr ** 2))


def plot_online_motion(df: pd.DataFrame):
    """ function to plot the online motion"""
    color_mapping = ut_plots.generate_palette_all_figures()
    df_melted = pd.melt(df.drop('day_index', axis=1), id_vars=['mice', 'trial', 'type', 'size', 'session_path'],
                        value_vars=['before', 'hit', 'reward', 'random'], var_name='time', value_name='pixels')
    df_melted['pixels_%'] = df_melted['pixels'] / df_melted['size'] * 100

    df_max = df_melted[df_melted.type=='max'].drop(columns=['type'])
    df_group = df_max.groupby(['mice', 'session_path', 'time']).median().reset_index()

    df_group = (df_group.drop(['session_path', 'trial'], axis=1).groupby(['mice', 'time']).
                mean().reset_index())

    fig1, ax1 = ut_plots.open_plot()
    sns.boxplot(data=df_group, x='time', y='pixels', order=['random', 'before', 'hit', 'reward'], color='gray', ax=ax1)
    sns.stripplot(data=df_group, x='time', y='pixels', hue='mice', palette=color_mapping, jitter=True, ax=ax1)
    a = df_group[df_group.time == 'reward']['pixels']
    b = df_group[df_group.time == 'random']['pixels']
    ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)


def plot_online_corr(df: pd.DataFrame):
    """ function to plot the online correlation"""
    color_mapping = ut_plots.generate_palette_all_figures()
    df_melted = pd.melt(df.drop('day_index', axis=1), id_vars=['mice', 'trial', 'type', 'session_path'],
                        value_vars=['hit', 'reward', 'random'], var_name='time', value_name='corr')

    df_image = df_melted[df_melted.type=='image'].drop(columns=['type'])
    df_im_group = df_image.groupby(['mice', 'session_path', 'time']).mean().reset_index()
    df_im_group = (df_im_group.drop(['session_path', 'trial'], axis=1).groupby(['mice', 'time']).
                mean().reset_index())

    fig2, ax2 = ut_plots.open_plot()
    sns.boxplot(data=df_im_group, x='time', y='corr', order=['random', 'hit', 'reward'], color='gray', ax=ax2,
                flierprops=dict(marker='D', color='black', markersize=4, markerfacecolor='black'))
    sns.stripplot(data=df_im_group, x='time', y='corr', order=['random', 'hit', 'reward'], hue='mice',
                  palette=color_mapping, jitter=True, ax=ax2)
    a = df_im_group[df_im_group.time == 'hit']['corr']
    b = df_im_group[df_im_group.time == 'random']['corr']
    c = df_im_group[df_im_group.time == 'reward']['corr']
    ut_plots.get_pvalues(a, b, ax2, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)
    ut_plots.get_pvalues(a, c, ax2, pos=1.5, height=b[~np.isnan(b)].max(), ind=False)
    ax2.set_xlabel('image')


    df_E1 = df_melted[df_melted.type == 'E1'].drop(columns=['type'])
    df_E1_group = df_E1.groupby(['mice', 'session_path', 'time']).mean().reset_index()
    df_E1_group = (df_E1_group.drop(['session_path', 'trial'], axis=1).groupby(['mice', 'time']).
                mean().reset_index())

    fig3, ax3 = ut_plots.open_plot()
    sns.boxplot(data=df_E1_group, x='time', y='corr', order=['random', 'hit', 'reward'], color='gray', ax=ax3,
                flierprops=dict(marker='D', color='black', markersize=4, markerfacecolor='black'))
    sns.stripplot(data=df_E1_group, x='time', y='corr', hue='mice', order=['random', 'hit', 'reward'],
                  palette=color_mapping, jitter=True, ax=ax3)
    a = df_E1_group[df_E1_group.time == 'hit']['corr']
    b = df_E1_group[df_E1_group.time == 'random']['corr']
    c = df_E1_group[df_E1_group.time == 'reward']['corr']
    ut_plots.get_pvalues(a, b, ax3, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)
    ut_plots.get_pvalues(a, c, ax3, pos=1.5, height=b[~np.isnan(b)].max(), ind=False)
    ax3.set_xlabel('E1')
    ax3.set_ylim([0.05, 0.20])

    df_E2 = df_melted[df_melted.type == 'E2'].drop(columns=['type'])
    df_E2_group = df_E2.groupby(['mice', 'session_path', 'time']).mean().reset_index()
    df_E2_group = (df_E2_group.drop(['session_path', 'trial'], axis=1).groupby(['mice', 'time']).
                mean().reset_index())

    fig4, ax4 = ut_plots.open_plot()
    sns.boxplot(data=df_E2_group, x='time', y='corr', order=['random', 'hit', 'reward'], color='gray', ax=ax4,
                flierprops=dict(marker='D', color='black', markersize=4, markerfacecolor='black'))
    sns.stripplot(data=df_E2_group, x='time', y='corr', order=['random', 'hit', 'reward'], hue='mice',
                  palette=color_mapping, jitter=True, ax=ax4)
    a = df_E2_group[df_E2_group.time == 'hit']['corr']
    b = df_E2_group[df_E2_group.time == 'random']['corr']
    c = df_E2_group[df_E2_group.time == 'reward']['corr']
    ut_plots.get_pvalues(a, b, ax4, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)
    ut_plots.get_pvalues(a, c, ax4, pos=1.5, height=b[~np.isnan(b)].max(), ind=False)
    ax4.set_xlabel('E2')
    ax4.set_ylim([0.05, 0.20])

    df_cursor = df_melted[df_melted.type == 'cursor'].drop(columns=['type'])
    df_cursor_group = df_cursor.groupby(['mice', 'session_path', 'time']).mean().reset_index()
    df_cursor_group = (df_cursor.drop(['session_path', 'trial'], axis=1).groupby(['mice', 'time']).
                mean().reset_index())

    fig5, ax5 = ut_plots.open_plot()
    sns.boxplot(data=df_cursor_group, x='time', y='corr', order=['random', 'hit', 'reward'], color='gray', ax=ax5,
                flierprops=dict(marker='D', color='black', markersize=4, markerfacecolor='black'))
    sns.stripplot(data=df_cursor_group, x='time', y='corr', hue='mice', order=['random', 'hit', 'reward'],
                  palette=color_mapping, jitter=True, ax=ax5)
    a = df_cursor_group[df_cursor_group.time == 'hit']['corr']
    b = df_cursor_group[df_cursor_group.time == 'random']['corr']
    c = df_cursor_group[df_cursor_group.time == 'reward']['corr']
    ut_plots.get_pvalues(a, b, ax5, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)
    ut_plots.get_pvalues(a, c, ax5, pos=1.5, height=b[~np.isnan(b)].max(), ind=False)
    ax5.set_xlabel('cursor')
    ax5.set_ylim([0.05, 0.20])


def plot_distance_direct_neurons(df: pd.DataFrame):
    """ function to plot the distance among sets of ensemble neurons """
    df = df[df.experiment_type == 'D1act']
    df['color'] = df['mice_name'].map(color_mapping)

    df['type_code'] = pd.Categorical(df['type']).codes
    df['distance_pix'] = df['distance'] / 512 * 300

    # Manually plot the points with jitter and custom colors
    for idx, row in df.iterrows():
        jittered_x = row['type_code'] + np.random.uniform(-0.2, 0.2)  # Adding jitter
        plt.scatter(jittered_x, row['distance_pix'], color=row['color'], edgecolor=row['color'])


    plt.xlabel('Type')
    plt.ylabel('Distance')

    type_categories = pd.Categorical(df['type']).categories
    plt.xticks(ticks=np.arange(len(type_categories)), labels=type_categories)


def plot_snr_e1e2(df: pd.DataFrame):
    """ function to plot the SNR of direct neurons """
    color_mapping = ut_plots.generate_palette_all_figures()
    df = df[df.experiment_type == 'D1act']
    df = df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(inplace=True)
    df['di'] = 'direct'
    df.loc[df.type=='all','di'] = 'indirect'
    df_group = df.drop(columns=['experiment_type', 'day_index', 'type']).groupby(['mice_name', 'di', 'session_path']).mean().reset_index()
    df_group = df_group.drop(columns=['session_path']).groupby(['mice_name', 'di']).mean().reset_index()
    fig1, ax1 = ut_plots.open_plot()
    sns.lineplot(data=df_group, y='snr', x='di', hue='mice_name', palette=color_mapping, ax=ax1)
    sns.stripplot(data=df_group, y='snr', x='di', hue='mice_name', palette=color_mapping, s=10, marker="D", jitter=False, ax=ax1)
    a = df_group[df_group.di == 'direct']['snr']
    b = df_group[df_group.di == 'indirect']['snr']
    ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)


def plot_traces(folder_suite2p: Path, file_path:Path):
    """ function to plot traces online/offline """
    # session used /m28/230414/D06

    dff = np.load(Path(folder_suite2p) / "dff.npy")
    f = np.load(Path(folder_suite2p) / "f.npy")
    index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
    index_dict = index_aux.take(0)
    indices = index_dict['target_index']
    aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
    direct_neurons = aux_dn.take(0)
    ensemble = direct_neurons['E1'] + direct_neurons['E2']
    dff_dn = dff[ensemble, :]
    pre_f_dn = f[ensemble, :]
    bmi_online = sio.loadmat(file_path, simplify_cells=True)
    pre_f_online = bmi_online['data']['bmiAct']
    f_dn = np.full(pre_f_dn.shape, np.nan)
    f_dn_online = np.full(pre_f_online.shape, np.nan)
    for neuron in np.arange(f_dn.shape[0]):
        smooth_filt = np.ones(AnalysisConstants.dff_win) / AnalysisConstants.dff_win
        f_dn[neuron, AnalysisConstants.dff_win - 1:] = np.convolve(pre_f_dn[neuron,:], smooth_filt, 'valid')
        f_dn_online[neuron, AnalysisConstants.dff_win - 1:] = np.convolve(pre_f_online[neuron, :], smooth_filt, 'valid')
    dff_tl = pp.create_time_locked_array(dff_dn, indices, (100, 0))
    f_tl = pp.create_time_locked_array(f_dn, indices, (100, 0))
    f_tl_online = pp.create_time_locked_array(f_dn_online[[0, 1, 3, 2],:], np.where(bmi_online['data']['selfHits'])[0], (100, 0))
    for trial in np.arange(f_tl.shape[1]-20, f_tl.shape[1]):
        fig1, ax1, ax2, ax3, ax4 = ut_plots.open_4subplots_line()
        axes = [ax1, ax2, ax3, ax4]
        for neuron in np.arange(f_dn.shape[0]):
            ax = axes[neuron]
            for arr in [f_tl_online[neuron, :, :], f_tl[neuron, :, :], dff_tl[neuron, :, :]]:
                ax.plot((arr[trial, :] - arr.min())/(arr.max() - arr.min()))
                ax.set_ylim([0,1])
                ut_plots.save_plot(fig1, ax, folder_plots, 'raw_traces_' , str(trial), False)


def plot_all_traces(sec_to_plot: int = 2):
    """ function to plot traces online/offline """
    len_to_plot = np.round(sec_to_plot * AnalysisConstants.framerate).astype(int)
    time_values = np.linspace(-sec_to_plot, 0, len_to_plot)
    df_sessions = ss.get_sessions_df(folder_list, 'D1act')
    for index, row in df_sessions.iterrows():
        folder_raw = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'raw'
        file_path = folder_raw / row['session_path'] / row['BMI_online']
        folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
        folder_processed_experiment = Path(folder_process) / row['session_path']
        folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
        dff = np.load(Path(folder_suite2p) / "dff.npy")
        index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['target_index']
        aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
        direct_neurons = aux_dn.take(0)
        ensemble = direct_neurons['E1'] + direct_neurons['E2']
        pre_f_E1 = dff[direct_neurons['E1'], :]
        pre_f_E2 = dff[direct_neurons['E2'], :]
        bmi_online = sio.loadmat(file_path, simplify_cells=True)
        pre_f_online = (bmi_online['data']['bmiAct'] - bmi_online['data']['baseVector'])/bmi_online['data']['baseVector']
        pre_f_online[:, :1800] = np.nan # from the experiment to remove initialization time
        pre_f_online_E1 = pre_f_online[:2, :]
        pre_f_online_E2 = pre_f_online[2:, :]

        number_trials = 16
        if len(indices) < number_trials:
            number_trials = len(indices)

        f_dn_E1 = np.full(pre_f_E1.shape, np.nan)
        f_dn_E2 = np.full(pre_f_E2.shape, np.nan)
        f_dn_online_E1 = np.full(pre_f_online_E1.shape, np.nan)
        f_dn_online_E2 = np.full(pre_f_online_E2.shape, np.nan)
        for neuron in np.arange(f_dn_E1.shape[0]):
            smooth_filt = np.ones(AnalysisConstants.dff_win) / AnalysisConstants.dff_win
            f_dn_E1[neuron, :] = pre_f_E1[neuron, :]
            f_dn_online_E1[neuron, AnalysisConstants.dff_win - 1:] = np.convolve(pre_f_online_E1[neuron, :], smooth_filt,
                                                                              'valid')
            f_dn_E2[neuron, :] = pre_f_E2[neuron, :]
            f_dn_online_E2[neuron, AnalysisConstants.dff_win - 1:] = np.convolve(pre_f_online_E2[neuron, :], smooth_filt,
                                                                              'valid')

        min_values_E1 = np.nanmin(f_dn_E1[:,AnalysisConstants.calibration_frames:], axis=1, keepdims=True)
        max_values_E1 = np.nanmax(f_dn_E1[:,AnalysisConstants.calibration_frames:], axis=1, keepdims=True)
        min_values_E2 = np.nanmin(f_dn_E2[:,AnalysisConstants.calibration_frames:], axis=1, keepdims=True)
        max_values_E2 = np.nanmax(f_dn_E2[:,AnalysisConstants.calibration_frames:], axis=1, keepdims=True)
        min_values_online_E1 = np.nanmin(f_dn_online_E1, axis=1, keepdims=True)
        max_values_online_E1 = np.nanmax(f_dn_online_E1, axis=1, keepdims=True)
        min_values_online_E2 = np.nanmin(f_dn_online_E2, axis=1, keepdims=True)
        max_values_online_E2 = np.nanmax(f_dn_online_E2, axis=1, keepdims=True)

        f_tl_E1 = pp.create_time_locked_array(np.nansum((f_dn_E1 - min_values_E1) / (max_values_E1 - min_values_E1),0)/2,
                                           indices, (len_to_plot, 0))
        f_tl_online_E1 = pp.create_time_locked_array(np.nansum((f_dn_online_E1 - min_values_online_E1) /
                                                               (max_values_online_E1 - min_values_online_E1),0)/2,
                                                     np.where(bmi_online['data']['selfHits'])[0], (len_to_plot, 0))
        f_tl_E2 = pp.create_time_locked_array(np.nansum((f_dn_E2 - min_values_E2) / (max_values_E2 - min_values_E2),0)/2,
                                           indices, (len_to_plot, 0))
        f_tl_online_E2 = pp.create_time_locked_array(np.nansum((f_dn_online_E2 - min_values_online_E2) /
                                                               (max_values_online_E2 - min_values_online_E2),0)/2,
                                                     np.where(bmi_online['data']['selfHits'])[0], (len_to_plot, 0))

        fig1, axes1 = plt.subplots(4, 4, figsize=(14, 14))
        axes1 = axes1.flatten()
        fig2, axes2 = plt.subplots(4, 4, figsize=(14, 14))
        axes2 = axes2.flatten()

        random_indices = np.sort(np.random.choice(np.arange(indices.shape[0]), size=number_trials, replace=False))

        for tt, trial in enumerate(random_indices):
            ax = axes1[tt]
            ax.plot(time_values, f_tl_online_E1[trial,:], label='online')
            ax.plot(time_values, f_tl_E1[trial, :], label='offline')
            ax.set_ylim([-0.1, 1.1])
            ax.set_title(f"Trial: {trial}")
            if tt < 12:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Time (s)')
            ax.legend()
            ax = axes2[tt]
            ax.plot(time_values, f_tl_online_E2[trial, :], label='online')
            ax.plot(time_values, f_tl_E2[trial, :], label='offline')
            ax.set_ylim([-0.1, 1.1])
            ax.set_title(f"Trial: {trial}")
            if tt < 12:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Time (s)')
            ax.legend()
        fig1.suptitle(f"E1 of {row['mice_name']}: {row['session_path']}")
        fig2.suptitle(f"E2 of {row['mice_name']}: {row['session_path']}")

        # Adjust layout for better spacing
        fig1.tight_layout()
        fig2.tight_layout()
        fig1.savefig(folder_plots / f"{row['mice_name']}_{row['session_date']}_E1", bbox_inches="tight")
        fig2.savefig(folder_plots / f"{row['mice_name']}_{row['session_date']}_E2", bbox_inches="tight")
        plt.close('all')


# Function to obtain the activity of each neuron
def obtain_Roi(Im, strcMask, neuropil=32, units=None):
    """
    Function to obtain the activity of each neuron, given a spatial filter
    units -> indices of the neurons in the neuronMask that we want
    Im -> Image (2D array)
    strcMask -> structure with the matrix for spatial filters with px*py*unit
    units -> list of neuron indices to return, if None, use all units
    """
    halfsize = neuropil // 2
    if units is None:
        units = range(len(strcMask['neuronMask'][0][0][0]))

    unitVals = np.zeros(len(units))
    neuropilVals = np.zeros(len(units))

    for idx, u in enumerate(units):
        posmaxx = int(strcMask['maxx'][0][0][0][u]) - 1
        posminx = int(strcMask['minx'][0][0][0][u]) - 1
        posmaxy = int(strcMask['maxy'][0][0][0][u]) - 1
        posminy = int(strcMask['miny'][0][0][0][u]) - 1

        xctr = int(strcMask['xctr'][0][0][0][u])
        yctr = int(strcMask['yctr'][0][0][0][u])

        x2size = int(np.ceil((posmaxx - posminx)/2))
        y2size = int(np.ceil((posmaxy - posminy)/2))

        Imd = Im[posminy:posmaxy + 1, posminx:posmaxx + 1].astype(float)
        Im_neuropil = Im[xctr - halfsize : xctr + halfsize, yctr - halfsize : yctr + halfsize]

        neuron_mask = strcMask['neuronMask'][0][0][0][u]/ (u+1)
        neuropil_mask = np.ones(Im_neuropil.shape)
        neuropil_mask[halfsize - x2size : halfsize + x2size, halfsize - y2size : halfsize + y2size] = 0

        unitVals[idx] = np.nansum(Imd * neuron_mask)/np.sum(neuron_mask)
        neuropilVals[idx] = np.nansum(Im_neuropil * neuropil_mask)/np.sum(neuropil_mask)

    return unitVals, neuropilVals

def prepare_video_single_session(folder_suite2p: Path, folder_raw: Path, folder_plots):
    """ function to create a video with online data"""

    import scipy.io
    import warnings
    import cv2
    import io
    from PIL import Image
    from tifffile import TiffFile
    from matplotlib.gridspec import GridSpec
    warnings.simplefilter("ignore", category=UserWarning)

    # video used m16_221116_D05
    # Load the roi_data and strcMask from .mat files
    folder_raw = Path(folder_raw)
    folder_suite2p = Path(folder_suite2p)
    im_folder = folder_raw / 'im/BMI_stim/BMI_stim_221116T111406-246/'

    bmi_online = sio.loadmat(folder_raw / 'BMI_online221116T114430.mat', simplify_cells=True)
    cursor = bmi_online['data']['cursor']
    roi_data = scipy.io.loadmat(folder_raw / 'roi_data.mat')
    im_bg = roi_data['roi_data']['im_bg'][0][0][:, :, 0]  # Assuming you want the first slice
    strcMask = scipy.io.loadmat(folder_raw / 'strcMask.mat')
    strcMask = strcMask['strcMask']

    # Load the stim_time_dict and stim_index
    stim_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
    stim_time_dict = stim_aux.take(0)
    stim_index = stim_time_dict['stim_index']

    ops_aux = np.load(Path(folder_suite2p) / "ops.npy", allow_pickle=True)
    ops = ops_aux.take(0)

    xoff = ops['xoff']
    yoff = ops['yoff']
    zoff = ops['corrXY']

    # Size parameters
    subsection_size = 32
    half_size = subsection_size // 2
    frame_rate = 30  # 30Hz frame rate
    f0_win_im = 30  # Window size for baseline calculation
    f0_win = 10  # Window size for baseline calculation
    calibration_limit = 27000  # Upper frame limit for the calibration period
    total_frames = 81000
    sec_to_plot = 5
    frames_before_hit = sec_to_plot * frame_rate
    frames_after_hit = 2
    time_values = np.linspace(-sec_to_plot - int(f0_win_im/frame_rate), frames_after_hit/frame_rate,
                              frames_before_hit+frames_after_hit + frame_rate)
    number_neurons = len(strcMask['neuronMask'][0][0][0])
    frame_width = 1920
    frame_height = 1080
    new_min = -1
    new_max = 1

    # Initialize Fbuffer and baseval
    Fbuffer = np.full((number_neurons, f0_win), np.nan)
    Fbuffer_np = np.full((number_neurons, f0_win), np.nan)
    baseBuffer_full = False

    # Select 100 random frames during the calibration period
    random_frames = np.random.choice(range(calibration_limit, total_frames), 100, replace=False)

   # Initialize lists to hold static images for each (xctr, yctr) pair
    static_images = []

    # Precompute the static images for each (xctr, yctr)
    for idx, (xctr, yctr) in enumerate(zip(strcMask['xctr'][0][0][0], strcMask['yctr'][0][0][0])):
        static_images.append(ops['meanImg'][yctr - half_size:yctr + half_size, xctr - half_size:xctr + half_size])

    # prefill the buffer of the baseline
    for frame_idx in random_frames:
        # Load the extracted frame
        adjusted_frame_idx = frame_idx - calibration_limit
        frame_filename = im_folder / f'BMI_stim_221116T111406-246_Cycle00001_Ch2_{adjusted_frame_idx:06d}.ome.tif'
        with Image.open(frame_filename) as img:
            image_buffer = np.array(img)

        # Iterate through each (xctr, yctr) pair
        for idx, (xctr, yctr) in enumerate(zip(strcMask['xctr'][0][0][0], strcMask['yctr'][0][0][0])):
            # Calculate the fluorescence for the relevant neurons using the calibration image
            unitVals, neuropilVals = obtain_Roi(image_buffer, strcMask)

            # Store in Fbuffer
            Fbuffer[:, :-1] = Fbuffer[:, 1:]  # Shift buffer
            Fbuffer[:, -1] = unitVals
            Fbuffer_np[:, :-1] = Fbuffer[:, 1:]  # Shift buffer
            Fbuffer_np[:, -1] = neuropilVals

    # initialize baseval and image_buffer
    baseval = np.nanmean(Fbuffer, 1)
    baseval_np = np.nanmean(Fbuffer_np, 1)
    smooth_buffer = np.full((4, f0_win_im, subsection_size, subsection_size), np.nan)


    # Create the figure with 3 subplots in each of the 4 rows
    fig = plt.figure(figsize=(9, 9))

    # Set up GridSpec with specific width ratios
    gs = GridSpec(4, 4, width_ratios=[0.5, 0.5, 0.75, 0.5], wspace=0.15,
                  hspace=0.2)  # Adjust 'hspace' to reduce vertical distance
    axes = []
    for i in range(4):
        ax1 = fig.add_subplot(gs[i, 0])  # First column for images
        ax2 = fig.add_subplot(gs[i, 1])  # Second column for images
        ax3 = fig.add_subplot(gs[i, 2])  # Third column for line plots
        ax4 = fig.add_subplot(gs[i, 3])  # Forth column for line plots
        axes.append([ax1, ax2, ax3, ax4])

    width, height = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video_writer = cv2.VideoWriter(Path(folder_plots) / 'ca_video.mp4', fourcc, 20, (width, height))

    # Loop through each stim_index

    for si, stim_idx in enumerate(stim_index):
        frame_to_buffer = f0_win_im
        for idx in np.arange(number_neurons):
            # Static image in the first column
            axes[idx][0].clear()
            axes[idx][0].imshow(static_images[idx], cmap='gray')
            axes[idx][0].set_title(f'Neuron {idx}')
            axes[idx][0].axis('off')

        # Initialize the fluorescence values for each (xctr, yctr) pair for the current stim_index
        fluorescence_values = np.full([frames_before_hit + frames_after_hit + frame_rate, number_neurons], np.nan)
        dff = np.full([frames_before_hit + frames_after_hit + frame_rate, number_neurons], np.nan)
        fluorescence_values_np = np.full([frames_before_hit + frames_after_hit + frame_rate, number_neurons], np.nan)

        # Loop through the frames_to_plot frames before each stim_idx
        for frame_count, frame_idx in enumerate(range(stim_idx - frames_before_hit - f0_win_im,
                                                      stim_idx + frames_after_hit)):
            if si > 0:
                if (frame_idx - frames_before_hit - f0_win_im) < stim_index[si-1]:
                    frame_to_buffer = stim_index[si-1] - frame_idx + f0_win_im
                    continue

            print('analyzing frame ' + str(frame_count) + ' of stim_idx: ' + str(si))
            if frame_idx < 0:
                continue  # Skip if the index goes negative

            # Load the extracted frame
            adjusted_frame_idx = frame_idx - calibration_limit
            frame_filename = im_folder / f'BMI_stim_221116T111406-246_Cycle00001_Ch2_{adjusted_frame_idx:06d}.ome.tif'
            with TiffFile(frame_filename) as tif:
                changing_image = tif.asarray()

            unitVals, neuropilVals = obtain_Roi(changing_image, strcMask)

            # buffer baseline for dff
            Fbuffer[:, :-1] = Fbuffer[:, 1:]
            Fbuffer[:, -1] = unitVals
            Fbuffer_np[:, :-1] = Fbuffer_np[:, 1:]  # Shift buffer
            Fbuffer_np[:, -1] = neuropilVals


            # Calculate the average fluorescence for the relevant neurons
            dff[frame_count :] = (np.nanmean(Fbuffer, axis=1) - baseval) / baseval * 100
            fluorescence_values[frame_count, :] = np.nanmean(Fbuffer, axis=1)
            fluorescence_values_np[frame_count, :] = np.nanmean(Fbuffer_np, axis=1)

            cursor = np.nansum(dff[:, 2:]/number_neurons - dff[:, :2]/number_neurons, 1)
            scaled_cursor = ((cursor - np.min(cursor)) / (np.max(cursor) - np.min(cursor))) * (new_max - new_min) + new_min

            # Iterate through each (xctr, yctr) pair
            for idx, (xctr, yctr) in enumerate(zip(strcMask['xctr'][0][0][0], strcMask['yctr'][0][0][0])):

                # Extract the subsection centered at (xctr, yctr)
                subsection = changing_image[yctr - half_size:yctr + half_size, xctr - half_size:xctr + half_size]

                # buffer the images
                smooth_buffer[idx, :-1] = smooth_buffer[idx, 1:]  # Shift buffer
                smooth_buffer[idx, -1] = subsection  # Add current frame

                if frame_count > frame_to_buffer:

                    smooth_image = np.mean(smooth_buffer[idx,:,:,:], axis=0)

                    # Changing image in the second column
                    axes[idx][1].clear()
                    axes[idx][1].imshow(smooth_image, cmap='gray', interpolation='hanning', vmin=20, vmax=200)
                    axes[idx][1].set_title(f'Neuron {idx}')
                    axes[idx][1].axis('off')

                    # Average fluorescence in the third column
                    axes[idx][2].clear()
                    axes[idx][2].plot(time_values[frame_rate:],
                                      fluorescence_values[frame_rate:, idx], color='darkgray', linewidth=2, label='Neuron') #darkorange
                    axes[idx][2].plot(time_values[frame_rate:],
                                      fluorescence_values_np[frame_rate:, idx], color='lightgray', linewidth=2, label='Neuropil') #darkorange

                    axes[idx][2].set_xlabel('Time (s)')
                    axes[idx][2].set_ylabel('Fraw')
                    axes[idx][2].set_xlim([-5, 0])
                    axes[idx][2].set_ylim([0, 200])
                    axes[idx][2].set_yticks([])
                    plt.legend

                    if idx == 3:
                        axes[idx][3].clear()
                        # cursor 4th column
                        axes[idx][3].plot(time_values[frame_count - frame_rate:frame_count],
                                          scaled_cursor[frame_count - frame_rate:frame_count],
                                          color='darkorange', linewidth=2)  # darkorange
                        if frame_count > len(scaled_cursor) - frame_rate:
                            axes[idx][3].axvline(x=0, color='red', linestyle='--', linewidth=2)
                        axes[idx][3].set_xlabel('Time (s)')
                        axes[idx][3].set_ylabel('Cursor')

                        axes[idx][3].set_ylim([-1, 1.2])
                        axes[idx][3].yaxis.tick_right()
                        axes[idx][3].yaxis.set_label_position('right')
                    elif idx == 0:
                        axes[idx][3].clear()
                        # cursor 4th column
                        axes[idx][3].plot(xoff[frame_idx - f0_win_im - frame_rate:frame_idx - f0_win_im],
                                          color='darkorange', linewidth=2)  # darkorange
                        axes[idx][3].set_ylabel('X motion')
                        axes[idx][3].set_ylim([-2, 2])
                        axes[idx][3].set_xticks([])
                        axes[idx][3].yaxis.set_label_position('right')
                        axes[idx][3].yaxis.tick_right()
                    elif idx == 1:
                        axes[idx][3].clear()
                        # cursor 4th column
                        axes[idx][3].plot(yoff[frame_idx - f0_win_im - frame_rate:frame_idx - f0_win_im],
                                          color='darkorange', linewidth=2)  # darkorange
                        axes[idx][3].set_ylabel('Y motion')
                        axes[idx][3].set_ylim([-2, 2])
                        axes[idx][3].set_xticks([])
                        axes[idx][3].yaxis.set_label_position('right')
                        axes[idx][3].yaxis.tick_right()
                    elif idx == 2:
                        axes[idx][3].clear()
                        # cursor 4th column
                        axes[idx][3].plot(zoff[frame_idx - f0_win_im - frame_rate:frame_idx - f0_win_im],
                                          color='darkorange', linewidth=2)  # darkorange
                        axes[idx][3].set_ylabel('XY corr')
                        axes[idx][3].set_ylim([0, zoff.max()])
                        axes[idx][3].set_xticks([])
                        axes[idx][3].yaxis.set_label_position('right')
                        axes[idx][3].yaxis.tick_right()

            if frame_count > frame_to_buffer:
                plt.tight_layout()
                fig.suptitle('Trial: ' + str(si), fontsize=16)
                fig.subplots_adjust(hspace=0.2)
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)

                # Open the image from the buffer
                img = Image.open(buf)
                plot_image = np.array(img)

                # Convert the PIL image to a format suitable for OpenCV
                plot_image_bgr = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)

                # Now you can write this to the video
                video_writer.write(plot_image_bgr)

        # Release the video writer and close all plots
    video_writer.release()
    plt.close('all')


def prepare_video_multiple_neuron(folder_suite2p: Path, folder_raw: Path, folder_plots, df:pandas.DataFrame):
    """ function to create a video with online data"""

    import scipy.io
    import cv2
    import io
    from PIL import Image
    from tifffile import TiffFile
    from matplotlib.gridspec import GridSpec

    # Size parameters
    subsection_size = 32
    half_size = subsection_size // 2
    frame_rate = 30  # 30Hz frame rate
    f0_win_im = 30  # Window size for baseline calculation
    f0_win = 10  # Window size for baseline calculation
    calibration_limit = 27000  # Upper frame limit for the calibration period
    total_frames = 81000
    sec_to_plot = 5
    frames_before_hit = sec_to_plot * frame_rate
    frames_after_hit = 2
    time_values = np.linspace(-5, frames_after_hit/frame_rate, frames_before_hit+frames_after_hit)
    number_neurons = 12
    frame_width = 1920
    frame_height = 1080
    new_min = -1
    new_max = 1

    # Load the roi_data and strcMask from .mat files
    df = ss.get_sessions_df(folder_list, experiment_type='D1act')
    # nos 'm16/221113/D02','m22/230419/D08',

    # names_session = ['m16/221113/D02', 'm21/230414/D06',
    #                    'm21/230416/D08', 'm22/230416/D05', 'm22/230417/D06',
    #                    'm22/230419/D08', 'm23/230419/D02', 'm23/230420/D03',
    #                    'm23/230421/D04', 'm26/230415/D07', 'm27/230414/D02',
    #                    'm28/230415/D07', 'm28/230417/D09', 'm29/230419/D02',
    #                    'm29/230421/D04']


    names_session = ['m21/230414/D06', 'm21/230416/D08', 'm22/230416/D05', 'm21/230414/D06',
                     'm29/230420/D03', 'm22/230417/D06', 'm23/230419/D02', 'm13/221113/D02',
                     'm23/230421/D04', 'm28/230415/D07', 'm28/230415/D07', 'm28/230417/D09',
                     'm21/230414/D06', 'm28/230414/D06', 'm29/230419/D02', 'm29/230421/D04']

    # Initialize an empty DataFrame
    df_selected = pd.DataFrame()

    # Iterate over names_session to ensure all occurrences are included
    for session in names_session:
        # Filter for each session path
        df_filtered = df[df['session_path'] == session]
        # Append to df_selected
        df_selected = pd.concat([df_selected, df_filtered], ignore_index=True)

    # Reset the index of the final DataFrame
    df_selected = df_selected.reset_index(drop=True)

    df_selected['neuron'] = [0, 0, 3, 3,
                             1, 1, 3, 3,
                             1, 1, 3, 2,
                             0, 0, 3, 2]

    df_selected['ensemble'] = ['E1', 'E1', 'E2', 'E2','', '', '', '', '', '','', '',  '', '', '', '']

    df_selected['stim_index'] = None
    df_selected['stim_index'] = df_selected['stim_index'].astype(object)
    min_stim = 9

    for index, row in df_selected.iterrows():
        folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
        folder_suite2p = Path(folder_process) / row['session_path'] / 'suite2p' / 'plane0'
        folder_raw = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'raw' / row['session_path']
        strcMask = scipy.io.loadmat(folder_raw / 'strcMask.mat')
        strcMask = strcMask['strcMask']
        df_selected.loc[index, 'path_im'] = (folder_raw / 'im' / 'BMI_stim'/ row['Experiment_im']/
                                             Path(row['Experiment_im'] + '_Cycle00001_Ch2'))
        df_selected.loc[index,'xctr'] = strcMask['xctr'][0][0][0][row['neuron'] ]
        df_selected.loc[index,'yctr'] = strcMask['yctr'][0][0][0][row['neuron'] ]

        # Load the stim_time_dict and stim_index
        stim_aux = np.load(Path(folder_suite2p) / "stim_time_dict.npy", allow_pickle=True)
        stim_time_dict = stim_aux.take(0)
        df_selected.at[index, 'stim_index'] = stim_time_dict['stim_index'][:min_stim]
    df_selected['xctr'] = df_selected['xctr'].astype(int)
    df_selected['yctr'] = df_selected['yctr'].astype(int)

    # Create the figure with 3 subplots in each of the 4 rows
    fig = plt.figure(figsize=(9, 9))

    # Set up GridSpec with specific width ratios
    gs = GridSpec(4, 4, width_ratios=[0.5, 0.5, 0.5, 0.5], wspace=0.05, hspace=0.05)
    axes = []
    for i in range(16):
        row = i // 4  # Determine the current row
        col = i % 4  # Determine the current column

        # Access the current subplot axes
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

    width, height = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video_writer = cv2.VideoWriter(Path(folder_plots) / 'ca_video_16.mp4', fourcc, 20, (width, height))

    #Loop through each stim_index
    for si in np.arange(min_stim):
        frame_to_buffer = -frames_before_hit
        smooth_buffer = np.full((len(df_selected), f0_win_im, subsection_size, subsection_size), np.nan)

        # Loop through the frames_to_plot frames before each stim_idx
        for frame_count in np.arange(-frames_before_hit - f0_win_im, frames_after_hit):
            print('analyzing frame ' + str(frame_count) + ' of stim_idx: ' + str(si))

            for index, row in df_selected.iterrows():
                flag_dontprint = False
                frame_idx = row['stim_index'][si] + frame_count
                if si > 0:
                    if frame_count < (row['stim_index'][si - 1] - frame_idx + 40):
                        flag_dontprint = True
                # Load the extracted frame
                adjusted_frame_idx = frame_idx - calibration_limit
                frame_filename = f"{row['path_im']}_{adjusted_frame_idx:06d}.ome.tif"
                with TiffFile(frame_filename) as tif:
                    changing_image = tif.asarray()

                # Extract the subsection centered at (xctr, yctr)
                subsection = changing_image[row['yctr'] - half_size:row['yctr'] + half_size,
                             row['xctr'] - half_size:row['xctr'] + half_size]

                # buffer the images
                smooth_buffer[index, :-1] = smooth_buffer[index, 1:]  # Shift buffer
                smooth_buffer[index, -1] = subsection  # Add current frame

                if frame_count > frame_to_buffer:
                    if flag_dontprint:
                        smooth_image = np.zeros(smooth_buffer[index, 0, :, :].shape)
                        flag_dontprint = False
                    else:
                        smooth_image = np.mean(smooth_buffer[index,:,:,:], axis=0)

                    # Changing image in the second column
                    axes[index].clear()
                    axes[index].imshow(smooth_image, cmap='gray', interpolation='hanning', vmin=20, vmax=200)
                    axes[index].axis('off')
                    axes[index].set_title(row['ensemble'])

            if frame_count > frame_to_buffer:
                plt.tight_layout()
                if frame_count >= -5:
                    fig.suptitle('HIT', fontsize=20)
                else:
                    fig.suptitle('Trial: ' + str(si), fontsize=16)
                fig.subplots_adjust(hspace=0.2)
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)

                # Open the image from the buffer
                img = Image.open(buf)
                plot_image = np.array(img)

                # Convert the PIL image to a format suitable for OpenCV
                plot_image_bgr = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)

                # Now you can write this to the video
                video_writer.write(plot_image_bgr)

        # Release the video writer and close all plots
    video_writer.release()
    plt.close('all')


def plot_corr_on_offline(df: pd.DataFrame):
    """ functio to plot the figures for the off-online correlation"""
    color_mapping = ut_plots.generate_palette_all_figures()
    df_group = df.drop(['session_path'], axis=1).groupby(['mice', 'time']).mean().reset_index()
    fig1, ax1 = ut_plots.open_plot()
    sns.boxplot(data=df_group, y='r2_E1', x='time', color='gray', ax=ax1)
    sns.stripplot(data=df_group, y='r2_E1', x='time', hue='mice', palette=color_mapping, jitter=True, ax=ax1)
    a = df_group[df_group.time == 'before']['r2_E1']
    b = df_group[df_group.time == 'hit']['r2_E1']
    c = df_group[df_group.time == 'reward']['r2_E1']
    ax1.set_ylim([0.14, 0.75])
    ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)
    ut_plots.get_pvalues(c, b, ax1, pos=1.5, height=b[~np.isnan(b)].max(), ind=False)

    fig2, ax2 = ut_plots.open_plot()
    sns.boxplot(data=df_group, y='r2_E2', x='time', color='gray', ax=ax2)
    sns.stripplot(data=df_group, y='r2_E2', x='time', hue='mice', palette=color_mapping, jitter=True, ax=ax2)
    a = df_group[df_group.time == 'before']['r2_E2']
    b = df_group[df_group.time == 'hit']['r2_E2']
    c = df_group[df_group.time == 'reward']['r2_E2']
    ax2.set_ylim([0.14, 0.75])
    ut_plots.get_pvalues(a, b, ax2, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)
    ut_plots.get_pvalues(c, b, ax2, pos=1.5, height=b[~np.isnan(b)].max(), ind=False)


def plot_online_comparison(df:pd.DataFrame):
    """ function to plot the comparison of E1 and E2 neurons """
    color_mapping = ut_plots.generate_palette_all_figures()
    df_group = df.drop(['session_path'], axis=1).groupby(['mice', 'type']).mean().reset_index()
    fig1, ax1 = ut_plots.open_plot()
    # sns.boxplot(data=df, y='std', x='Type', color='gray', ax=ax1)
    sns.stripplot(data=df, y='std', x='type', hue='mice', palette=color_mapping, jitter=True, ax=ax1)
    a = df[df.type == 'E1']['std']
    b = df[df.type == 'E2']['std']
    ax1.legend().remove()
    ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)

    fig2, ax2 = ut_plots.open_plot()
    # sns.boxplot(data=df, y='pks', x='Type', color='gray', ax=ax2)
    sns.stripplot(data=df, y='pks', x='type', hue='mice', palette=color_mapping, jitter=True, ax=ax2)
    a = df[df.type == 'E1']['pks']
    b = df[df.type == 'E2']['pks']
    ax2.legend().remove()
    ut_plots.get_pvalues(a, b, ax2, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)

    fig3, ax3 = ut_plots.open_plot()
    # sns.boxplot(data=df.dropna(), y='ampl', x='Type', color='gray', ax=ax3)
    sns.stripplot(data=df.dropna(), y='ampl', x='type', hue='mice', palette=color_mapping, jitter=True, ax=ax3)
    a = df.dropna()[df.dropna().type == 'E1']['ampl']
    b = df.dropna()[df.dropna().type == 'E2']['ampl']
    ax3.legend().remove()
    ut_plots.get_pvalues(a, b, ax3, pos=0.5, height=b[~np.isnan(b)].max(), ind=True)


def plot_comparison_trial(df:pd.DataFrame):
    """ function to plot the comparison of E1 and E2 neurons """
    color_mapping = ut_plots.generate_palette_all_figures()
    df_group = df.drop(['session_path'], axis=1).groupby(['mice', 'type', 'trial']).mean().reset_index()
    fig1, ax1 = ut_plots.open_plot()
    # sns.boxplot(data=df, y='std', x='Type', color='gray', ax=ax1)
    sns.regplot(data=df_group[df_group.type == 'E1'], y='std', x='trial',ax=ax1, label='E1')
    sns.regplot(data=df_group[df_group.type == 'E2'], y='std', x='trial', ax=ax1, label='E2')
    a = df_group[df_group.type == 'E1']['std']
    b = df_group[df_group.type == 'E2']['std']
    plt.legend()
    # ut_plots.get_anova_pvalues(a, b, axis=0, ax=ax1, pos=10, height=0.6)

    fig2, ax2 = ut_plots.open_plot()
    # sns.boxplot(data=df, y='pks', x='Type', color='gray', ax=ax2)
    sns.regplot(data=df_group[df_group.type == 'E1'], y='pks', x='trial',ax=ax2, label='E1')
    sns.regplot(data=df_group[df_group.type == 'E2'], y='pks', x='trial', ax=ax2, label='E2')
    a = df_group[df_group.type == 'E1']['pks']
    b = df_group[df_group.type == 'E2']['pks']
    plt.legend()
    # ut_plots.get_anova_pvalues(a, b, axis=0, ax=ax2, pos=10, height=0.6)

    fig3, ax3 = ut_plots.open_plot()
    # sns.boxplot(data=df.dropna(), y='ampl', x='Type', color='gray', ax=ax3)
    sns.regplot(data=df_group[df_group.type == 'E1'], y='ampl', x='trial',ax=ax3, label='E1')
    sns.regplot(data=df_group[df_group.type == 'E2'], y='ampl', x='trial', ax=ax3, label='E2')
    a = df_group[df_group.type == 'E1']['ampl']
    b = df_group[df_group.type == 'E2']['ampl']
    plt.legend()
    # ut_plots.get_anova_pvalues(a, b, axis=0, ax=ax3, pos=10, height=0.6)


def prepare_traces_all_session(folder_plots):
    """ function to create a video with online data"""

    from PIL import Image
    from tifffile import TiffFile

    subsection_size = 32
    half_size = subsection_size // 2
    frame_rate = 30  # 30Hz frame rate
    f0_win = 10  # Window size for baseline calculation
    f0_win_im = 30
    calibration_limit = 27000  # Upper frame limit for the calibration period
    total_frames = 81000
    sec_to_plot = 2
    frames_before_hit = sec_to_plot * frame_rate
    frames_after_hit = 2
    time_values = np.linspace(-sec_to_plot - int(f0_win_im / frame_rate), frames_after_hit / frame_rate,
                              frames_before_hit + frames_after_hit + frame_rate)

    df_sessions = ss.get_sessions_df(folder_list, 'D1act')
    for index, row in df_sessions.iterrows():
        folder_raw = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'raw'
        file_path = folder_raw / row['session_path'] / row['BMI_online']
        folder_process = Path(folder_list[ss.find_folder_path(row['mice_name'])]) / 'process'
        folder_processed_experiment = Path(folder_process) / row['session_path']
        folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
        im_folder = folder_raw / row['session_path'] / 'im/BMI_stim' / row['Experiment_im']

        strcMask = scipy.io.loadmat(folder_raw / row['session_path'] / 'strcMask.mat')
        strcMask = strcMask['strcMask']

        # Load the stim_time_dict and stim_index
        index_aux = np.load(Path(folder_suite2p) / "target_time_dict.npy", allow_pickle=True)
        index_dict = index_aux.take(0)
        indices = index_dict['target_index']

        # Initialize Fbuffer
        number_neurons = len(strcMask['neuronMask'][0][0][0])
        Fbuffer = np.full((number_neurons, f0_win), np.nan)
        Fbuffer_np = np.full((number_neurons, f0_win), np.nan)

        number_trials = 10
        fig1, axes1 = plt.subplots(number_trials, 4, figsize=(6, 14))
        axes1 = axes1.flatten()

        if len(indices) < number_trials:
            number_trials = len(indices)

        random_indices = np.sort(np.random.choice(indices[~np.append(False, np.diff(indices)<(frames_before_hit+30))],
                                                  size=number_trials, replace=False))

        for si, stim_idx in enumerate(random_indices):
            frame_to_buffer = f0_win_im

            # Initialize the fluorescence values for each (xctr, yctr) pair for the current stim_index
            fluorescence_values = np.full([frames_before_hit + frames_after_hit + frame_rate, number_neurons], np.nan)
            dff = np.full([frames_before_hit + frames_after_hit + frame_rate, number_neurons], np.nan)
            fluorescence_values_np = np.full([frames_before_hit + frames_after_hit + frame_rate, number_neurons], np.nan)

            # Loop through the frames_to_plot frames before each stim_idx
            for frame_count, frame_idx in enumerate(range(stim_idx - frames_before_hit - f0_win_im,
                                                          stim_idx + frames_after_hit)):

                print('analyzing ' + row['session_path'] + ': frame ' + str(frame_count) + ' of stim_idx: ' + str(si))
                if frame_idx < 0:
                    continue  # Skip if the index goes negative

                # Load the extracted frame
                adjusted_frame_idx = frame_idx - calibration_limit
                frame_filename = im_folder / f"{row['Experiment_im']}_Cycle00001_Ch2_{adjusted_frame_idx:06d}.ome.tif"
                with TiffFile(frame_filename) as tif:
                    changing_image = tif.asarray()

                unitVals, neuropilVals = obtain_Roi(changing_image, strcMask)

                # buffer baseline for dff
                Fbuffer[:, :-1] = Fbuffer[:, 1:]
                Fbuffer[:, -1] = unitVals
                Fbuffer_np[:, :-1] = Fbuffer_np[:, 1:]  # Shift buffer
                Fbuffer_np[:, -1] = neuropilVals

                # Calculate the average fluorescence for the relevant neurons
                fluorescence_values[frame_count, :] = np.nanmean(Fbuffer, axis=1)
                fluorescence_values_np[frame_count, :] = np.nanmean(Fbuffer_np, axis=1)

                # Iterate through each (xctr, yctr) pair

            for idx, (xctr, yctr) in enumerate(zip(strcMask['xctr'][0][0][0], strcMask['yctr'][0][0][0])):
                if frame_count > frame_to_buffer:
                    # Average fluorescence in the third column
                    ax = axes1[4*si+idx]
                    ax.plot(time_values[frame_rate:], fluorescence_values[frame_rate:, idx],
                            color='darkgray', linewidth=1, label='Neuron') #darkorange
                    ax.plot(time_values[frame_rate:], fluorescence_values_np[frame_rate:, idx],
                            color='lightgray', linewidth=1, label='Neuropil') #darkorange

                    ax.set_xlim([-sec_to_plot, 0])
                    ax.set_ylim([0, fluorescence_values.max()])
                    ax.set_yticks([])
                    if si == 0:
                        if idx <=1:
                            ax.set_title('E1')
                        else:
                            ax.set_title('E2')
                    if si < number_trials - 1:
                        ax.set_xticks([])
                    else:
                        ax.set_xlabel('Time (s)')
                    if idx == 3:
                        ax.yaxis.set_label_position('right')
                        ax.set_ylabel('Trial ' + str(np.where(indices==stim_idx)[0][0]))
                    elif idx == 0:
                        ax.set_ylabel('Fraw')
        fig1.suptitle(f"{row['mice_name']}: {row['session_path']}")
        fig1.savefig(folder_plots / f"{row['mice_name']}_{row['session_date']}_traces_trial", bbox_inches="tight")
        plt.close('all')







