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
    sns.boxplot(data=df_im_group, x='time', y='corr', order=['random', 'hit', 'reward'], color='gray', ax=ax2)
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
    sns.boxplot(data=df_E1_group, x='time', y='corr', order=['random', 'hit', 'reward'], color='gray', ax=ax3)
    sns.stripplot(data=df_E1_group, x='time', y='corr', hue='mice', order=['random', 'hit', 'reward'],
                  palette=color_mapping, jitter=True, ax=ax3)
    a = df_E1_group[df_E1_group.time == 'hit']['corr']
    b = df_E1_group[df_E1_group.time == 'random']['corr']
    c = df_E1_group[df_E1_group.time == 'reward']['corr']
    ut_plots.get_pvalues(a, b, ax3, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)
    ut_plots.get_pvalues(a, c, ax3, pos=1.5, height=b[~np.isnan(b)].max(), ind=False)
    ax3.set_xlabel('E1')
    # ax3.set_ylim([0.01, 0.055])

    df_E2 = df_melted[df_melted.type == 'E2'].drop(columns=['type'])
    df_E2_group = df_E2.groupby(['mice', 'session_path', 'time']).mean().reset_index()
    df_E2_group = (df_E2_group.drop(['session_path', 'trial'], axis=1).groupby(['mice', 'time']).
                mean().reset_index())

    fig4, ax4 = ut_plots.open_plot()
    sns.boxplot(data=df_E2_group, x='time', y='corr', order=['random', 'hit', 'reward'], color='gray', ax=ax4)
    sns.stripplot(data=df_E2_group, x='time', y='corr', order=['random', 'hit', 'reward'], hue='mice',
                  palette=color_mapping, jitter=True, ax=ax4)
    a = df_E2_group[df_E1_group.time == 'hit']['corr']
    b = df_E2_group[df_E1_group.time == 'random']['corr']
    c = df_E2_group[df_E1_group.time == 'reward']['corr']
    ut_plots.get_pvalues(a, b, ax4, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)
    ut_plots.get_pvalues(a, c, ax4, pos=1.5, height=b[~np.isnan(b)].max(), ind=False)
    ax4.set_xlabel('E2')
    # ax4.set_ylim([0.01, 0.055])

    df_cursor = df_melted[df_melted.type == 'cursor'].drop(columns=['type'])
    df_cursor_group = df_cursor.groupby(['mice', 'session_path', 'time']).mean().reset_index()
    df_cursor_group = (df_cursor.drop(['session_path', 'trial'], axis=1).groupby(['mice', 'time']).
                mean().reset_index())

    fig5, ax5 = ut_plots.open_plot()
    sns.boxplot(data=df_cursor_group, x='time', y='corr', order=['random', 'hit', 'reward'], color='gray', ax=ax5)
    sns.stripplot(data=df_cursor_group, x='time', y='corr', hue='mice', order=['random', 'hit', 'reward'],
                  palette=color_mapping, jitter=True, ax=ax5)
    a = df_cursor_group[df_cursor_group.time == 'hit']['corr']
    b = df_cursor_group[df_cursor_group.time == 'random']['corr']
    c = df_cursor_group[df_cursor_group.time == 'reward']['corr']
    ut_plots.get_pvalues(a, b, ax5, pos=0.5, height=b[~np.isnan(b)].max(), ind=False)
    ut_plots.get_pvalues(a, c, ax5, pos=1.5, height=b[~np.isnan(b)].max(), ind=False)
    ax5.set_xlabel('cursor')
    # ax5.set_ylim([0.01, 0.055])


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



# Function to obtain the activity of each neuron
def obtain_Roi(Im, strcMask, units=None):
    """
    Function to obtain the activity of each neuron, given a spatial filter
    units -> indices of the neurons in the neuronMask that we want
    Im -> Image (2D array)
    strcMask -> structure with the matrix for spatial filters with px*py*unit
    units -> list of neuron indices to return, if None, use all units
    """
    if units is None:
        units = range(len(strcMask['neuronMask'][0][0][0]))

    unitVals = np.zeros(len(units))

    for idx, u in enumerate(units):
        posmaxx = int(strcMask['maxx'][0][0][0][u]) - 1
        posminx = int(strcMask['minx'][0][0][0][u]) - 1
        posmaxy = int(strcMask['maxy'][0][0][0][u]) - 1
        posminy = int(strcMask['miny'][0][0][0][u]) - 1

        Imd = Im[posminy:posmaxy + 1, posminx:posmaxx + 1].astype(float)
        neuron_mask = strcMask['neuronMask'][0][0][0][u]

        unitVals[idx] = np.nansum(Imd * (neuron_mask / (u+1) / np.nansum(neuron_mask)))

    return unitVals

def prepare_video(folder_suite2p: Path, folder_raw: Path, folder_plots):
    """ function to create a video with online data"""

    import scipy.io
    import warnings
    import cv2
    from PIL import Image
    from tifffile import TiffFile
    warnings.simplefilter("ignore", category=UserWarning)

    # video used m16_221116_D05
    # Load the roi_data and strcMask from .mat files
    folder_raw = Path(folder_raw)
    folder_suite2p = Path(folder_suite2p)
    im_folder = folder_raw / 'im/BMI_stim/BMI_stim_221116T111406-246/'

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

    # Size parameters
    subsection_size = 32
    half_size = subsection_size // 2
    frame_rate = 30  # 30Hz frame rate
    f0_win_im = 30  # Window size for baseline calculation
    f0_win = 10  # Window size for baseline calculation
    calibration_limit = 27000  # Upper frame limit for the calibration period
    total_frames = 81000
    sec_to_plot = 5
    frames_to_plot = sec_to_plot * frame_rate
    number_neurons = len(strcMask['neuronMask'][0][0][0])
    frame_width = 1920
    frame_height = 1080

    # Initialize Fbuffer and baseval
    Fbuffer = np.zeros((number_neurons, f0_win))
    baseval = np.zeros(number_neurons)
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
            unitVals = obtain_Roi(image_buffer, strcMask)

            # Store in Fbuffer
            Fbuffer[:, :-1] = Fbuffer[:, 1:]  # Shift buffer
            Fbuffer[:, -1] = unitVals

    # initialize baseval and image_buffer
    baseval = np.nanmean(Fbuffer, 1)
    smooth_buffer = np.zeros((4, f0_win_im, subsection_size, subsection_size))


    # Create the figure with 3 subplots in each of the 4 rows
    fig = plt.figure(figsize=(9, 9))

    # Set up GridSpec with specific width ratios
    gs = GridSpec(4, 3, width_ratios=[0.5, 0.5, 0.75], wspace=0.05,
                  hspace=0.2)  # Adjust 'hspace' to reduce vertical distance
    axes = []
    for i in range(4):
        ax1 = fig.add_subplot(gs[i, 0])  # First column for images
        ax2 = fig.add_subplot(gs[i, 1])  # Second column for images
        ax3 = fig.add_subplot(gs[i, 2])  # Third column for line plots
        axes.append([ax1, ax2, ax3])

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
            axes[idx][0].set_title(f'Neuron {idx + 1}')
            axes[idx][0].axis('off')

        # Initialize the fluorescence values for each (xctr, yctr) pair for the current stim_index
        fluorescence_values = np.zeros([frames_to_plot, number_neurons])

        # Loop through the frames_to_plot frames before each stim_idx
        for frame_count, frame_idx in enumerate(range(stim_idx - frames_to_plot - f0_win_im, stim_idx)):
            if si > 0:
                if (frame_idx - frames_to_plot - f0_win_im) < stim_index[si-1]:
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

            unitVals = obtain_Roi(changing_image, strcMask)

            # buffer baseline for dff
            Fbuffer[:, :-1] = Fbuffer[:, 1:]
            Fbuffer[:, -1] = unitVals

            if frame_count < frame_to_buffer:
                baseval = (baseval * (frame_count + 1) + unitVals) / (frame_count + 2)
            else:
                baseval = (baseval * (f0_win - 1) + unitVals) / f0_win
                Fsmooth = np.nanmean(Fbuffer, axis=1)
                # Calculate the average fluorescence for the relevant neurons
                fluorescence_values[frame_count - f0_win_im, :] = (Fsmooth - baseval) / baseval * 100

            # Iterate through each (xctr, yctr) pair
            for idx, (xctr, yctr) in enumerate(zip(strcMask['xctr'][0][0][0], strcMask['yctr'][0][0][0])):

                # Extract the subsection centered at (xctr, yctr)
                subsection = changing_image[yctr - half_size:yctr + half_size, xctr - half_size:xctr + half_size]

                # buffer the images
                smooth_buffer[idx, :-1] = smooth_buffer[idx, 1:]  # Shift buffer
                smooth_buffer[idx, -1] = subsection  # Add current frame

                if frame_count > frame_to_buffer:

                    smooth_image = np.mean(smooth_buffer[idx], axis=0)

                    # Changing image in the second column
                    axes[idx][1].clear()
                    axes[idx][1].imshow(smooth_image, cmap='gray', interpolation='hanning')
                    axes[idx][1].set_title(f'Neuron {idx}')
                    axes[idx][1].axis('off')

                    time_values = np.linspace(-5, 0, frames_to_plot)
                    # Average fluorescence in the third column
                    axes[idx][2].clear()
                    axes[idx][2].plot(time_values[:len(fluorescence_values[:, idx])],
                                      fluorescence_values[:, idx], color='darkgray', linewidth=2) #darkorange
                    axes[idx][2].yaxis.set_label_position('right');
                    axes[idx][2].yaxis.tick_right()
                    axes[idx][2].set_xlabel('Time (s)')
                    axes[idx][2].set_ylabel('dFF')
                    axes[idx][2].set_xlim([-5, 0])
                    axes[idx][2].set_ylim([-20, 55])

            if frame_count > f0_win_im:
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
