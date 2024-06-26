__author__ = 'Nuria'

import copy

import imageio
import pandas as pd
import seaborn as sns
import numpy as np

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

