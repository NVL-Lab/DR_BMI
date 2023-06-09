__author__ = 'Nuria'

import pandas as pd
import seaborn as sns
import numpy as np

from pathlib import Path
from matplotlib import interactive

from utils import util_plots as ut_plots
interactive(True)


def plot_bout_init():
    df_motion = pd.read_parquet(Path("C:/Users/Nuria/Documents/DATA/D1exp/df_data") / "df_motion.parquet")
    mice = df_motion.mice.unique()
    copper_palette = sns.color_palette("copper", n_colors=len(mice))
    feature = "initiations_per_min"
    df_group = df_motion.groupby(["mice", "Laser"]).mean().reset_index()
    df_group = df_group[df_group.Laser.isin(['ON', 'OFF'])]
    fig1, ax1 = ut_plots.open_plot()
    sns.lineplot(data=df_group, x='Laser', y=feature, hue='mice', palette=copper_palette, ax=ax1)
    sns.stripplot(data=df_group, x="Laser", y=feature, hue='mice', palette=copper_palette, s=10,
                  marker="D", jitter=False, ax=ax1)
    ax1.set_ylim([0.2, 2.5])
    a = df_group[df_group.Laser == 'ON'][feature]
    b = df_group[df_group.Laser == 'OFF'][feature]
    ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)

    df_group = df_motion.groupby(["mice", "Laser"]).mean().reset_index()
    df_group = df_group[df_group.Laser.isin(['BMI', 'OFF'])]
    fig1, ax1 = ut_plots.open_plot()
    sns.lineplot(data=df_group, x='Laser', y=feature, hue='mice', palette=copper_palette, ax=ax1)
    sns.stripplot(data=df_group, x="Laser", y=feature, hue='mice', palette=copper_palette, s=10,
                  marker="D", jitter=False, ax=ax1)
    ax1.set_ylim([0.2, 2.5])
    a = df_group[df_group.Laser == 'BMI'][feature]
    b = df_group[df_group.Laser == 'OFF'][feature]
    ut_plots.get_pvalues(a, b, ax1, pos=0.5, height=a[~np.isnan(a)].max(), ind=True)




