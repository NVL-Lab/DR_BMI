__author__ = 'Nuria'

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from pathlib import Path
from matplotlib import interactive
from scipy.stats import gmean

import analysis.learning_population as lp
import utils.utils_analysis as ut
import utils.util_plots as ut_plots
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


