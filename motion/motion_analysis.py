__author__ = 'Nuria'

from typing import Optional

import traja
import pandas as pd
import numpy as np

from utils.analysis_command import AnalysisConfiguration


def extract_XY_data(trigger_file: str, XY_file: str) -> traja:
    """ Function to synchronize the XY data to the trigger file init """
    trigger_data = pd.read_csv(trigger_file, sep=";", encoding="utf_16_le")
    XY_raw_data = pd.read_csv(XY_file, sep=";")
    if len(XY_raw_data) > 0:
        XY_raw_data = XY_raw_data.rename(columns={"xPos": "x", "yPos": "y"})
        if len(trigger_data) > 0 :
            input_times = np.where(trigger_data.unitLabel == "Input_trigger")[0]
            if len(input_times) > 0:
                synch_event = input_times[-1]
            else:
                synch_event = 0

            synch_time = trigger_data.iloc[synch_event].DateTime
        else:
            synch_time = 0
        XY_data = traja.TrajaDataFrame(XY_raw_data[XY_raw_data.DateTime > synch_time].reset_index(drop=True))
        XY_smooth = traja.smooth_sg(XY_data, w=AnalysisConfiguration.speed_smooth_factor)
    else:
        XY_smooth = None
    return XY_smooth


def obtain_movement_parameters(XY_smooth: Optional[pd.DataFrame] = None) -> pd.Series:
    """ Function to obtain movement features out of a traja pd.Dataframe """
    if XY_smooth is not None:
        movement = pd.Series()
        movement['timedelta'] = (XY_smooth.iloc[-1].DateTime - XY_smooth.iloc[0].DateTime) * 24 * 60 * 60  # in seconds
        movement['total_distance'] = traja.trajectory.length(XY_smooth)
        movement['distance_per_min'] = movement['total_distance'] / (movement['timedelta'] / 60)
        XY_derivatives = traja.trajectory.get_derivatives(XY_smooth)
        movement['speed_max'] = XY_derivatives.speed.max()
        movement['speed_mean'] = XY_derivatives.speed.mean()
        movement['speed_std'] = XY_derivatives.speed.std()
        movement['acceleration_max'] = XY_derivatives.acceleration.max()
        movement['acceleration_mean'] = XY_derivatives.acceleration.mean()
        movement['acceleration_std'] = XY_derivatives.acceleration.std()
        run_data = traja.trajectory.speed_intervals(XY_smooth, faster_than=AnalysisConfiguration.run_speed_min)
        walk_data = traja.trajectory.speed_intervals(XY_smooth, faster_than=AnalysisConfiguration.walk_speed_min)
        movement['bout_run'] = len(run_data)
        movement['bout_run_per_min'] = len(run_data) / (movement['timedelta'] / 60)
        movement['bout_run_speed'] = XY_derivatives[XY_derivatives.speed > AnalysisConfiguration.run_speed_min].speed.mean()
        movement['bout_walk'] = len(walk_data)
        movement['bout_walk_per_min'] = len(walk_data) / (movement['timedelta'] / 60)
        movement['bout_walk_speed'] = XY_derivatives[XY_derivatives.speed > AnalysisConfiguration.walk_speed_min].speed.mean()
    else:
        movement = None
    return movement