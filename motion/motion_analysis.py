__author__ = 'Nuria'

from typing import Optional, Tuple

import traja
import pandas as pd
import numpy as np
from traja import TrajaDataFrame

from utils.analysis_command import AnalysisConfiguration


def extract_XY_data(trigger_file: str, XY_file: str) -> traja:
    """ Function to synchronize the XY data to the trigger file init """
    trigger_data = pd.read_csv(trigger_file, sep=";", encoding="utf_16_le")
    XY_raw_data = pd.read_csv(XY_file, sep=";")
    if len(XY_raw_data) > 0:
        XY_raw_data = XY_raw_data.rename(columns={"xPos": "x", "yPos": "y"})
        if len(trigger_data) > 0 :
            input_times = np.where(trigger_data.SystemMsg == "Start")[0]
            if len(input_times) > 0:
                synch_event = input_times[-1]
            else:
                synch_event = 0

            synch_time = trigger_data.iloc[synch_event].DateTime
        else:
            synch_time = 0
        XY_data = traja.TrajaDataFrame(XY_raw_data[XY_raw_data.DateTime > synch_time].reset_index(drop=True))
        XY_smooth = traja.smooth_sg(XY_data, w=AnalysisConfiguration.speed_smooth_factor)
        XY_smooth.loc[slice(None), "DateTime"] = ((XY_smooth.loc[slice(None), "DateTime"] -
                                                   XY_smooth.loc[0, "DateTime"]) * 24 * 60 * 60)
    else:
        XY_smooth = None
    return XY_smooth


def obtain_movement_parameters(XY_smooth: Optional[TrajaDataFrame] = None,
                               speed_min=AnalysisConfiguration.run_speed_min) -> Tuple[pd.Series, pd.Series]:
    """ Function to obtain movement features out of a traja pd.Dataframe """
    if XY_smooth is not None:
        movement = pd.Series()
        movement['timedelta'] = XY_smooth.iloc[-1].DateTime
        movement['total_distance'] = traja.trajectory.length(XY_smooth)
        movement['distance_per_min'] = movement['total_distance'] / (movement['timedelta'] / 60)
        XY_derivatives = traja.trajectory.get_derivatives(XY_smooth)
        movement['speed_max'] = XY_derivatives.speed.max()
        movement['speed_mean'] = XY_derivatives.speed.mean()
        movement['speed_std'] = XY_derivatives.speed.std()
        movement['acceleration_max'] = XY_derivatives.acceleration.max()
        movement['acceleration_mean'] = XY_derivatives.acceleration.mean()
        movement['acceleration_std'] = XY_derivatives.acceleration.std()
        run_data = traja.trajectory.speed_intervals(XY_smooth, faster_than=speed_min)
        stop_data = traja.trajectory.speed_intervals(XY_smooth, slower_than=speed_min)
        movement['initiations_per_min'] = len(run_data) / (movement['timedelta'] / 60)
        movement['bout_duration'] = run_data.duration.mean()
        movement['bout_speed_mean'] = XY_derivatives[XY_derivatives.speed > AnalysisConfiguration.run_speed_min].speed.mean()
        movement['bout_speed_max'] = XY_derivatives[XY_derivatives.speed > AnalysisConfiguration.run_speed_min].speed.max()
        movement['stops_min'] = len(stop_data) / (movement['timedelta'] / 60)
        movement['time_moving'] = run_data.duration.sum() / movement['timedelta'] * 100
        speed = XY_derivatives.speed
    else:
        movement = None
        speed = np.empty(0)

    return movement, speed
