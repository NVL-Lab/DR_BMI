__author__ = 'Nuria'

import traja
import pandas as pd

from utils.analysis_command import AnalysisConfiguration


def extract_XY_data(trigger_file: str, XY_file: str) -> traja:
    """ Function to synchronize the XY data to the trigger file init """
    trigger_data = pd.read_csv(trigger_file, sep=";", encoding="utf_16_le")
    XY_raw_data = pd.read_csv(XY_file, sep=";")
    XY_raw_data = XY_raw_data.rename(columns={"xPos": "x", "yPos": "y"})
    synch_time = trigger_data.iloc[np.where(trigger_data.unitLabel == "Input_trigger")[0][-1]].DateTime
    XY_data = traja.TrajaDataFrame(XY_raw_data[XY_raw_data.DateTime > synch_time].reset_index(drop=True))
    XY_smooth = traja.smooth_sg(XY_data, w=AnalysisConfiguration.speed_smooth_factor)
    return XY_smooth


def obtain_movement_parameters(XY_smooth: traja) -> pd.Series:
    """ Function to obtain movement features out of a traja pd.Dataframe """
    movement = pd.Series()
    movement['timedelta'] = (XY_smooth.iloc[-1].DateTime - XY_smooth.iloc[0].DateTime) * 24 * 60 * 60  # in seconds
    movement['total_distance'] = traja.trajectory.length(XY_smooth)
    movement['distance/min'] = movement['total_distance'] / (movement['timedelta'] / 60)
    XY_derivatives = traja.trajectory.get_derivatives(XY_smooth)
    movement['speed_max_'] = XY_derivatives.speed.max()
    movement['speed_mean'] = XY_derivatives.speed.mean()
    movement['speed_std'] = XY_derivatives.speed.std()
    movement['acceleration_max'] = XY_derivatives.acceleration.max()
    movement['acceleration_mean'] = XY_derivatives.acceleration.mean()
    movement['acceleration_std'] = XY_derivatives.acceleration.std()
    run_data = traja.trajectory.speed_intervals(XY_smooth, faster_than=AnalysisConfiguration.run_speed_min)
    walk_data = traja.trajectory.speed_intervals(XY_smooth, faster_than=AnalysisConfiguration.walk_speed_min)
    movement['boat_run'] = len(run_data)
    movement['boat_run/min'] = len(run_data) / (movement['timedelta'] / 60)
    movement['boat_run_speed'] = run_data.speed.mean()
    movement['boat_walk'] = len(walk_data)
    movement['boat_walk/min'] = len(walk_data) / (movement['timedelta'] / 60)
    movement['boat_walk_spped'] = walk_data.speed.mean()
    return movement