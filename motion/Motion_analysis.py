__author__ = 'Nuria'

import traja
import pandas as pd

from utils.analysis_command import AnalysisConfiguration


def extract_XY_data(trigger_file: str, XY_file: str) -> pd.Series:
    """ Function to synchronize the XY data to the trigger file init """
    movement = pd.Series()
    trigger_data = pd.read_csv(trigger_file, sep=";", encoding="utf_16_le")
    XY_raw_data = pd.read_csv(XY_file, sep=";")
    XY_raw_data = XY_raw_data.rename(columns={"xPos": "x", "yPos": "y"})
    synch_time = trigger_data.iloc[np.where(trigger_data.unitLabel == "Input_trigger")[0][-1]].DateTime
    XY_data = traja.TrajaDataFrame(XY_raw_data[XY_raw_data.DateTime > synch_time].reset_index(drop=True))
    XY_smooth = traja.smooth_sg(XY_data, w=AnalysisConfiguration.speed_smooth_factor)
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
    movement['boat_run'] = len(traja.trajectory.speed_intervals(XY_smooth,
                                                                faster_than=AnalysisConfiguration.run_speed_min))
    movement['boat_walk'] = len(traja.trajectory.speed_intervals(XY_smooth,
                                                                 faster_than=AnalysisConfiguration.walk_speed_min))
    return movement
