# constants to be used on analysis (for offline processing)

import posixpath
from dataclasses import dataclass


def learning_directory(aux_dir: posixpath) -> str:
    return posixpath.join(aux_dir, "learning")


def population_directory(aux_dir: posixpath) -> str:
    return posixpath.join(aux_dir, "population")


def name_parquet(aux_name: str) -> str:
    return f'{aux_name}.parquet'


def analysis_configuration_file(session_name: str) -> str:
    return f'{session_name}_analysis_configuration.pkl'


def info_units_directory(aux_dir: posixpath) -> str:
    return posixpath.join(aux_dir, "info_units")


@dataclass
class AnalysisConstants:
    """  Class containing various constants for analysis, such as str for filenames """
    var_sweep = 'sweep'
    var_tuned = 'tuned'
    var_error = 'error'
    var_count = 'count'
    var_bins = 'bins'
    var_slope = 'slope'
    # from Prairie
    framerate = 29.752  # framerate of acquisition of images
    calibration_frames = 27000  # number of frames during calibration
    dff_win = 10  # number of frames to smooth dff
    len_calibration = 15  # length calibration in minutes
    experiment_types = ['D1act',
                        'CONTROL', 'CONTROL_LIGHT', 'CONTROL_AGO',
                        'RANDOM', 'NO_AUDIO', 'DELAY']
    behav_type = ['Initial_behavior', 'Behavior_before']
    # stim removal
    height_stim_artifact = 3  # height of the stim artifact in std
    # preprocess
    fast_disk = 'C:/Users/Nuria/Documents/DATA'