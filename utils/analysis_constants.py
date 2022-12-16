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
    experiment_types = ['BMI_STIM_AGO', 'BMI_CONTROL_RANDOM', 'BMI_CONTROL_LIGHT', 'BMI_CONTROL_AGO']
    behav_type = ['Initial_behavior', 'Behavior_before']