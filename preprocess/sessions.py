
import collections
import pandas as pd
from typing import Optional
from pathlib import Path


__author__ = 'Nuria'


_BMI_STIM_AGO = {
    'D13': [
        'ago13/221113/D02',
        'ago13/221114/D03',
        'ago13/221115/D04-2',
        'ago13/221116/D05',
    ],
    'D15': [
        'ago15/221113/D02',
        'ago15/221114/D03',
        'ago15/221115/D04-2',
        'ago15/221116/D05',
        'ago15/221117/D06-2',
        'ago15/221119/D08',
        'ago15/221119/D08-2',
    ],
    'D16': [
        'ago16/221113/D02',
        'ago16/221114/D03',
        'ago16/221116/D05',
        'ago16/221118/D07',
        'ago16/221118/D07-2',
        'ago16/221119/D08',
        'ago16/221119/D08-2',
    ],
    'D18': [
        'ago18/221113/D02',
        'ago18/221114/D03',
        'ago18/221116/D05',
        'ago18/221117/D06-2',
        'ago18/221118/D07',
        'ago18/221118/D07-2',
        'ago18/221118/D07-3',
    ],
}

_BMI_RANDOM = {
    'D13': [
        'ago13/221115/D01'],
    'D15': [
        'ago15/221115/D04',
        'ago15/221116/D05-2',
        'ago15/221117/D06',
        'ago15/221118/D07',
        'ago15/221118/D07-3',
    ],
    'D16': [
        'ago16/221115/D04',
        'ago16/221116/D05-2',
        'ago16/221117/D06',
        'ago16/221117/D06-2',
        'ago16/221119/D08-3',
    ],
    'D18': [
        'ago18/221115/D04',
        'ago18/221116/D05-2',
        'ago18/221117/D06',
        'ago18/221119/D08',
        'ago18/221119/D08-2',
    ],
}

_BMI_STIM = {
    'D13': [
        'ago13/221112/D01'],
    'D15': [
        'ago15/221112/D01'],
    'D16': [
        'ago16/221112/D01'],
    'D18': [
        'ago18/221112/D01'],
}

_BMI_AGO = {
    'D13': [
        'ago13/221117/D06'],
    'D15': [
        'ago15/221118/D07-2'],
    'D16': [
        'ago16/221115/D04-2'],
    'D18': [
        'ago18/221116/D05-3'],
}

_BEHAVIOR = {
    'D13': [
        'ago13/221113/D02'],
    'D15': [
        'ago15/221113/D02'],
    'D16': [
        'ago16/221113/D02'],
    'D18': [
        'ago18/221113/D02'],
}


def get_sessions_df(folder_experiments: Path, experiment_type: Optional[str] = None) -> pd.DataFrame:
    """ Function to retrieve the name of the sessions that will be used depending on the experiment type
    and the files that are useful for that experiment, baselines, bmis, behaviors, etc"""
    if experiment_type == 'BMI_STIM_AGO':
        dict_items = _BMI_STIM_AGO.items()
    elif experiment_type == 'BMI_RANDOM':
        dict_items = _BMI_RANDOM.items()
    elif experiment_type == 'BMI_STIM':
        dict_items = _BMI_STIM.items()
    elif experiment_type == 'BMI_AGO':
        dict_items = _BMI_AGO.items()
    elif experiment_type == 'BEHAVIOR':
        dict_items = _BEHAVIOR.items()
    else:
        raise ValueError(
            f'Could not find any controls for {experiment_type} try BMI_STIM_AGO, BMI_RANDOM, BMI_STIM, BMI_AGO or BEHAVIOR')
    ret = collections.defaultdict(list)
    for mice_name, sessions_per_type in dict_items:
        for day_index, session_path in enumerate(sessions_per_type):
            # TODO something to split up the session
            ret['mice_name'].append(mice_name)
            ret['session_path'].append(session_path)
            ret['day_index'].append(day_index)

            dir_files = Path(folder_experiments) / session_path
            list_files = []  # TODO find the way to ls the contents of the folder
            for file_name in list_files:
                if file_name[:2] == 'im':
                    dir_im = Path(folder_experiments) / session_path / 'im'
                    list_im_directories = []  # TODO find the way to ls the contents of the folder
                    for file_name_im_dir in list_im_directories:
                        dir_im2 = dir_im / file_name_im_dir
                        list_im_files = [] # TODO find the way to ls the contents of the folder without . and ..
                        for file_name_im_file in list_im_files:
                            if file_name[:8] == 'baseline':
                                ret['Baseline_im'].append(file_name_im_file)
                                # TODO check that the + here works properly
                                ret['Voltage_Baseline'].append(file_name_im_file + '_Cycle00001_VoltageRecording_001.csv')
                            else:
                                ret['Voltage_rec'].append(file_name_im_file + '_Cycle00001_VoltageRecording_001.csv')
                                if experiment_type == 'BEHAVIOR':
                                    if True: # TODO find how to know if it already has checked behavior
                                        ret['Behavior_pre'].append(file_name_im_file)
                                        ret['Behavior_post'].append(file_name_im_file)
                                else:
                                    ret['Experiment'].append(file_name_im_file)

                if experiment_type != 'BEHAVIOR':
                    if file_name[:10] == 'BaselineOn':
                        ret['Baseline_online'].append(file_name)
                    elif file_name[:10] == 'BMI_online':
                        ret['BMI_online'].append(file_name)
                    elif file_name[:10] == 'BMI_target':
                        ret['BMI_target'].append(file_name)
                    elif file_name[:8] == 'roi_data':
                        ret['roi_data'].append(file_name)
                    elif file_name[:8] == 'strcMask':
                        ret['mask_data'].append(file_name)
                    elif file_name[:10] == 'target_cal':
                        ret['target_calibration'].append(file_name)
    return pd.DataFrame(ret)
