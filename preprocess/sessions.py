"""
This Python script is responsible for organizing and accessing data that is distributed across three hard drives.
It manages the data to ensures seamless access regardless of its physical location.
Additionally, it includes a dictionary with the different experiments and their respective sessions.
And it creates a DataFrame with this information.
"""

import collections
import os
import pandas as pd
import numpy as np
from pathlib import Path

from utils.analysis_constants import AnalysisConstants

__author__ = 'Nuria'


_FOLDER_PATHS = {
    'FA': ['m13', 'm15', 'm16', 'm18', 'm25'],
    'FB': ['m21', 'm22', 'm26'],
    'FC': ['m23', 'm27', 'm28', 'm29']
}

_D1act = {
    'm13': [
        'm13/221113/D02',
        'm13/221114/D03',
        'm13/221116/D05',
    ],
    'm15': [
        'm15/221113/D02',
        'm15/221114/D03',
        'm15/221116/D05',
        'm15/221119/D08',
    ],
    'm16': [
        'm16/221113/D02',
        'm16/221114/D03',
        'm16/221116/D05',
        'm16/221118/D07',
        'm16/221119/D08',
    ],
    'm18': [
        'm18/221113/D02',
        'm18/221114/D03',
        'm18/221116/D05',
        'm18/221118/D07',
    ],
    'm21': [
        'm21/230414/D06',
        'm21/230415/D07',
        'm21/230416/D08',
        'm21/230417/D09',
        'm21/230418/D10'
    ],
    'm22': [
        'm22/230414/D03',
        'm22/230415/D04',
        'm22/230416/D05',
        'm22/230417/D06',
        'm22/230418/D07',
        'm22/230419/D08'
    ],
    'm23': [
        'm23/230419/D02',
        'm23/230420/D03',
        'm23/230421/D04',
        'm23/230422/D05'
    ],
    'm26': [
        'm26/230414/D06',
        'm26/230415/D07',
        'm26/230416/D08',
        'm26/230417/D09',
    ],
    'm27': [
        'm27/230414/D02',
        'm27/230415/D03'
    ],
    'm28': [
        'm28/230414/D06',
        'm28/230415/D07',
        'm28/230416/D08',
        'm28/230417/D09'
    ],
    'm29': [
        'm29/230419/D02',
        'm29/230420/D03',
        'm29/230421/D04',
    ]
}

_RANDOM = {
    'm15': [
        'm15/221115/D04',
        'm15/221117/D06',
        'm15/221118/D07',
    ],
    'm16': [
        'm16/221115/D04',
        'm16/221117/D06',
    ],
    'm18': [
        'm18/221115/D04',
        'm18/221117/D06',
        'm18/221119/D08',
    ],
    'm21': [
        'm21/230412/D04',
        'm21/230413/D05',
    ],
    'm22': [
        'm22/230413/D02',
    ],
    'm23': [
        'm23/230423/D06',
        'm23/230424/D07',
    ],
    'm26': [
        'm26/230412/D04',
        'm26/230413/D05',
    ],
    'm28': [
        'm28/230412/D04',
        'm28/230413/D05',
        'm28/230421/D13'
    ]
}

_CONTROL_LIGHT = {
    'm13': [
        'm13/221112/D01'
    ],
    'm15': [
        'm15/221112/D01'
    ],
    'm16': [
        'm16/221112/D01'
    ],
    'm18': [
        'm18/221112/D01'
    ],
    'm21': [
        'm21/230406/D01'
    ],
    'm23': [
        'm23/230418/D01'
    ],
    'm26': [
        'm26/230406/D01'
    ],
    'm27': [
        'm27/230406/D01'
    ],
    'm28': [
        'm28/230406/D01'
    ],
    'm29': [
        'm29/230418/D01'
    ]
}

_CONTROL_AGO = {
    'm13': [
        'm13/221117/D06'
    ],
    'm21': [
        'm21/230422/D14'
    ],
    'm22': [
        'm22/230422/D10'
    ],
    'm26': [
        'm26/230418/D10'
    ],
    'm28': [
        'm28/230422/D14'
    ],
    'm29': [
        'm29/230422/D05'
    ]
}

_NO_AUDIO = {
    'm21': [
        'm21/230407/D02',
        'm21/230408/D03'
    ],
    'm22': [
        'm22/230423/D11',
        'm22/230424/D12'
    ],
    'm26': [
        'm26/230407/D02',
        'm26/230408/D03',
    ],
    'm28': [
        'm28/230407/D02',
        'm28/230408/D03',
    ],
    'm29': [
        'm29/230424/D07'
    ]
}

_DELAY = {
    'm21': [
        'm21/230419/D11',
        'm21/230420/D12',
        'm21/230421/D13'
    ],
    'm22': [
        'm22/230420/D09',
    ],
    'm28': [
        'm28/230418/D10',
        'm28/230419/D11',
        'm28/230420/D12'
    ],
    'm29': [
        'm29/230423/D06'
    ]
}

_EXTINCTION = {
    'm21': [
        'm21/230418/D10'
    ],
    'm22': [
        'm22/230419/D08'
    ],
    'm23': [
        'm23/230422/D05'
    ],
    'm26': [
        'm26/230417/D09-2'
    ],
    'm28': [
        'm28/230417/D09'
    ]
}

_CONTROL = {
    'm25': [
        'm25/230425/D11',
    ],
    'm26': [
        'm26/230425/D11',
    ],
    'm27': [
        'm27/230424/D02',
    ],
    'm28': [
        'm28/230424/D14',
        'm28/230425/D15'
    ],
    'm29': [
        'm29/230425/D08'
    ]
}

_BEHAVIOR = {
    'm13': [
        'm13/221113/D02',
        'm13/221114/D03'
    ],
    'm15': [
        'm15/221113/D02',
        'm15/221114/D03'
    ],
    'm16': [
        'm16/221113/D02',
        'm16/221114/D03'
    ],
    'm18': [
        'm18/221113/D02',
        'm18/221114/D03'
    ],
    'm21': [
        'm21/230407/D02',
        'm21/230408/D03'
    ],
    'm22': [
        'm22/230413/D02',
        'm22/230419/D08'
    ],
    'm23': [
        'm23/230419/D02'
    ],
    'm26': [
        'm26/230407/D02',
        'm26/230408/D03'
    ],
    'm28': [
        'm28/230407/D02'
    ],
    'm29': [
        'm29/230419/D02'
    ]
}

_MOTOR_beh_before_BMI = {
    'm13': [
        'm13/221113/D02',
        'm13/221114/D03'
    ],
    'm15': [
        'm15/221113/D02',
        'm15/221114/D03'
    ],
    'm16': [
        'm16/221113/D02',
        'm16/221114/D03'
    ],
    'm18': [
        'm18/221113/D02',
        'm18/221114/D03'
    ],
    'm21': [
        'm21/230407/D02',
        'm21/230408/D03'
    ],
    'm22': [
        'm22/230413/D02',
        'm22/230419/D08'
    ],
    'm23': [
        'm23/230419/D02'
    ],
    'm26': [
        'm26/230407/D02',
        'm26/230408/D03'
    ],
    'm28': [
        'm28/230407/D02'
    ],
    'm29': [
        'm29/230419/D02'
    ]

}

_MOTOR_initial_behavior = {
    'm13': [
        'm13/221113/D02'
    ],
    'm15': [
        'm15/221113/D02'
    ],
    'm16': [
        'm16/221113/D02'
    ],
    'm18': [
        'm18/221113/D02'
    ]
}


def get_all_sessions() -> pd.DataFrame:
    """ function to get a df with all sessions"""
    df_d1act = pd.DataFrame(index=np.concatenate(list(_D1act.values())))
    df_d1act['experiment_type'] = 'D1act'
    df_c = pd.DataFrame(index=np.concatenate(list(_CONTROL.values())))
    df_c['experiment_type'] = 'CONTROL'
    df_c_light = pd.DataFrame(index=np.concatenate(list(_CONTROL_LIGHT.values())))
    df_c_light['experiment_type'] = 'CONTROL_LIGHT'
    df_c_ago = pd.DataFrame(index=np.concatenate(list(_CONTROL_AGO.values())))
    df_c_ago['experiment_type'] = 'CONTROL_AGO'
    df_random = pd.DataFrame(index=np.concatenate(list(_RANDOM.values())))
    df_random['experiment_type'] = 'RANDOM'
    df_no_audio = pd.DataFrame(index=np.concatenate(list(_NO_AUDIO.values())))
    df_no_audio['experiment_type'] = 'NO_AUDIO'
    df_delay = pd.DataFrame(index=np.concatenate(list(_DELAY.values())))
    df_delay['experiment_type'] = 'DELAY'
    df_extinction = pd.DataFrame(index=np.concatenate(list(_EXTINCTION.values())))
    df_extinction['experiment_type'] = 'EXTINCTION'
    df_behavior = pd.DataFrame(index=np.concatenate(list(_BEHAVIOR.values())))
    df_behavior['experiment_type'] = 'BEHAVIOR'
    list_experiments = [df_d1act, df_c, df_c_light, df_c_ago, df_random, df_no_audio, df_delay, df_extinction,
                        df_behavior]
    df_experiments = pd.concat(list_experiments)
    return df_experiments.sort_index().reset_index()


def get_sessions_df(folder_list: list, experiment_type: str) -> pd.DataFrame:
    """ Function to retrieve the name of the sessions that will be used depending on the experiment type
    and the files that are useful for that experiment, baselines, bmis, behaviors, etc"""
    df_experiments = get_all_sessions()
    if experiment_type == 'D1act':
        dict_items = _D1act.items()
    elif experiment_type == 'CONTROL':
        dict_items = _CONTROL.items()
    elif experiment_type == 'CONTROL_LIGHT':
        dict_items = _CONTROL_LIGHT.items()
    elif experiment_type == 'CONTROL_AGO':
        dict_items = _CONTROL_AGO.items()
    elif experiment_type == 'RANDOM':
        dict_items = _RANDOM.items()
    elif experiment_type == 'NO_AUDIO':
        dict_items = _NO_AUDIO.items()
    elif experiment_type == 'DELAY':
        dict_items = _DELAY.items()
    elif experiment_type == 'EXTINCTION':
        dict_items = _EXTINCTION.items()
    elif experiment_type == 'BEHAVIOR':
        dict_items = _BEHAVIOR.items()
    else:
        raise ValueError(
            f'Could not find any controls for {experiment_type} '
            f'try D1act, CONTROL, CONTROL_LIGHT, CONTROL_AGO, RANDOM, NO_AUDIO, DELAY, EXTINCTION or BEHAVIOR')
    ret = collections.defaultdict(list)
    for mice_name, sessions_per_type in dict_items:
        for day_index, session_path in enumerate(sessions_per_type):
            [mice_name, session_date, day_init] = session_path.split('/')
            ret['mice_name'].append(mice_name)
            ret['session_date'].append(session_date)
            ret['day_init'].append(day_init)
            location_session = np.where(df_experiments["index"] == session_path)[0][0]
            if day_init[-2:] == '-2':
                ret['session_day'].append('2nd')
                ret['previous_session'].append(df_experiments.iloc[location_session - 1].experiment_type)
            elif day_init[-2:] == '-3':
                ret['session_day'].append('3rd')
                ret['previous_session'].append(df_experiments.iloc[location_session - 1].experiment_type)
            elif day_init[-2:] == '-4':
                ret['session_day'].append('4th')
                ret['previous_session'].append(df_experiments.iloc[location_session - 1].experiment_type)
            else:
                ret['session_day'].append('1st')
                ret['previous_session'].append('None')
            ret['experiment_type'].append(experiment_type)
            ret['session_path'].append(session_path)
            ret['day_index'].append(day_index)

            folder_raw = Path(folder_list[find_folder_path(mice_name)]) / 'raw'
            dir_files = Path(folder_raw) / session_path
            for file_name in os.listdir(dir_files):
                if experiment_type.lower() not in ['behavior', 'extinction']:
                    if file_name[:2] == 'im':
                        dir_im = Path(folder_raw) / session_path / 'im'
                        for file_name_im_dir in os.listdir(dir_im):
                            if file_name_im_dir.lower() not in ['behavior', 'extinction']:
                                dir_im2 = dir_im / file_name_im_dir
                                for file_name_im_file in os.listdir(dir_im2):
                                    if file_name_im_file[:8] == 'baseline':
                                        ret['Baseline_im'].append(file_name_im_file)
                                        ret['Voltage_Baseline'].append(file_name_im_file + '_Cycle00001_VoltageRecording_001.csv')
                                    elif file_name_im_file[:8] in ['BMI_stim', 'RandomDR']:
                                        ret['Voltage_rec'].append(file_name_im_file + '_Cycle00001_VoltageRecording_001.csv')
                                        ret['Experiment_im'].append(file_name_im_file)
                                        ret['Experiment_dir'].append(file_name_im_dir)

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

                if file_name[:2] == 'mo':
                    dir_motor = Path(folder_raw) / session_path / 'motor'
                    for file_name_motor_file in os.listdir(dir_motor):
                        if file_name_motor_file[-7:-4] in ['ine', 'BMI']:
                            [_, trigger_XY, _, baseline_BMI] = file_name_motor_file.split('_')
                            if trigger_XY == 'XY':
                                if baseline_BMI == 'baseline.csv':
                                    ret['XY_baseline'].append(file_name_motor_file)
                                elif baseline_BMI == 'BMI.csv':
                                    ret['XY_BMI'].append(file_name_motor_file)
                            elif trigger_XY == 'Trigger':
                                if baseline_BMI == 'baseline.csv':
                                    ret['trigger_baseline'].append(file_name_motor_file)
                                elif baseline_BMI == 'BMI.csv':
                                    ret['trigger_BMI'].append(file_name_motor_file)

    return pd.DataFrame(ret)


def get_motor_data_behav(folder_list: list, experiment_type: str) -> pd.DataFrame:
    """ Function to retrieve the name of the sessions that will be used depending on the experiment type
    and the files that are useful for that experiment, baselines, bmis, behaviors, etc"""
    if experiment_type == 'Initial_behavior':
        dict_items = _MOTOR_initial_behavior.items()
        ending_str = 'ior'
    elif experiment_type == 'Behavior_before':
        dict_items = _MOTOR_beh_before_BMI.items()
        ending_str = 'ore'
    else:
        raise ValueError(
            f'Could not find any controls for {experiment_type} try Initial_behavior, Behavior_before')
    ret = collections.defaultdict(list)
    for mice_name, sessions_per_type in dict_items:
        for day_index, session_path in enumerate(sessions_per_type):
            [mice_name, session_date, day_init] = session_path.split('/')
            ret['mice_name'].append(mice_name)
            ret['session_date'].append(session_date)
            ret['day_init'].append(day_init)
            ret['experiment_type'].append(experiment_type)
            ret['session_path'].append(session_path)
            ret['day_index'].append(day_index)

            folder_raw = Path(folder_list[find_folder_path(mice_name)])/ 'raw'
            dir_files = Path(folder_raw) / session_path
            for file_name in os.listdir(dir_files):
                if file_name[:2] == 'mo':
                    dir_motor = Path(folder_raw) / session_path / 'motor'
                    for file_name_motor_file in os.listdir(dir_motor):
                        if file_name_motor_file[-7:-4] == ending_str:
                            [_, trigger_XY, _, _, _] = file_name_motor_file.split('_')
                            if trigger_XY == 'XY':
                                ret['XY'].append(file_name_motor_file)
                            elif trigger_XY == 'Trigger':
                                ret['trigger'].append(file_name_motor_file)

    return pd.DataFrame(ret)


def get_neural_data_behav(folder_list: list) -> pd.DataFrame:
    """ Function to retrieve the name of the sessions that will be used depending on the experiment type
    and the files that are useful for that experiment, baselines, bmis, behaviors, etc"""
    dict_items = _BEHAVIOR.items()
    ret = collections.defaultdict(list)
    for mice, sessions_per_type in dict_items:
        for day_index, session_path in enumerate(sessions_per_type):
            [mice, session_date, day_init] = session_path.split('/')
            ret['mice_name'].append(mice)
            ret['session_date'].append(session_date)
            ret['day_init'].append(day_init)
            ret['session_path'].append(session_path)

            folder_raw = Path(folder_list[find_folder_path(mice)]) / 'raw'
            dir_files = Path(folder_raw) / session_path
            for file_name in os.listdir(dir_files):
                if file_name[:2] == 'im':
                    dir_im = Path(folder_raw) / session_path / 'im'
                    for file_name_im_dir in os.listdir(dir_im):
                        if file_name_im_dir.lower() in ['behavior', 'baseline']:
                            dir_im2 = dir_im / file_name_im_dir
                            for file_name_im_file in os.listdir(dir_im2):
                                if file_name_im_file[:8] == 'baseline':
                                    ret['Baseline_im'].append(file_name_im_file)
                                    ret['Voltage_Baseline'].append(file_name_im_file + '_Cycle00001_VoltageRecording_001.csv')
                                elif file_name_im_file[:8] == 'behavior':
                                    ret['Voltage_Behavior'].append(file_name_im_file + '_Cycle00001_VoltageRecording_001.csv')
                                    ret['Behavior_im'].append(file_name_im_file)

    return pd.DataFrame(ret)


def get_extinction(folder_list: list) -> pd.DataFrame:
    """ Function to retrieve the name of the sessions that will be used depending on the experiment type
    and the files that are useful for that experiment, baselines, bmis, behaviors, etc"""
    dict_items = _EXTINCTION.items()
    ret = collections.defaultdict(list)
    for mice, sessions_per_type in dict_items:
        for day_index, session_path in enumerate(sessions_per_type):
            [mice, session_date, day_init] = session_path.split('/')
            ret['mice'].append(mice)
            ret['session_date'].append(session_date)
            ret['day_init'].append(day_init)
            ret['session_path'].append(session_path)
            flag_extinction = False
            flag_extinction_2 = False
            folder_raw = Path(folder_list[find_folder_path(mice)]) / 'raw'
            dir_files = Path(folder_raw) / session_path
            for file_name in os.listdir(dir_files):
                if file_name[:10] == 'BaselineOn':
                    ret['Baseline_online'].append(file_name)
                elif file_name[:10] == 'BMI_online':
                    ret['BMI_online'].append(file_name)
                elif file_name.lower()[:10] == 'extinction':
                    if flag_extinction:
                        ret['extinction_2'].append(file_name)
                        flag_extinction_2 = True
                    else:
                        ret['extinction'].append(file_name)
                        flag_extinction = True
            if not flag_extinction:
                ret['extinction'].append('None')
            if not flag_extinction_2:
                ret['extinction_2'].append('None')

    return pd.DataFrame(ret)


def get_simulations_df(folder_list: list, experiment_type: str) -> pd.DataFrame:
    """ Function to retrieve the name of the simulations that will be used depending on the experiment type
    and the files that are useful for that experiment, baselines, bmis, behaviors, etc"""
    df_experiments = get_all_sessions()
    if experiment_type == 'D1act':
        dict_items = _D1act.items()
    elif experiment_type == 'CONTROL':
        dict_items = _CONTROL.items()
    elif experiment_type == 'CONTROL_LIGHT':
        dict_items = _CONTROL_LIGHT.items()
    elif experiment_type == 'CONTROL_AGO':
        dict_items = _CONTROL_AGO.items()
    elif experiment_type == 'RANDOM':
        dict_items = _RANDOM.items()
    elif experiment_type == 'NO_AUDIO':
        dict_items = _NO_AUDIO.items()
    elif experiment_type == 'DELAY':
        dict_items = _DELAY.items()
    else:
        raise ValueError(
            f'Could not find any controls for {experiment_type} '
            f'try D1act, CONTROL, CONTROL_LIGHT, CONTROL_AGO, RANDOM, NO_AUDIO, DELAY')
    ret = collections.defaultdict(list)
    for mice_name, sessions_per_type in dict_items:
        for day_index, session_path in enumerate(sessions_per_type):
            [mice_name, session_date, day_init] = session_path.split('/')
            ret['mice_name'].append(mice_name)
            ret['session_date'].append(session_date)
            ret['day_init'].append(day_init)
            location_session = np.where(df_experiments["index"] == session_path)[0][0]
            if day_init[-2:] == '-2':
                ret['session_day'].append('2nd')
                ret['previous_session'].append(df_experiments.iloc[location_session - 1].experiment_type)
            elif day_init[-2:] == '-3':
                ret['session_day'].append('3rd')
                ret['previous_session'].append(df_experiments.iloc[location_session - 1].experiment_type)
            elif day_init[-2:] == '-4':
                ret['session_day'].append('4th')
                ret['previous_session'].append(df_experiments.iloc[location_session - 1].experiment_type)
            else:
                ret['session_day'].append('1st')
                ret['previous_session'].append('None')
            ret['experiment_type'].append(experiment_type)
            ret['session_path'].append(session_path)
            ret['day_index'].append(day_index)

            folder_process = Path(folder_list[find_folder_path(mice_name)]) / 'process'
            dir_files = Path(folder_process) / session_path / 'simulation'
            for file_name in os.listdir(dir_files):
                if file_name[:17] == 'simulated_data_T1':
                    ret['Sim_T1'].append(file_name)
                elif file_name[:17] == 'simulated_data_T2':
                    ret['Sim_T2'].append(file_name)
                elif file_name[:10] == 'BMI_target':
                    ret['BMI_target'].append(file_name)
                elif file_name[:8] == 'strcMask':
                    ret['mask_data'].append(file_name)
                elif file_name[:10] == 'target_cal':
                    ret['target_calibration'].append(file_name)
    return pd.DataFrame(ret)


def find_folder_path(target: str):
    """ Function to find to which hard drive each mice belongs to """
    for key, paths in _FOLDER_PATHS.items():
        if target in paths:
            return key
    return None


def get_sessions_parquet(folder_save: Path, folder_list: list):
    """ Function to get every type of experiment DF saved in parquet """
    for experiment_type in AnalysisConstants.experiment_types:
        df = get_sessions_df(folder_list, experiment_type)
        df.to_parquet(folder_save / ("df_" + experiment_type + ".parquet"))


def get_simulations_parquet(folder_save: Path, folder_list: list):
    """ Function to get every type of experiment DF saved in parquet """
    for experiment_type in AnalysisConstants.experiment_types:
        df = get_simulations_df(folder_list, experiment_type)
        df.to_parquet(folder_save / ("df_" + experiment_type + "_simulations.parquet"))
