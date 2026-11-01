import os
from posixpath import basename
import sys
import re
import difflib
import datetime
import time
import copy
from functools import partial
from typing import List, Literal, Union
import json
import traceback
import cv2
import tempfile
import shutil

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pprint import pprint

from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
from natsort import natsorted

import tifffile

import skimage
import skimage.io
import skimage.color

from . import GUI_INSTALLED

if GUI_INSTALLED:
    import pyqtgraph as pg
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import QMessageBox
    from qtpy.QtCore import (
        QRect, QRectF, QThread, QMutex, QWaitCondition, QEventLoop
    )

    from cellacdc import apps as acdc_apps
    from cellacdc import widgets as acdc_widgets

    from . import dialogs, html_func, qtworkers

try:
    from cellacdc.urls import data_structure_docs_url
except ImportError:
    from cellacdc import data_structure_docs_url
    
from cellacdc import myutils as acdc_myutils
from cellacdc import load as acdc_load
try:
    from cellacdc.regex import float_regex
except Exception as err:
    from cellacdc.acdc_regex import float_regex

float_re = float_regex()

import cellacdc.features

from . import utils, config
from . import core, printl, error_up_str
from . import settings_path
from . import last_used_ini_text_filepath
from . import transformations
from . import DFs_FILENAMES
from . import spotmax_appdata_path
from . import docs

acdc_df_bool_cols = [
    'is_cell_dead',
    'is_cell_excluded',
    'is_history_known',
    'corrected_assignment'
]

# Keep compatibility with older ini files that had older kwargs names
# see io.nnet_params_from_ini_params
LEGACY_MODEL_KWARGS = {
    'model_doi_url_rdf_or_zip_path': 'model_doi_url_or_zip_path'
}

df_spots_filename_parts_pattern = (
    r'^(\d+)_(4_[012]|[012])_(detected_spots|valid_spots|spotfit)(.*)\.(\w+)$'
)

group_feature_to_col_mapper = docs.single_spot_features_column_names()

class MultipleFilesFound(Exception):
    pass

def read_json(json_path, logger_func=print, desc='custom annotations'):
    json_data = {}
    try:
        with open(json_path) as file:
            json_data = json.load(file)
    except Exception as e:
        print('****************************')
        logger_func(traceback.format_exc())
        print('****************************')
        logger_func(f'json path: {json_path}')
        print('----------------------------')
        logger_func(f'Error while reading saved {desc}. See above')
        print('============================')
    return json_data

def h5py_iter(g, prefix=''):
    for key, item in g.items():
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group): # test for group (go down)
            yield from h5py_iter(item, path)

def h5dump_to_arr(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as f:
        for (path, dset) in h5py_iter(f):
            data_dict[dset.name] = dset[()]
    sorted_keys = natsorted(data_dict.keys())
    arr = np.array([data_dict[key] for key in sorted_keys])
    return arr

def get_user_ch_paths(images_paths, user_ch_name):
    user_ch_file_paths = []
    for images_path in images_paths:
        img_aligned_found = False
        for filename in utils.listdir(images_path):
            if filename.find(f'{user_ch_name}_aligned.np') != -1:
                img_path_aligned = f'{images_path}/{filename}'
                img_aligned_found = True
            elif filename.find(f'{user_ch_name}.tif') != -1:
                img_path_tif = f'{images_path}/{filename}'

        if img_aligned_found:
            img_path = img_path_aligned
        else:
            img_path = img_path_tif
        user_ch_file_paths.append(img_path)
        print(f'Loading {img_path}...')
    return user_ch_file_paths

def get_segm_files(images_path):
    ls = utils.listdir(images_path)

    segm_files = [
        file for file in ls if file.endswith('.npz')
        and file.find('_segm') != -1
    ]
    return segm_files

def get_existing_segm_endnames(basename, segm_files):
    existing_endnames = []
    for f in segm_files:
        filename, _ = os.path.splitext(f)
        endname = filename[len(basename):]
        # Remove the 'segm_' part
        # endname = endname.replace('segm', '', 1).replace('_', '', 1)
        # endname = endname.replace('_', '', 1)
        existing_endnames.append(endname)
    return existing_endnames

def get_endname_from_channels(filename, channels):
    endname = None
    for ch in channels:
        ch_aligned = f'{ch}_aligned'
        m = re.search(fr'{ch}(.\w+)*$', filename)
        m_aligned = re.search(fr'{ch_aligned}(.\w+)*$', filename)
        if m_aligned is not None:
            return endname
        elif m is not None:
            return endname

def get_basename(files):
    basename = files[0]
    for file in files:
        # Determine the basename based on intersection of all files
        _, ext = os.path.splitext(file)
        sm = difflib.SequenceMatcher(None, file, basename)
        i, j, k = sm.find_longest_match(
            0, len(file), 0, len(basename)
        )
        basename = file[i:i+k]
    return basename

def get_pos_foldernames(exp_path):
    ls = utils.listdir(exp_path)
    pos_foldernames = [
        pos for pos in ls if pos.find('Position_')!=-1
        and os.path.isdir(os.path.join(exp_path, pos))
        and os.path.exists(os.path.join(exp_path, pos, 'Images'))
    ]
    return pos_foldernames

def pd_int_to_bool(acdc_df, colsToCast=None):
    if colsToCast is None:
        colsToCast = acdc_df_bool_cols
    for col in colsToCast:
        try:
            acdc_df[col] = acdc_df[col] > 0
        except KeyError:
            continue
    return acdc_df

def pd_bool_to_int(acdc_df, colsToCast=None, csv_path=None, inplace=True):
    """
    Function used to convert "FALSE" strings and booleans to 0s and 1s
    to avoid pandas interpreting as strings or numbers
    """
    if not inplace:
        acdc_df = acdc_df.copy()
    if colsToCast is None:
        colsToCast = acdc_df_bool_cols
    for col in colsToCast:   
        try:
            series = acdc_df[col]
            notna_idx = series.notna()
            notna_series = series.dropna()
            isInt = pd.api.types.is_integer_dtype(notna_series)
            isFloat = pd.api.types.is_float_dtype(notna_series)
            isObject = pd.api.types.is_object_dtype(notna_series)
            isString = pd.api.types.is_string_dtype(notna_series)
            isBool = pd.api.types.is_bool_dtype(notna_series)
            if isFloat or isBool:
                acdc_df.loc[notna_idx, col] = acdc_df.loc[notna_idx, col].astype(int)
            elif isString or isObject:
                # Object data type can have mixed data types so we first convert
                # to strings
                acdc_df.loc[notna_idx, col] = acdc_df.loc[notna_idx, col].astype(str)
                acdc_df.loc[notna_idx, col] = (
                    acdc_df.loc[notna_idx, col].str.lower() == 'true'
                ).astype(int)
        except KeyError:
            continue
        except Exception as e:
            printl(col)
            traceback.print_exc()
    if csv_path is not None:
        acdc_df.to_csv(csv_path)
    return acdc_df

def is_pos_path(path):
    if not os.path.isdir(path):
        return False
    folder_name = os.path.basename(path)
    if folder_name.find('Position_') != -1:
        return True
    else:
        return False

def is_images_path(path, check_parent=False):
    if not os.path.isdir(path):
        return False
    folder_name = os.path.basename(path)
    if not check_parent:
        return folder_name == 'Images'
    
    parent_foldername = os.path.basename(os.path.dirname(path))
    return folder_name == 'Images' and parent_foldername.startswith('Position_')

def _load_video(path):
    video = cv2.VideoCapture(path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(num_frames):
        _, frame = video.read()
        if frame.shape[-1] == 3:
            frame = skimage.color.rgb2gray(frame)
        if i == 0:
            chData = np.zeros((num_frames, *frame.shape), frame.dtype)
        chData[i] = frame
    return chData

def load_image_data(path: os.PathLike, to_float=False, return_dtype=False):
    filename, ext = os.path.splitext(path)
    if ext == '.h5':
        h5f = h5py.File(path, 'r')
        image_data = h5f['data']
    elif ext == '.npz':
        with np.load(path) as data:
            key = list(data.keys())[0]
            image_data = np.load(path)[key]
    elif ext == '.npy':
        image_data = np.load(path)
    elif ext == '.tif' or ext == '.tiff':
        image_data = tifffile.imread(path)
    else:
        try:
            image_data = skimage.io.imread(path)
        except Exception as e:
            image_data = _load_video(path)
    _dtype = image_data.dtype
    if to_float:
        image_data = acdc_myutils.img_to_float(image_data)
    if return_dtype:
        return image_data, _dtype
    else:
        return image_data

def save_image_data(filepath, img_data):
    filename, ext = os.path.splitext(filepath)
    if ext == '.h5':
        with h5py.File(filepath, "w") as h5f:
            h5f.create_dataset("data", data=img_data)
    elif ext == '.npz':
        np.savez_compressed(filepath, img_data)
    elif ext == '.npy':
        np.save(filepath, img_data)
    elif ext == '.tif' or ext == '.tiff':
        tifffile.imwrite(filepath, img_data)
    else:
        skimage.io.imsave(filepath, img_data)

def readStoredParamsCSV(csv_path, params):
    """Read old format of analysis_inputs.csv file from SpotMAX v1"""
    old_csv_options_to_anchors = {
        # 'Calculate ref. channel network length?':
        #     ('Reference channel', 'calcRefChNetLen'),
        'Compute spots size?':
            ('Spots channel', 'doSpotFit'),
        'emission wavelength (nm):':
            ('METADATA', 'emWavelen'),
        'Filter spots by reference channel?':
            ('Reference channel', 'bkgrMaskOutsideRef'),
        'Fit 3D Gaussians?':
            ('Spots channel', 'doSpotFit'),
        'Gaussian filter sigma:':
            ('Pre-processing', 'gaussSigma'),
        'Is ref. channel a single object per cell?':
            ('Reference channel', 'refChSingleObj'),
        'Load a reference channel?':
            ('Reference channel', 'segmRefCh'),
        'Local or global threshold for spot detection?':
            ('Pre-processing', 'aggregate'),
        'Numerical aperture:':
            ('METADATA', 'numAperture'),
        'Peak finder threshold function:':
            ('Spots channel', 'spotThresholdFunc'),
        'Reference channel threshold function:':
            ('Reference channel', 'refChThresholdFunc'),
        'Sharpen image prior spot detection?':
            ('Pre-processing', 'sharpenSpots'),
        'Spotsize limits (pxl)':
            ('Spots channel', ('minSpotSize', 'maxSpotSize')),
        'YX resolution multiplier:':
            ('METADATA', 'yxResolLimitMultiplier'),
        'Z resolution limit (um):':
            ('METADATA', 'zResolutionLimit'),
        'ZYX voxel size (um):':
            ('METADATA', ('voxelDepth', 'pixelHeight', 'pixelWidth')),
    }
    df = pd.read_csv(csv_path, index_col='Description')
    for idx, section_anchor in old_csv_options_to_anchors.items():
        section, anchor = section_anchor
        try:
            value = df.at[idx, 'Values']
        except Exception as e:
            try:
                idxMask = df.index.str.contains(idx, regex=False)
                if any(idxMask):
                    value = df['Values'][idxMask].iloc[0]
                else:
                    raise e
            except Exception as e:
                print(f'"{idx}" not found in CSV file. Using default value.')
                value = None
        
        if idx == 'ZYX voxel size (um):':
            # ZYX voxel size (um): in .csv is saved as `[x, y, z]`
            value = value[1:-1].split(',')
        if idx == 'Local or global threshold for spot detection?':
            # Store global or local as boolean
            value = True if value == 'Global' else False
        if idx == 'Spotsize limits (pxl)':
            value = value.split(',')
        
        if idx == 'Filter spots by reference channel?':
            # Manually set this since old spotmax was also keeping only spots 
            # on reference channel on top of comparing to reference channel
            params[section]['keepPeaksInsideRef']['loadedVal'] = value
        
        if isinstance(anchor, tuple):
            for val, sub_anchor in zip(value, anchor):
                params[section][sub_anchor]['loadedVal'] = val
        else:
            params[section][anchor]['loadedVal'] = value
    
    # Read the filtering method and values and format into ini format
    gop_method = df.at['Filter good peaks method:', 'Values']
    SECTION = 'Spots channel'
    ANCHOR = 'gopThresholds'
    if gop_method == 't-test':
        p_val = df.at['p-value limit:', 'Values']
        value = (
            f'spot_vs_ref_ch_ttest_pvalue,None,{p_val}'
            '\nspot_vs_ref_ch_ttest_tstat,0'
        )
        params[SECTION][ANCHOR]['loadedVal'] = value
    elif gop_method == 'effect size':
        eff_size_limit = df.at['Effect size limit:', 'Values']
        which_eff_size = df.at['Effect size used:', 'Values']
        eff_size_name = which_eff_size.split('_')[1]
        value = f'spot_vs_backgr_effect_size_{eff_size_name},{eff_size_limit}'
        params[SECTION][ANCHOR]['loadedVal'] = value
    return params

def readStoredParamsINI(ini_path, params, cast_dtypes=True):
    sections = list(params.keys())
    section_params = list(params.values())
    configPars = config.ConfigParser()
    configPars.read(ini_path, encoding="utf-8")
    for section, section_params in zip(sections, section_params):
        anchors = list(section_params.keys())
        for anchor in anchors:
            isEditableDesc = section_params[anchor].get('useEditableLabel', False)
            if isEditableDesc:
                if section in configPars.sections():
                    options = configPars.options(section)
                else:
                    options = ('', )
            else:
                options = (section_params[anchor]['desc'],)
            
            for o, option in enumerate(options):
                if o > 0:
                    added_anchor = f'{anchor}_{o}'
                    params[section][added_anchor] = (
                        params[section][anchor].copy()
                    )
                    anchor = added_anchor
                
                defaultVal = section_params[anchor]['initialVal']
                config_value = None
                if not configPars.has_section(section):
                    params[section][anchor]['isSectionInConfig'] = False
                    params[section][anchor]['loadedVal'] = None
                    params[section][anchor]['desc'] = option
                    continue
                else:
                    params[section][anchor]['isSectionInConfig'] = True
                
                is_option_in_ini = configPars.has_option(section, option)
                is_endname = option.endswith('end name')
                if not is_option_in_ini and is_endname:
                    # Check old option that had "or path" at the end
                    option = f'{option} or path' 
                                    
                is_do_spotfit = option == 'Compute spots size (fit gaussian peak(s))'
                
                if not is_option_in_ini and is_do_spotfit:
                    # New doSpotFit desc is not in INI --> check old one
                    option = 'Compute spots size'
                
                if not configPars.has_option(section, option):
                    params[section][anchor]['loadedVal'] = None
                    continue
                
                if cast_dtypes:
                    dtype = params[section][anchor].get('dtype')
                else:
                    dtype = None
                
                try:
                    str_val = configPars.get(section, option)
                except Exception as e:
                    str_val = None
                
                if callable(dtype):
                    config_value = dtype(configPars.get(section, option))
                elif isinstance(defaultVal, bool):
                    try:
                        config_value = configPars.getboolean(section, option)
                    except Exception as e:
                        config_value = None
                elif isinstance(defaultVal, float):
                    try:
                        config_value = configPars.getfloat(section, option)
                    except Exception as e:
                        config_value = str_val
                elif isinstance(defaultVal, int):
                    try:
                        config_value = configPars.getint(section, option)
                    except Exception as e:
                        config_value = str_val
                elif isinstance(defaultVal, str) or defaultVal is None:
                    try:
                        config_value = configPars.get(section, option)
                    except Exception as e:
                        config_value = None

                if option == 'Spots segmentation method':
                    if config_value == 'Neural network':
                        # Keep compatibility with oldere ini files that had 
                        # Spots segmentation method = Neural network
                        config_value = 'spotMAX AI'
                
                params[section][anchor]['loadedVal'] = config_value
                params[section][anchor]['desc'] = option
    
    params = add_neural_network_params(params, configPars)
    return params

def save_preprocessed_img_data(
        img_data, 
        raw_img_filepath, 
        basename, 
        ch_endname, 
        run_number, 
        text_id='_preprocessed',
        text_to_append='', 
        cast_to_dtype=None, 
        pad_width=None, 
        verbose=True, 
        logger_func=print
    ):
    if verbose:
        logger_func(
            f'Saving pre-processed image data from channel "{ch_endname}"'
        )
    
    if not basename.endswith('_'):
        basename = f'{basename}_'
        
    if cast_to_dtype is not None:
        img_data = acdc_myutils.float_img_to_dtype(img_data, cast_to_dtype)
    
    if pad_width is not None:
        img_data = np.pad(img_data, pad_width)
    
    in_filename = os.path.basename(raw_img_filepath)
    _, ext = os.path.splitext(in_filename)
    
    out_filename = f'{basename}run_num{run_number}_{ch_endname}{text_id}'
    if text_to_append:
        if not text_to_append.startswith('_'):
            text_to_append = f'_{text_to_append}'
        out_filename = f'{out_filename}{text_to_append}'
    out_filename = f'{out_filename}{ext}'    
    in_folderpath = os.path.dirname(raw_img_filepath)
    
    out_filepath = os.path.join(in_folderpath, out_filename)
    
    save_image_data(out_filepath, np.squeeze(img_data))
    
    if verbose:
        logger_func(
            f'{text_id.capitalize()} image data from channel '
            f'"{ch_endname}" saved to "{out_filepath}"...'
        )

def save_nnet_pred_map(
        nnet_pred_map, raw_img_filepath, basename, ch_endname, run_number, 
        text_to_append='', pad_width=None, verbose=True, logger_func=print
    ):
    if verbose:
        logger_func(
            f'Saving SpotMAX AI prediction map from channel "{ch_endname}"...'
        )
    
    if not basename.endswith('_'):
        basename = f'{basename}_'
    
    if pad_width is not None:
        nnet_pred_map = np.pad(nnet_pred_map, pad_width)
    
    out_filename = f'{basename}run_num{run_number}_{ch_endname}_AI_pred_map'
    if text_to_append:
        if not text_to_append.startswith('_'):
            text_to_append = f'_{text_to_append}'
        out_filename = f'{out_filename}{text_to_append}'
    out_filename = f'{out_filename}.npz'    
    in_folderpath = os.path.dirname(raw_img_filepath)
    
    out_filepath = os.path.join(in_folderpath, out_filename)
    
    np.savez_compressed(out_filepath, np.squeeze(nnet_pred_map))
    
    if verbose:
        logger_func(
            f'SpotMAX AI prediction map from channel "{ch_endname}" saved to '
            f'"{out_filepath}"...'
        )

def add_neural_network_params(params, configPars):
    sections = [
        'neural_network.init', 'neural_network.segment',
        'bioimageio_model.init', 'bioimageio_model.segment',
        'spotiflow.init', 'spotiflow.segment'
    ]
    sub_sections = ['spots', 'ref']
    for section in sections:
        for sub_section in sub_sections:
            section_name = f'{section}.{sub_section}'
            if section_name not in configPars.sections():
                continue
            for key, value in configPars[section_name].items():
                if section_name not in params:
                    params[section_name] = {}
                params[section_name][key] = {
                    'desc': key, 'loadedVal': value, 'isParam': True
                }
    return params

def sort_config_parser_ini(cp, params):
    default_params = config.getDefaultParams()
    sorted_cp = config.ConfigParser()
    for section, section_params in default_params.items():
        sorted_cp[section] = {}
        for anchor, param in section_params.items():
            option = param['desc']
            value = cp[section].get(option)
            if value is None:
                value = str(params[section][anchor]['loadedVal'])
            sorted_cp[section][option] = value
    
    # Add sections that were not in default params
    for section in cp.sections():
        if not sorted_cp.has_section(section):
            sorted_cp[section] = {}
        for option in cp.options(section):
            if sorted_cp.has_option(section, option):
                continue
            
            sorted_cp[section][option] = cp[section][option]

    return sorted_cp

def add_metadata_from_csv(csv_path, ini_params):
    df = pd.read_csv(csv_path).set_index('Description')
    metadata = ini_params['METADATA']

    SizeT = df.at['SizeT', 'values']
    SizeZ = df.at['SizeZ', 'values']
    metadata['SizeT']['loadedVal'] = int(SizeT)
    metadata['SizeZ']['loadedVal'] = int(SizeZ)

    pixelWidth = df.at['PhysicalSizeX', 'values']
    pixelHeight = df.at['PhysicalSizeY', 'values']
    voxelDepth = df.at['PhysicalSizeZ', 'values']
    loadedPixelWidth = metadata['pixelWidth']['loadedVal']
    if loadedPixelWidth == 0:
        metadata['pixelWidth']['loadedVal'] = float(pixelWidth)

    loadedpixelHeight = metadata['pixelHeight']['loadedVal']
    if loadedpixelHeight == 0:
        metadata['pixelHeight']['loadedVal'] = float(pixelHeight)

    loadedVoxelDepth = metadata['voxelDepth']['loadedVal']
    if loadedVoxelDepth == 0:
        metadata['voxelDepth']['loadedVal'] = float(voxelDepth)
    return ini_params

def writeConfigINI(params=None, ini_path=None):
    configPars = config.ConfigParser()

    if params is None:
        params = config.analysisInputsParams()

    # Create sections
    for section, anchors in params.items():
        configPars[section] = {}
        for param in anchors.values():
            if not param.get('isParam', True):
                continue
            key = param['desc']
            if not key:
                continue
            val = param.get('loadedVal')
            if val is None:
                val = param['initialVal']
            parser_func = param.get('parser')
            if parser_func is not None:
                val = parser_func(val)
            comment = param.get('comment')
            if comment is not None:
                # Add comment to config file
                configPars.set(section, comment())
            configPars[section][key] = str(val)

    if ini_path is None:
        return configPars

    # Write config to file
    with open(ini_path, 'w', encoding="utf-8") as file:
        configPars.write(file)
    
    return configPars

def _load_spots_table_h5(filepath):
    with pd.HDFStore(filepath, mode='r') as store:
        dfs = []
        keys = []
        for key in store.keys():
            df = store.get(key)
            frame_i = int(re.findall(r'frame_(\d+)', key)[0])
            dfs.append(df)
            keys.append(frame_i)
    df = pd.concat(dfs, keys=keys, names=['frame_i']).sort_index()
    return df

def disable_saving_masks_configparser(configparser):
    section = 'Reference channel'
    anchor = 'saveRefChFeatures'
    configparser[section][anchor] = 'False'
    
    anchor = 'saveRefChMask'
    configparser[section][anchor] = 'False'
    
    anchor = 'saveRefChPreprocImage'
    configparser[section][anchor] = 'False'
    
    section = 'Spots channel'
    anchor = 'saveSpotsMask'
    configparser[section][anchor] = 'False'
    
    anchor = 'saveSpotsPreprocImage'
    configparser[section][anchor] = 'False'
    return configparser

def load_spots_table(spotmax_out_path, filename, filepath=None):
    if filepath is not None:
        return load_table_to_df(filepath, index_col=['frame_i', 'Cell_ID'])
    
    filepath = os.path.join(spotmax_out_path, filename)
    if not os.path.exists(filepath):
        return
    
    if not filename:
        return
    
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath, index_col=['frame_i', 'Cell_ID'])
    elif filename.endswith('.h5'):
        df = _load_spots_table_h5(filepath)
    
    if df.empty:
        return
    
    return df

def load_spots_table_from_run_num(
        spotmax_out_path: os.PathLike, 
        run_num: int,
        category: Literal['detected_spots', 'valid_spots', 'spotfit'], 
        only_filepath=False
    ) -> Union[pd.DataFrame, str]:
    for file in os.listdir(spotmax_out_path):
        if not file.startswith(f'{run_num}'):
            continue
        
        if category not in file:
            continue
        
        filepath = os.path.join(spotmax_out_path, file)
        if only_filepath:
            return filepath

        return load_table_to_df(filepath)

def get_ini_value_from_run_num(
        spotmax_out_path: os.PathLike,
        run_num: int,
        section: str,
        option: str,
        getter='get'
    ):
    for file in os.listdir(spotmax_out_path):
        if not file.startswith(f'{run_num}_analysis_parameters'):
            continue
        
        cp = config.ConfigParser()
        cp.read(os.path.join(spotmax_out_path, file))
        getter_func = getattr(cp, getter)
        return getter_func(section, option)

def get_spots_channel_name_from_run_num(
        spotmax_out_path: os.PathLike,
        run_num: int
    ) -> str:
    section = 'File paths and channels'
    option = 'Spots channel end name'
    channel_name = get_ini_value_from_run_num(
        spotmax_out_path, run_num, section, option
    )
    return channel_name            

def get_spots_channel_name_from_params(params):
    section = 'File paths and channels'
    anchor = 'spotsEndName'
    try:
        return params[section][anchor]['loadedVal']
    except Exception as err:
        return ''

def load_table_to_df(filepath, index_col=None):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, index_col=index_col)
    elif filepath.endswith('.h5'):
        df = _load_spots_table_h5(filepath)
    return df

class channelName:
    def __init__(self, which_channel=None, QtParent=None, load=True):
        self.parent = QtParent
        self.is_first_call = True
        self.which_channel = which_channel
        if load:
            self.last_sel_channel = self._load_last_selection()
        else:
            self.last_sel_channel = None
        self.was_cancelled = False

    def reloadLastSelectedChannel(self, which):
        self.which_channel = which
        self.last_sel_channel = self._load_last_selection()

    def checkDataIntegrity(self, filenames):
        char = filenames[0][:2]
        startWithSameChar = all([f.startswith(char) for f in filenames])
        if not startWithSameChar:
            txt = html_func.paragraph("""
                The system detected files inside the folder
                that <b>do not start with the same, common basename</b>
                (see which filenames in the box below).<br><br>
                To ensure correct loading of the data, the folder where
                the file(s) is/are should either contain a single image file or
                only files that <b>start with the same, common basename.</b><br><br>
                For example the following filenames:<br><br>
                F014_s01_phase_contr.tif<br>
                F014_s01_mCitrine.tif<br><br>
                are named correctly since they all start with the
                the common basename "F014_s01_". After the common basename you
                can write whatever text you want. In the example above,
                "phase_contr"  and "mCitrine" are the channel names.<br><br>
                We recommend using the module 0. or the provided Fiji/ImageJ
                macro to create the right data structure.<br><br>
                Data loading may still be successfull, so the system will
                still try to load data now.
            """)
            msg = acdc_widgets.myMessageBox()
            details = "\n".join(filenames)
            details = f'Files detected:\n\n{details}'
            msg.setDetailedText(details)
            msg.warning(
                self.parent, 'Data structure compromised', txt
            )
            return False
        return True

    def getChannels(
            self, filenames, images_path, useExt=('.tif', '.h5'),
            channelExt=('.h5', '.tif', '_aligned.npz'), 
            validEndnames=('aligned.npz', 'acdc_output.csv', 'segm.npz')
        ):
        # First check if metadata.csv already has the channel names
        metadata_csv_path = None
        for file in utils.listdir(images_path):
            if file.endswith('metadata.csv'):
                metadata_csv_path = os.path.join(images_path, file)
                break
        
        chNames_found = False
        channel_names = set()
        if metadata_csv_path is not None:
            df = pd.read_csv(metadata_csv_path)
            basename = None
            if 'Description' in df.columns:
                channelNamesMask = df.Description.str.contains(r'channel_\d+_name')
                channelNames = df[channelNamesMask]['values'].to_list()
                try:
                    basename = df.set_index('Description').at['basename', 'values']
                except Exception as e:
                    basename = None
                if channelNames:
                    # There are channel names in metadata --> check that they 
                    # are still existing as files
                    channel_names = channelNames.copy()
                    for chName in channelNames:
                        chSaved = []
                        for file in filenames:
                            patterns = (
                                f'{chName}.tif', f'{chName}_aligned.npz'
                            )
                            ends = [p for p in patterns if file.endswith(p)]
                            if ends:
                                pattern = ends[0]
                                chSaved.append(True)
                                m = tuple(re.finditer(pattern, file))[-1]
                                chName_idx = m.start()
                                if basename is None:
                                    basename = file[:chName_idx]
                                break
                        if not any(chSaved):
                            channel_names.remove(chName)

                    if basename is not None:
                        self.basenameNotFound = False
                        self.basename = basename
                elif channelNames and basename is not None:
                    self.basename = basename
                    self.basenameNotFound = False
                    channel_names = channelNames

            if channel_names and basename is not None:
                # Add additional channels existing as file but not in metadata.csv
                for file in filenames:
                    ends = [ext for ext in channelExt if file.endswith(ext)]
                    if ends:
                        endName = file[len(basename):]
                        chName = endName.replace(ends[0], '')
                        if chName not in channel_names:
                            channel_names.append(chName)
                return channel_names, False

        # Find basename as intersection of filenames
        channel_names = set()
        self.basenameNotFound = False
        isBasenamePresent = self.checkDataIntegrity(filenames)
        basename = filenames[0]
        for file in filenames:
            # Determine the basename based on intersection of all .tif
            _, ext = os.path.splitext(file)
            validFile = False
            if useExt is None:
                validFile = True
            elif ext in useExt:
                validFile = True
            elif any([file.endswith(end) for end in validEndnames]):
                validFile = True
            else:
                validFile = (
                    (file.find('_acdc_output_') != -1 and ext == '.csv')
                    or (file.find('_segm_') != -1 and ext == '.npz')
                )
            if not validFile:
                continue
            sm = difflib.SequenceMatcher(None, file, basename)
            i, j, k = sm.find_longest_match(0, len(file),
                                            0, len(basename))
            basename = file[i:i+k]
        self.basename = basename
        basenameNotFound = [False]
        for file in filenames:
            filename, ext = os.path.splitext(file)
            validImageFile = False
            if ext in channelExt:
                validImageFile = True
            elif file.endswith('aligned.npz'):
                validImageFile = True
                filename = filename[:-len('_aligned')]
            if not validImageFile:
                continue

            channel_name = filename.split(basename)[-1]
            channel_names.add(channel_name)
            if channel_name == filename:
                # Warn that an intersection could not be found
                basenameNotFound.append(True)
        channel_names = list(channel_names)
        if any(basenameNotFound):
            self.basenameNotFound = True
            filenameNOext, _ = os.path.splitext(basename)
            self.basename = f'{filenameNOext}_'
        if self.which_channel is not None:
            # Search for "phase" and put that channel first on the list
            if self.which_channel == 'segm':
                is_phase_contr_li = [c.lower().find('phase')!=-1
                                     for c in channel_names]
                if any(is_phase_contr_li):
                    idx = is_phase_contr_li.index(True)
                    channel_names[0], channel_names[idx] = (
                                      channel_names[idx], channel_names[0])
        return channel_names, any(basenameNotFound)

    def _load_last_selection(self):
        last_sel_channel = None
        ch = self.which_channel
        if self.which_channel is not None:
            _path = os.path.dirname(os.path.realpath(__file__))
            txt_path = os.path.join(settings_path, f'{ch}_last_sel.txt')
            if os.path.exists(txt_path):
                with open(txt_path) as txt:
                    last_sel_channel = txt.read()
        return last_sel_channel

    def _save_last_selection(self, selection):
        ch = self.which_channel
        if self.which_channel is not None:
            _path = os.path.dirname(os.path.realpath(__file__))
            if not os.path.exists(settings_path):
                os.mkdir(settings_path)
            txt_path = os.path.join(settings_path, f'{ch}_last_sel.txt')
            with open(txt_path, 'w') as txt:
                txt.write(selection)

    def askSelectChannel(self, parent, channel_names, informativeText='',
                 CbLabel='Select channel name to load:  '):
        font = QFont()
        font.setPixelSize(11)
        win = dialogs.QDialogCombobox(
            'Select channel name', channel_names,
            informativeText, CbLabel=CbLabel,
            parent=parent, defaultChannelName=self.last_sel_channel
        )
        win.setFont(font)
        win.exec_()
        if win.cancel:
            self.was_cancelled = True
        self.channel_name = win.selectedItemText
        if not win.cancel:
            self._save_last_selection(self.channel_name)
        self.is_first_call = False

    def setUserChannelName(self):
        if self.basenameNotFound:
            reverse_ch_name = self.channel_name[::-1]
            idx = reverse_ch_name.find('_')
            if idx != -1:
                self.user_ch_name = self.channel_name[-idx:]
            else:
                self.user_ch_name = self.channel_name[-4:]
        else:
            self.user_ch_name = self.channel_name

class expFolderScanner:
    def __init__(self, homePath='', logger_func=print):
        self.is_first_call = True
        self.expPaths = []
        self.homePath = homePath
        if homePath and logger_func is not None:
            logger_func(
                f'Experiment folder scanner initialized with path "{homePath}"'
            )
        self.logger_func = logger_func

    def getExpPaths(self, path, signals=None):
        """Recursively scan the directory tree to search for folders that
        contain Position folders. When found, the path will be appended to
        self.expPaths attribute

        Parameters
        ----------
        path : str or os.PathLike
            Path to check if it contains Position folders.
        signals : attribute of QObject subclass or None.
            If not None it is used to emit signals and communicate with
            main GUI thread (e.g., to update progress bar).

        Returns
        -------
        None

        """
        if not os.path.isdir(path):
            return

        if self.is_first_call:
            self.is_first_call = False
            if signals is not None:
                signals.progress.emit(
                    'Searching valid experiment folders...', 'INFO'
                )
                signals.initProgressBar.emit(0)

        ls = natsorted(utils.listdir(path))
        isExpPath = any([
            f.find('Position_')!=-1 and os.path.isdir(os.path.join(path, f))
            for f in ls
        ])

        if isExpPath:
            self.expPaths.append(path)
        else:
            with ThreadPoolExecutor(4) as ex:
                ex.map(self.getExpPaths, [os.path.join(path, f) for f in ls])

    def _setInfoExpPath(self, exp_path):
        """
        See infoExpPaths for more details
        """
        exp_path = os.path.normpath(exp_path)
        ls = natsorted(utils.listdir(exp_path))

        posFoldernames = natsorted([
            f for f in ls
            if f.find('Position_')!=-1
            and os.path.isdir(os.path.join(exp_path, f))
        ])

        self.paths[1][exp_path] = {
            'numPosSpotCounted': 0,
            'numPosSpotSized': 0,
            'posFoldernames': posFoldernames, 
            'spotCountedPosFoldernames': [], 
            'spotSizedPosFoldernames': [],
            'dateModified': np.nan
        }
        for p, pos in enumerate(posFoldernames):
            posPath = os.path.join(exp_path, pos)
            spotmaxOutPath = os.path.join(posPath, 'spotMAX_output')
            imagesPath = os.path.join(posPath, 'Images')
            isSpotmaxOutPresent = os.path.exists(spotmaxOutPath)
            if not isSpotmaxOutPresent:
                self.paths[1][exp_path][pos] = {
                    'isPosSpotCounted': False,
                    'isPosSpotSized': False, 
                    'timeAnalysed': np.nan
                }
            else:
                spotmaxFiles = natsorted(utils.listdir(spotmaxOutPath))
                if not spotmaxFiles:
                    continue
                run_nums = self.runNumbers(spotmaxOutPath)
                for run in run_nums:
                    analysisParamsFilepath = (
                        get_analysis_params_filepath_from_run(
                            spotmaxOutPath, run
                        )
                    )
                    runInfo = self.paths.get(run, {})
                    expInfo = runInfo.get(exp_path, {})
                    analysisInputs = expInfo.get('analysisInputs', False)
                    initRun = (
                        not runInfo or not expInfo or not analysisInputs
                    )
                    if initRun:
                        try:
                            analysisInputs = read_ini(analysisParamsFilepath)
                        except Exception as err:
                            continue
                        self.paths[run][exp_path] = {
                            'numPosSpotCounted': 0,
                            'numPosSpotSized': 0,
                            'posFoldernames': posFoldernames,
                            'analysisInputs': analysisInputs,
                            'spotCountedPosFoldernames': [],
                            'spotSizedPosFoldernames': [],
                        }

                    timeAnalysed = os.path.getmtime(analysisParamsFilepath)
                    isSpotCounted, isSpotSized = self.analyseRunNumber(
                        spotmaxOutPath, run
                    )
                    self.paths[run][exp_path][pos] = {
                        'isPosSpotCounted': isSpotCounted,
                        'isPosSpotSized': isSpotSized, 
                        'timeAnalysed': timeAnalysed
                    }
                    expInfo = self.paths[run][exp_path]
                    if isSpotCounted:
                        expInfo['numPosSpotCounted'] += 1
                        expInfo['spotCountedPosFoldernames'].append(pos)
                    if isSpotSized:
                        expInfo['numPosSpotSized'] += 1
                        expInfo['spotSizedPosFoldernames'].append(pos)

    def addMissingRunsInfo(self):
        # paths = copy.deepcopy(self.paths)
        missingKeys = []
        for run, runInfo in self.paths.items():
            for exp_path, expInfo in runInfo.items():
                posFoldernames = expInfo['posFoldernames']
                for pos in posFoldernames:
                    try:
                        posInfo = expInfo[pos]
                    except KeyError as e:
                        missingKey = (run, os.path.normpath(exp_path), pos)
                        missingKeys.append(missingKey)

        for keys in missingKeys:
            run, exp_path, pos = keys
            self.paths[run][exp_path][pos] = {
                'isPosSpotCounted': False,
                'isPosSpotSized': False, 
                'timeAnalysed': np.nan
            }
    
    def getExpPathsWithPosFoldernames(self):
        if not self.expPaths:
            self.getExpPaths(self.homePath)
        
        expPathsWithPosFoldernames = {}
        if not self.expPaths:
            self.expPaths = [self.homePath]
        for expPath in self.expPaths:
            isPosFolder = is_pos_folder(expPath)
            isImagesFolder = is_images_folder(expPath)
            if isPosFolder:
                exp_path = os.path.dirname(expPath)
                posFoldernames = [os.path.basename(expPath)]
            elif isImagesFolder:
                pos_path = os.path.dirname(expPath)
                exp_path = os.path.dirname(pos_path)
                posFoldernames = [os.path.basename(pos_path)]
            else:
                exp_path = expPath
                posFoldernames = get_pos_foldernames(exp_path)
            expPathsWithPosFoldernames[exp_path] = posFoldernames
        return expPathsWithPosFoldernames

    def infoExpPaths(self, expPaths, signals=None):
        """Method used to determine how each experiment was analysed.

        Parameters
        ----------
        expPaths : type
            Description of parameter `expPaths`.

        Sets
        -------
        self.paths: dict
            A nested dictionary with the following keys:
                expInfo = paths[run_number][exp_path] --> dict
                numPosSpotCounted = expInfo['numPosSpotCounted'] --> int
                numPosSpotSized = expInfo['numPosSpotSized'] --> int
                posFoldernames = expInfo['posFoldernames'] --> list of strings
                analysisInputs_df = expInfo['analysisInputs'] --> config.ConfigParser
                pos1_info = expInfo['Position_1'] --> dict
                    isPos1_spotCounted = pos1_info['isPosSpotCounted'] --> bool
                    isPos1_spotSized = pos1_info['isPosSpotSized'] --> bool
                    timeAnalysed = pos1_info['timeAnalysed'] --> seconds as returned by `os.path.getmtime`
        """
        self.paths = defaultdict(lambda: defaultdict(dict))

        if signals is not None:
            print('')
            signals.progress.emit(
                'Scanning experiment folder(s)...', 'INFO'
            )
        elif self.logger_func is not None:
            self.logger_func('Scanning experiment folders...')
        for exp_path in tqdm(expPaths, unit=' folder', ncols=100, leave=False):
            self._setInfoExpPath(exp_path)
            if signals is not None:
                signals.progressBar.emit(1)

        self.addMissingRunsInfo()

    def loadAnalysisInputs(self, spotmaxOutPath, run):
        df = None
        for file in utils.listdir(spotmaxOutPath):
            match_csv = re.match(fr'{run}_(\w*)analysis_inputs\.csv', file)
            match_ini = re.match(fr'{run}_analysis_parameters(.*)\.ini', file)
            if match_csv is not None:
                csvPath = os.path.join(spotmaxOutPath, file)
                df = pd.read_csv(csvPath, index_col='Description')
                df1 = utils.pdDataFrame_boolTo0s1s(df, labelsToCast='allRows')
                if not df.equals(df1):
                    df1.to_csv(csvPath)
                    df = df1
                return df, csvPath
            if match_ini is not None:
                iniPath = os.path.join(spotmaxOutPath, file)
                configPars = config.ConfigParser()
                configPars.read(os.path.join(spotmaxOutPath, file))
                return configPars, iniPath

    def runNumbers(self, spotmaxOutPath):
        run_nums = set()
        spotmaxFiles = natsorted(utils.listdir(spotmaxOutPath))
        if not spotmaxFiles:
            return run_nums
        run_nums = [
            re.findall(r'(\d+)_(\d)_detected', f) for f in spotmaxFiles
        ]
        run_nums = [int(m[0][0]) for m in run_nums if m]
        run_nums = set(run_nums)
        return run_nums

    def analyseRunNumber(self, spotmaxOutPath, run):
        patterns = [
            filename.replace('*rn*', str(run)).replace('*desc*', '')
            for filename in DFs_FILENAMES.values()
        ]

        isPosSpotCounted = False
        isPosSpotSized = False
        for file in utils.listdir(spotmaxOutPath):
            for pattern in patterns:
                if file.find(pattern) != -1:
                    isPosSpotCounted = True
                    if file.find('2_spotfit') != -1:
                        isPosSpotSized = True

        return isPosSpotCounted, isPosSpotSized

    def warnNoValidExperimentsFound(self, homePath, parent):
        txt = html_func.paragraph(f"""
            The following folder does not contain any valid experiment:<br><br>
            <code>{homePath}</code><br><br>
            For more information about the correct folder structure see 
            {html_func.href('here', data_structure_docs_url)}.
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        helpButton = acdc_widgets.helpPushButton('Help...')
        msg.addButton(helpButton)
        helpButton.clicked.disconnect()
        helpButton.clicked.connect(
            partial(acdc_myutils.browse_url, data_structure_docs_url)
        )
        msg.addShowInFileManagerButton(str(homePath))
        msg.warning(parent, 'No valid experiments found!', txt)
    
    def input(self, parent=None, app=None, singleTopLevelItemSelection=True):
        if len(self.paths) == 0:
            self.warnNoValidExperimentsFound(self.homePath, parent)
            self.selectedPaths = []
            return
        win = dialogs.selectPathsSpotmax(
            self.paths, self.homePath, parent=parent, app=app,
            singleTopLevelItemSelection=singleTopLevelItemSelection
        )
        win.exec_()
        self.selectedPaths = win.selectedPaths

    def validPosPaths(self):
        pass

class loadData:
    def __init__(self, channelDataPath, user_ch_name, QParent=None):
        # Additional loaded data
        self.loadedRelativeFilenamesData = {}

        # Dictionary of keys to keep track which channels are merged
        self.loadedMergeRelativeFilenames = {}

        # Dictionary of keys to keep track which channel is skeletonized
        self.skeletonizedRelativeFilename = ''

        # Dictionary of keys to keep track which channel is contoured
        self.contouredRelativeFilename = ''

        # Gradient levels for each channel name (layer 0, layer 1, etc)
        self.gradientLevels = {}

        # Skeleton coords as calulcated in self.skeletonize()
        self.skelCoords = {}

        # Contour coords as calulcated in self.contours()
        self.contCoords = {}

        # For .h5 files we can load a subset of the entire file.
        # loadSizeT and loadSizeZ are asked at askInputMetadata method
        self.loadSizeT, self.loadSizeZ = None, None

        self.bkgrROIs = []
        self.parent = QParent
        self.channelDataPath = str(channelDataPath)
        self.user_ch_name = user_ch_name
        self.images_path = os.path.dirname(channelDataPath)
        self.pos_path = os.path.dirname(self.images_path)
        self.h5_path = ''
        self.spotmaxOutPath = os.path.join(self.pos_path, 'spotMAX_output')
        self.exp_path = os.path.dirname(self.pos_path)
        self.pos_foldername = os.path.basename(self.pos_path)
        self.cropROI = None
        path_li = os.path.normpath(channelDataPath).split(os.sep)
        self.relPath = os.path.join('', *path_li[-4:])
        filename_ext = os.path.basename(channelDataPath)
        self.filename, self.ext = os.path.splitext(filename_ext)
        self.cca_df_colnames = [
            'cell_cycle_stage',
            'generation_num',
            'relative_ID',
            'relationship',
            'emerg_frame_i',
            'division_frame_i',
            'is_history_known',
            'corrected_assignment'
        ]
        self.loadLastEntriesMetadata()

    def loadLastEntriesMetadata(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(settings_path):
            self.last_md_df = None
            return
        csv_path = os.path.join(settings_path, 'last_entries_metadata.csv')
        if not os.path.exists(csv_path):
            self.last_md_df = None
        else:
            self.last_md_df = pd.read_csv(csv_path).set_index('Description')

    def saveLastEntriesMetadata(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists:
            return
        csv_path = os.path.join(settings_path, 'last_entries_metadata.csv')
        self.metadata_df.to_csv(csv_path)

    def getBasenameAndChNames(self, load=True):
        ls = utils.listdir(self.images_path)
        channelNameUtil = channelName(load=load)
        self.chNames, _ = channelNameUtil.getChannels(ls, self.images_path)
        self.basename = channelNameUtil.basename
        self.allRelFilenames = [
            file[len(self.basename):] for file in ls
            if os.path.splitext(file)[1] == '.tif'
            or os.path.splitext(file)[1] == '.npy'
            or os.path.splitext(file)[1] == '.npz'
            or os.path.splitext(file)[1] == '.h5'
        ]

    def checkH5memoryFootprint(self):
        if self.ext != '.h5':
            return 0
        else:
            Y, X = self.h5_dset.shape[-2:]
            size = self.loadSizeT*self.loadSizeZ*Y*X
            itemsize = self.h5_dset.dtype.itemsize
            required_memory = size*itemsize
            return required_memory

    def shouldLoadTchunk(self, current_t):
        if self.ext != '.h5':
            return False

        coord1_window = self.t0_window + self.loadSizeT - 1
        halfWindowSize = int(self.loadSizeT/2)

        coord0_chunk = coord1_window + 1
        chunkSize = current_t + halfWindowSize - coord0_chunk + 1

        rightBoundary = self.SizeT-halfWindowSize
        leftBoundary = halfWindowSize

        if current_t <= halfWindowSize and leftBoundary >= self.t0_window:
            return False
        elif current_t >= rightBoundary and rightBoundary <= coord1_window:
            return False

        return True

    def shouldLoadZchunk(self, current_idx):
        if self.ext != '.h5':
            return False

        coord1_window = self.z0_window + self.loadSizeZ - 1
        halfWindowSize = int(self.loadSizeZ/2)

        coord0_chunk = coord1_window + 1
        chunkSize = current_idx + halfWindowSize - coord0_chunk + 1

        rightBoundary = self.SizeZ-halfWindowSize
        leftBoundary = halfWindowSize
        if current_idx <= halfWindowSize and leftBoundary >= self.z0_window:
            return False
        elif current_idx >= rightBoundary and rightBoundary <= coord1_window:
            return False

        return True

    def loadChannelDataChunk(self, current_idx, axis=0, worker=None):
        is4D = self.SizeZ > 1 and self.SizeT > 1
        is3Dz = self.SizeZ > 1 and self.SizeT == 1
        is3Dt = self.SizeZ == 1 and self.SizeT > 1
        is2D = self.SizeZ == 1 and self.SizeT == 1
        if is4D:
            if axis==0:
                axis1_range = (self.z0_window, self.z0_window+self.loadSizeZ)
                chData, t0_window, z0_window = utils.shiftWindow_axis0(
                    self.h5_dset, self.chData, self.loadSizeT, self.t0_window,
                    current_idx, axis1_interval=axis1_range, worker=worker
                )
            elif axis==1:
                axis0_range = (self.t0_window, self.t0_window+self.loadSizeT)
                chData, t0_window, z0_window = utils.shiftWindow_axis1(
                    self.h5_dset, self.chData, self.loadSizeZ, self.z0_window,
                    current_idx, axis0_interval=axis0_range, worker=worker
                )
        elif is3Dz:
            chData, t0_window, z0_window = utils.shiftWindow_axis0(
                self.h5_dset, self.chData, self.loadSizeZ, self.z0_window,
                current_idx, axis1_interval=None, worker=worker
            )
        elif is3Dt:
            chData, t0_window, z0_window = utils.shiftWindow_axis0(
                self.h5_dset, self.chData, self.loadSizeT, self.t0_window,
                current_idx, axis1_interval=None, worker=worker
            )
        self.chData = chData
        self.t0_window = t0_window
        self.z0_window = z0_window

    def loadChannelData(self):
        self.z0_window = 0
        self.t0_window = 0
        if self.ext == '.h5':
            self.h5f = h5py.File(self.channelDataPath, 'r')
            self.h5_dset = self.h5f['data']
            self.chData_shape = self.h5_dset.shape
            readH5 = self.loadSizeT is not None or self.loadSizeZ is not None
            if not readH5:
                return

            is4D = self.SizeZ > 1 and self.SizeT > 1
            is3Dz = self.SizeZ > 1 and self.SizeT == 1
            is3Dt = self.SizeZ == 1 and self.SizeT > 1
            is2D = self.SizeZ == 1 and self.SizeT == 1
            if is4D:
                midZ = int(self.SizeZ/2)
                halfZLeft = int(self.loadSizeZ/2)
                halfZRight = self.loadSizeZ-halfZLeft
                z0 = midZ-halfZLeft
                z1 = midZ+halfZRight
                self.z0_window = z0
                self.t0_window = 0
                self.chData = self.h5_dset[:self.loadSizeT, z0:z1]
            elif is3Dz:
                midZ = int(self.SizeZ/2)
                halfZLeft = int(self.loadSizeZ/2)
                halfZRight = self.loadSizeZ-halfZLeft
                z0 = midZ-halfZLeft
                z1 = midZ+halfZRight
                self.z0_window = z0
                self.chData = np.squeeze(self.h5_dset[z0:z1])
            elif is3Dt:
                self.t0_window = 0
                self.chData = np.squeeze(self.h5_dset[:self.loadSizeT])
            elif is2D:
                self.chData = self.h5_dset[:]
        elif self.ext == '.npz':
            self.chData = np.load(self.channelDataPath)['arr_0']
            self.chData_shape = self.chData.shape
        elif self.ext == '.npy':
            self.chData = np.load(self.channelDataPath)
            self.chData_shape = self.chData.shape
        else:
            try:
                self.chData = skimage.io.imread(self.channelDataPath)
                self.chData_shape = self.chData.shape
            except ValueError:
                self.chData = _load_video(self.channelDataPath)
                self.chData_shape = self.chData.shape
            except Exception as e:
                traceback.print_exc()
                self.criticalExtNotValid()

    def absoluteFilename(self, relFilename):
        absoluteFilename = f'{self.basename}{relFilename}'
        return absoluteFilename

    def absoluteFilepath(self, relFilename):
        absoluteFilename = f'{self.basename}{relFilename}'
        return os.path.join(self.images_path, absoluteFilename)

    def loadOtherFiles(
            self,
            load_segm_data=False,
            load_acdc_df=False,
            load_shifts=False,
            loadSegmInfo=False,
            load_delROIsInfo=False,
            loadBkgrData=False,
            loadBkgrROIs=False,
            load_last_tracked_i=False,
            load_metadata=False,
            load_dataPrep_ROIcoords=False,
            getTifPath=False,
            load_ref_ch_mask=False,
            endNameSegm=''
        ):
        self.segmFound = False if load_segm_data else None
        self.acd_df_found = False if load_acdc_df else None
        self.shiftsFound = False if load_shifts else None
        self.segmInfoFound = False if loadSegmInfo else None
        self.delROIsInfoFound = False if load_delROIsInfo else None
        self.bkgrDataFound = False if loadBkgrData else None
        self.bkgrROisFound = False if loadBkgrROIs else None
        self.last_tracked_i_found = False if load_last_tracked_i else None
        self.metadataFound = False if load_metadata else None
        self.dataPrep_ROIcoordsFound = False if load_dataPrep_ROIcoords else None
        self.TifPathFound = False if getTifPath else None
        self.refChMaskFound = False if load_ref_ch_mask else None
        ls = utils.listdir(self.images_path)

        if not hasattr(self, 'basename'):
            self.getBasenameAndChNames(load=False)

        for file in ls:
            filePath = os.path.join(self.images_path, file)
            filename, ext = os.path.splitext(file)
            endName = filename[len(self.basename):]

            loadMetadata = (
                load_metadata and file.endswith('metadata.csv')
                and not file.endswith('segm_metadata.csv')
            )

            if endNameSegm:
                self.endNameSegm = endNameSegm
                is_segm_file = endName == endNameSegm and ext == '.npz'
            else:
                is_segm_file = file.endswith('segm.npz')

            if load_segm_data and is_segm_file:
                self.segmFound = True
                self.segm_npz_path = filePath
                self.segm_data = np.load(filePath)['arr_0']
                squeezed_arr = np.squeeze(self.segm_data)
                if squeezed_arr.shape != self.segm_data.shape:
                    self.segm_data = squeezed_arr
                    np.savez_compressed(filePath, squeezed_arr)
            elif getTifPath and file.find(f'{self.user_ch_name}.tif')!=-1:
                self.tif_path = filePath
                self.TifPathFound = True
            elif load_acdc_df and file.endswith('acdc_output.csv'):
                self.acd_df_found = True
                acdc_df = pd.read_csv(
                      filePath, index_col=['frame_i', 'Cell_ID']
                )
                acdc_df = pd_bool_to_int(acdc_df, acdc_df_bool_cols, inplace=True)
                acdc_df = pd_int_to_bool(acdc_df, acdc_df_bool_cols)
                self.acdc_df = acdc_df
            elif load_shifts and file.endswith('align_shift.npy'):
                self.shiftsFound = True
                self.loaded_shifts = np.load(filePath)
            elif loadSegmInfo and file.endswith('segmInfo.csv'):
                self.segmInfoFound = True
                df = pd.read_csv(filePath)
                if 'filename' not in df.columns:
                    df['filename'] = self.filename
                self.segmInfo_df = df.set_index(['filename', 'frame_i'])
            elif load_delROIsInfo and file.endswith('delROIsInfo.npz'):
                self.delROIsInfoFound = True
                self.delROIsInfo_npz = np.load(filePath)
            elif loadBkgrData and file.endswith(f'{self.filename}_bkgrRoiData.npz'):
                self.bkgrDataFound = True
                self.bkgrData = np.load(filePath)
            elif loadBkgrROIs and file.endswith('dataPrep_bkgrROIs.json'):
                self.bkgrROisFound = True
                with open(filePath) as json_fp:
                    bkgROIs_states = json.load(json_fp)

                for roi_state in bkgROIs_states:
                    Y, X = self.chData_shape[-2:]
                    roi = pg.ROI(
                        [0, 0], [1, 1],
                        rotatable=False,
                        removable=False,
                        pen=pg.mkPen(color=(150,150,150)),
                        maxBounds=QRectF(QRect(0,0,X,Y))
                    )
                    roi.setState(roi_state)
                    self.bkgrROIs.append(roi)
            elif load_dataPrep_ROIcoords and file.endswith('dataPrepROIs_coords.csv'):
                df = pd.read_csv(filePath)
                if 'description' in df.columns:
                    df = df.set_index('description')
                    if 'value' in df.columns:
                        self.dataPrep_ROIcoordsFound = True
                        self.dataPrep_ROIcoords = df
            elif loadMetadata:
                self.metadataFound = True
                self.metadata_df = pd.read_csv(filePath).set_index('Description')
                self.extractMetadata()
            elif file.endswith('mask.npy') or file.endswith('mask.npz'):
                self.refChMaskFound = True
                self.refChMask = np.load(filePath)

        if load_last_tracked_i:
            self.last_tracked_i_found = True
            try:
                self.last_tracked_i = max(self.acdc_df.index.get_level_values(0))
            except AttributeError as e:
                # traceback.print_exc()
                self.last_tracked_i = None

        else:
            is_segm_file = file.endswith('segm.npz')

        if load_segm_data and not self.segmFound:
            # Try to check if there is npy segm data
            for file in ls:
                if file.endswith('segm.npy'):
                    filePath = os.path.join(self.images_path, file)
                    self.segmFound = True
                    self.segm_npz_path = filePath
                    self.segm_data = np.load(filePath)
                    break

        self.setNotFoundData()

    def segmLabels(self, frame_i):
        if self.segm_data is None:
            return None

        if self.SizeT > 1:
            lab = self.segm_data[frame_i]
        else:
            lab = self.segm_data
        return lab

    def computeSegmRegionprops(self):
        if self.segm_data is None:
            self.rp = None
            return

        if self.SizeT > 1:
            self.rp = [
                skimage.measure.regionprops(lab) for lab in self.segm_data
            ]
            self.newIDs = []
            self._IDs = []
            self.rpDict = []
            for frame_i, rp in enumerate(self.rp):
                if frame_i == 0:
                    self.newIDs.append([])
                    continue
                prevIDs = [obj.label for obj in self.regionprops(frame_i-1)]
                currentIDs = [obj.label for obj in rp]
                newIDs = [ID for ID in currentIDs if ID not in prevIDs]
                self._IDs.append(currentIDs)
                self.newIDs.append(newIDs)
                rp = cellacdc.features.add_rotational_volume_regionprops(rp)
                rpDict = {obj.label:obj for obj in rp}
                self.rpDict.append(rpDict)
        else:
            self.rp = skimage.measure.regionprops(self.segm_data)
            self._IDs = [obj.label for obj in self.rp]
            self.rpDict = {obj.label:obj for obj in self.rp}
            cellacdc.features.add_rotational_volume_regionprops(self.rp)

    def getNewIDs(self, frame_i):
        if frame_i == 0:
            return []

        if self.SizeT > 1:
            return self.newIDs[frame_i]
        else:
            return []

    def IDs(self, frame_i):
        if self.SizeT > 1:
            return self._IDs[frame_i]
        else:
            return self._IDs

    def regionprops(self, frame_i, returnDict=False):
        if self.SizeT > 1:
            if returnDict:
                return self.rpDict[frame_i]
            else:
                return self.rp[frame_i]
        else:
            if returnDict:
                return self.rpDict
            else:
                return self.rp

    def cca_df(self, frame_i):
        if self.acdc_df is None:
            return None

        cca_df = self.acdc_df.loc[frame_i][self.cca_df_colnames]
        return cca_df

    def extractMetadata(self):
        if 'SizeT' in self.metadata_df.index:
            self.SizeT = int(self.metadata_df.at['SizeT', 'values'])
        elif self.last_md_df is not None and 'SizeT' in self.last_md_df.index:
            self.SizeT = int(self.last_md_df.at['SizeT', 'values'])
        else:
            self.SizeT = 1

        if 'SizeZ' in self.metadata_df.index:
            self.SizeZ = int(self.metadata_df.at['SizeZ', 'values'])
        elif self.last_md_df is not None and 'SizeZ' in self.last_md_df.index:
            self.SizeZ = int(self.last_md_df.at['SizeZ', 'values'])
        else:
            self.SizeZ = 1

        if 'TimeIncrement' in self.metadata_df.index:
            self.TimeIncrement = float(
                self.metadata_df.at['TimeIncrement', 'values']
            )
        elif self.last_md_df is not None and 'TimeIncrement' in self.last_md_df.index:
            self.TimeIncrement = float(self.last_md_df.at['TimeIncrement', 'values'])
        else:
            self.TimeIncrement = 1

        if 'PhysicalSizeX' in self.metadata_df.index:
            self.PhysicalSizeX = float(
                self.metadata_df.at['PhysicalSizeX', 'values']
            )
        elif self.last_md_df is not None and 'PhysicalSizeX' in self.last_md_df.index:
            self.PhysicalSizeX = float(self.last_md_df.at['PhysicalSizeX', 'values'])
        else:
            self.PhysicalSizeX = 1

        if 'PhysicalSizeY' in self.metadata_df.index:
            self.PhysicalSizeY = float(
                self.metadata_df.at['PhysicalSizeY', 'values']
            )
        elif self.last_md_df is not None and 'PhysicalSizeY' in self.last_md_df.index:
            self.PhysicalSizeY = float(self.last_md_df.at['PhysicalSizeY', 'values'])
        else:
            self.PhysicalSizeY = 1

        if 'PhysicalSizeZ' in self.metadata_df.index:
            self.PhysicalSizeZ = float(
                self.metadata_df.at['PhysicalSizeZ', 'values']
            )
        elif self.last_md_df is not None and 'PhysicalSizeZ' in self.last_md_df.index:
            self.PhysicalSizeZ = float(self.last_md_df.at['PhysicalSizeZ', 'values'])
        else:
            self.PhysicalSizeZ = 1

        load_last_segmSizeT = (
            self.last_md_df is not None
            and 'segmSizeT' in self.last_md_df.index
            and self.SizeT > 1
        )
        if 'segmSizeT' in self.metadata_df.index:
             self.segmSizeT = int(
                 self.metadata_df.at['segmSizeT', 'values']
             )
        elif load_last_segmSizeT:
            self.segmSizeT = int(self.last_md_df.at['segmSizeT', 'values'])
        else:
            self.segmSizeT = self.SizeT

    def setNotFoundData(self):
        if self.segmFound is not None and not self.segmFound:
            self.segm_data = None
        if self.acd_df_found is not None and not self.acd_df_found:
            self.acdc_df = None
        if self.shiftsFound is not None and not self.shiftsFound:
            self.loaded_shifts = None
        if self.segmInfoFound is not None and not self.segmInfoFound:
            self.segmInfo_df = None
        if self.delROIsInfoFound is not None and not self.delROIsInfoFound:
            self.delROIsInfo_npz = None
        if self.bkgrDataFound is not None and not self.bkgrDataFound:
            self.bkgrData = None
        if self.dataPrep_ROIcoordsFound is not None and not self.dataPrep_ROIcoordsFound:
            self.dataPrep_ROIcoords = None
        if self.last_tracked_i_found is not None and not self.last_tracked_i_found:
            self.last_tracked_i = None
        if self.TifPathFound is not None and not self.TifPathFound:
            self.tif_path = None
        if self.refChMaskFound is not None and not self.refChMaskFound:
            self.refChMask = None

        if self.metadataFound is None:
            # Loading metadata was not requested
            return

        if self.metadataFound:
            return

        if self.chData.ndim == 3:
            if len(self.chData) > 49:
                self.SizeT, self.SizeZ = len(self.chData), 1
            else:
                self.SizeT, self.SizeZ = 1, len(self.chData)
        elif self.chData.ndim == 4:
            self.SizeT, self.SizeZ = self.chData_shape[:2]
        else:
            self.SizeT, self.SizeZ = 1, 1

        self.TimeIncrement = 1.0
        self.PhysicalSizeX = 1.0
        self.PhysicalSizeY = 1.0
        self.PhysicalSizeZ = 1.0
        self.segmSizeT = self.SizeT
        self.metadata_df = None

        if self.last_md_df is None:
            # Last entered values do not exists
            return

        # Since metadata was not found use the last entries saved in temp folder
        if 'TimeIncrement' in self.last_md_df.index:
            self.TimeIncrement = float(self.last_md_df.at['TimeIncrement', 'values'])
        if 'PhysicalSizeX' in self.last_md_df.index:
            self.PhysicalSizeX = float(self.last_md_df.at['PhysicalSizeX', 'values'])
        if 'PhysicalSizeY' in self.last_md_df.index:
            self.PhysicalSizeY = float(self.last_md_df.at['PhysicalSizeY', 'values'])
        if 'PhysicalSizeZ' in self.last_md_df.index:
            self.PhysicalSizeZ = float(self.last_md_df.at['PhysicalSizeZ', 'values'])
        if 'segmSizeT' in self.last_md_df.index:
            self.segmSizeT = int(self.last_md_df.at['segmSizeT', 'values'])

    def checkMetadata_vs_shape(self):
        pass

    def buildPaths(self):
        if self.basename.endswith('_'):
            basename = self.basename
        else:
            basename = f'{self.basename}_'
        base_path = f'{self.images_path}/{basename}'
        self.slice_used_align_path = f'{base_path}slice_used_alignment.csv'
        self.slice_used_segm_path = f'{base_path}slice_segm.csv'
        self.align_npz_path = f'{base_path}{self.user_ch_name}_aligned.npz'
        self.align_old_path = f'{base_path}phc_aligned.npy'
        self.align_shifts_path = f'{base_path}align_shift.npy'
        self.segm_npz_path = f'{base_path}segm.npz'
        self.last_tracked_i_path = f'{base_path}last_tracked_i.txt'
        self.acdc_output_csv_path = f'{base_path}acdc_output.csv'
        self.segmInfo_df_csv_path = f'{base_path}segmInfo.csv'
        self.delROIs_info_path = f'{base_path}delROIsInfo.npz'
        self.dataPrepROI_coords_path = f'{base_path}dataPrepROIs_coords.csv'
        # self.dataPrepBkgrValues_path = f'{base_path}dataPrep_bkgrValues.csv'
        self.dataPrepBkgrROis_path = f'{base_path}dataPrep_bkgrROIs.json'
        self.metadata_csv_path = f'{base_path}metadata.csv'
        self.analysis_inputs_path = f'{base_path}analysis_inputs.ini'

    def setBlankSegmData(self, SizeT, SizeZ, SizeY, SizeX):
        Y, X = self.chData_shape[-2:]
        if self.segmFound is not None and not self.segmFound:
            if SizeT > 1:
                self.segm_data = np.zeros((SizeT, Y, X), int)
            else:
                self.segm_data = np.zeros((Y, X), int)

    def loadAllChannelsPaths(self):
        tif_paths = []
        npy_paths = []
        npz_paths = []
        basename = self.basename[0:-1]
        for filename in utils.listdir(self.images_path):
            file_path = os.path.join(self.images_path, filename)
            f, ext = os.path.splitext(filename)
            m = re.match(fr'{basename}.*\.tif', filename)
            if m is not None:
                tif_paths.append(file_path)
                # Search for npy fluo data
                npy = f'{f}_aligned.npy'
                npz = f'{f}_aligned.npz'
                npy_found = False
                npz_found = False
                for name in utils.listdir(self.images_path):
                    _path = os.path.join(self.images_path, name)
                    if name == npy:
                        npy_paths.append(_path)
                        npy_found = True
                    if name == npz:
                        npz_paths.append(_path)
                        npz_found = True
                if not npy_found:
                    npy_paths.append(None)
                if not npz_found:
                    npz_paths.append(None)
        self.tif_paths = tif_paths
        self.npy_paths = npy_paths
        self.npz_paths = npz_paths

    def askInputMetadata(
            self, numPos,
            ask_SizeT=False,
            ask_TimeIncrement=False,
            ask_PhysicalSizes=False,
            save=False
        ):
        font = QFont()
        font.setPixelSize(11)
        metadataWin = dialogs.QDialogMetadata(
            self.SizeT, self.SizeZ, self.TimeIncrement,
            self.PhysicalSizeZ, self.PhysicalSizeY, self.PhysicalSizeX,
            ask_SizeT, ask_TimeIncrement, ask_PhysicalSizes, numPos,
            parent=self.parent, font=font, imgDataShape=self.chData_shape,
            PosData=self, fileExt=self.ext
        )
        metadataWin.setFont(font)
        metadataWin.exec_()
        if metadataWin.cancel:
            return False

        self.SizeT = metadataWin.SizeT
        self.SizeZ = metadataWin.SizeZ
        self.loadSizeS = metadataWin.loadSizeS
        self.loadSizeT = metadataWin.loadSizeT
        self.loadSizeZ = metadataWin.loadSizeZ

        source = metadataWin if ask_TimeIncrement else self
        self.TimeIncrement = source.TimeIncrement

        source = metadataWin if ask_PhysicalSizes else self
        self.PhysicalSizeZ = source.PhysicalSizeZ
        self.PhysicalSizeY = source.PhysicalSizeY
        self.PhysicalSizeX = source.PhysicalSizeX
        if save:
            self.saveMetadata()
        return True

    def transferMetadata(self, from_PosData):
        self.SizeT = from_PosData.SizeT
        self.SizeZ = from_PosData.SizeZ
        self.PhysicalSizeZ = from_PosData.PhysicalSizeZ
        self.PhysicalSizeY = from_PosData.PhysicalSizeY
        self.PhysicalSizeX = from_PosData.PhysicalSizeX

    def saveMetadata(self):
        if self.metadata_df is None:
            self.metadata_df = pd.DataFrame({
                'SizeT': self.SizeT,
                'SizeZ': self.SizeZ,
                'TimeIncrement': self.TimeIncrement,
                'PhysicalSizeZ': self.PhysicalSizeZ,
                'PhysicalSizeY': self.PhysicalSizeY,
                'PhysicalSizeX': self.PhysicalSizeX,
                'segmSizeT': self.segmSizeT
            }, index=['values']).T
            self.metadata_df.index.name = 'Description'
        else:
            self.metadata_df.at['SizeT', 'values'] = self.SizeT
            self.metadata_df.at['SizeZ', 'values'] = self.SizeZ
            self.metadata_df.at['TimeIncrement', 'values'] = self.TimeIncrement
            self.metadata_df.at['PhysicalSizeZ', 'values'] = self.PhysicalSizeZ
            self.metadata_df.at['PhysicalSizeY', 'values'] = self.PhysicalSizeY
            self.metadata_df.at['PhysicalSizeX', 'values'] = self.PhysicalSizeX
            self.metadata_df.at['segmSizeT', 'values'] = self.segmSizeT
        try:
            self.metadata_df.to_csv(self.metadata_csv_path)
        except PermissionError:
            msg = QMessageBox()
            warn_cca = msg.critical(
                self.parent, 'Permission denied',
                f'The below file is open in another app (Excel maybe?).\n\n'
                f'{self.metadata_csv_path}\n\n'
                'Close file and then press "Ok".',
                msg.Ok
            )
            self.metadata_df.to_csv(self.metadata_csv_path)

        self.saveLastEntriesMetadata()

    def validRuns(self):
        if not os.path.exists(self.spotmaxOutPath):
            return []

        scanner = expFolderScanner()
        run_nums = scanner.runNumbers(self.spotmaxOutPath)
        valid_run_nums = [
            run for run in run_nums
            if scanner.analyseRunNumber(self.spotmaxOutPath, run)[0]
        ]
        return valid_run_nums

    def h5_files(self, run):
        if not os.path.exists(self.spotmaxOutPath):
            return []

        orig_data_h5_filename = f'{run}_0_Orig_data'
        ellip_test_h5_filename = f'{run}_1_ellip_test_data'
        p_test_h5_filename = f'{run}_2_p-_test_data'
        p_ellip_test_h5_filename = f'{run}_3_p-_ellip_test_data'
        spotSize_h5_filename = f'{run}_4_spotFIT_data'
        h5_files = []
        for file in utils.listdir(self.spotmaxOutPath):
            _, ext = os.path.splitext(file)
            if ext != '.h5':
                continue
            if file.find(orig_data_h5_filename) != -1:
                h5_files.append(file)
            elif file.find(ellip_test_h5_filename) != -1:
                h5_files.append(file)
            elif file.find(p_test_h5_filename) != -1:
                h5_files.append(file)
            elif file.find(p_ellip_test_h5_filename) != -1:
                h5_files.append(file)
            elif file.find(spotSize_h5_filename) != -1:
                h5_files.append(file)
        return natsorted(h5_files)

    def skeletonize(self, dataToSkel):
        if self.SizeT > 1:
            self.skelCoords = []
            for data in dataToSkel:
                skelCoords = core.skeletonize(data, is_zstack=self.SizeZ>1)
                self.skelCoords.append(skelCoords)
        else:
            skelCoords = core.skeletonize(dataToSkel, is_zstack=self.SizeZ>1)
            self.skelCoords = skelCoords

    def contours(self, dataToCont):
        if self.SizeT > 1:
            self.contCoords = []
            self.contScatterCoords = []
            for data in dataToCont:
                contCoords = core.findContours(data, is_zstack=self.SizeZ>1)
                self.contCoords.append(contCoords)
                contScatterCoords = self.scatterContCoords(contCoords)
                self.contScatterCoords.append(contScatterCoords)
        else:
            contCoords = core.findContours(dataToCont, is_zstack=self.SizeZ>1)
            self.contCoords = contCoords
            self.contScatterCoords = self.scatterContCoords(contCoords)

    def scatterContCoords(self, contCoords):
        contScatterCoords = {}
        for z, allObjContours in contCoords.items():
            xx_cont = []
            yy_cont = []
            for objID, contours_li in allObjContours.items():
                for cont in contours_li:
                    xx_cont.extend(cont[:,0])
                    yy_cont.extend(cont[:,1])
            contScatterCoords[z] = (np.array(xx_cont), np.array(yy_cont))
        return contScatterCoords

    def intToBoolean(self, acdc_df):
        colsToCast = ['is_cell_dead', 'is_cell_excluded']
        for col in colsToCast:
            acdc_df[col] = acdc_df[col] > 0
        return acdc_df

    def criticalExtNotValid(self):
        err_title = f'File extension {self.ext} not valid.'
        err_msg = (
            f'The requested file {self.relPath}\n'
            'has an invalid extension.\n\n'
            'Valid extensions are .tif, .tiff, .npy or .npz'
        )
        if self.parent is None:
            print('-------------------------')
            print(err_msg)
            print('-------------------------')
            raise FileNotFoundError(err_title)
        else:
            print('-------------------------')
            print(err_msg)
            print('-------------------------')
            msg = QMessageBox()
            msg.critical(self.parent, err_title, err_msg, msg.Ok)
            return None

def get_relpath(path, depth=4):
    path_parts = os.path.normpath(path).split(os.sep)
    if len(path_parts) <= depth:
        return path
    relpath = os.path.join('', *path_parts[-depth:])
    return relpath

def is_part_of_path(path, relative_path):
    path = to_system_path(path)
    relative_path = to_system_path(path)
    return path.endswith(relative_path)

def to_system_path(path):
    path = path.replace('\\', f'{os.sep}')
    path = path.replace('/', f'{os.sep}')

    return path

def get_abspath(path, src_folderpath=''):
    if not path:
        return ''
    
    path = to_system_path(path)
    path = os.path.expanduser(path)
    path = os.path.normpath(path)

    if path == '.':
        path = src_folderpath
    
    if os.path.isabs(path):
        return path
    
    # UNIX full paths starts with '/'. Check if user forgot that
    unix_path = f'/{path}'
    if os.path.isabs(unix_path) and os.path.exists(unix_path):
        return unix_path
    
    path_parts = os.path.normpath(path).split(os.sep)
    
    # Check if relative to input src_folderpath
    if src_folderpath:
        abs_path = os.path.abspath(os.path.join(src_folderpath, *path_parts))
        if os.path.exists(abs_path):
            return abs_path
    
    # Check if relative to work directory
    cwd_path = os.getcwd()
    path = path.lstrip('.')
    abs_path = os.path.join(cwd_path, *path_parts)

    return abs_path

def get_basename_and_ch_names(images_path):
    ls = utils.listdir(images_path)
    channelNameUtil = channelName(load=False)
    channel_names, _ = channelNameUtil.getChannels(ls, images_path)
    basename = channelNameUtil.basename
    return basename, channel_names

def _get_user_input_cli(
        question_text: str, options: Iterable[str]=None, 
        default_option: str='', info_txt=None, logger=print, dtype=None,
        format_vertical=False
    ):
    if info_txt is not None:
        logger(info_txt)
        logger('')
    
    if dtype == int or dtype == 'int' or dtype == 'uint':
        input_text = f'{question_text}: '
        while True:
            try:
                answer = input(input_text)
                if answer.lower() == 'q':
                    # User requested to exit
                    return
                integer = int(answer)
                if dtype == 'uint' and integer<1:
                    raise TypeError 
            except Exception as e:
                logger('Not a valid answer. Insert an integer or type "q" to exit.')
                logger('')
                continue
            
            return integer
    
    is_yes_no = False
    if options is None:
        options = ['yes', 'no']
        is_yes_no = True
    options_txt = []
    options_nums = []
    for i, option in enumerate(options):
        if is_yes_no:
            choice = option[0]
        else:
            choice = i+1
        if option == default_option:
            options_nums.append(f'[{choice}]')
        else:
            options_nums.append(f'{choice}')
        if not is_yes_no:
            options_txt.append(f'{i+1}) {option}.')   
    options_nums = '/'.join(options_nums)
    if options_txt:
        options_txt.append('q) Quit.')
        if format_vertical:
            options_txt = [f'    {option}' for option in options_txt]
            options_txt = '\n'.join(options_txt)
            options_txt = f'\n\n{options_txt}'
            options_nums = f'\n\n({options_nums})'
        else:
            options_txt = ' '.join(options_txt)
            options_txt = f' {options_txt}'
            options_nums = f' ({options_nums})'
        input_text = f'{question_text}:{options_txt}{options_nums}?: '
    else:
        input_text = f'{question_text} ({options_nums})?: '
    
    while True:
        try:
            answer = input(input_text)
            if not answer:
                if default_option:
                    # User selected default option (with "Enter")
                    return default_option
                else:
                    # There is no default option. Force user to choose or exit
                    raise ValueError
            
            if answer.lower() == 'q':
                # User requested to exit
                return
            try:
                if not is_yes_no:
                    # Try to get the selected option
                    idx = int(answer)-1
                    selected_option = options[idx]
                elif answer.lower() == 'y' or answer.lower() == 'n':
                    is_yes = answer.lower() == 'y'
                    selected_option = 'Yes' if is_yes else 'No'
                else:
                    raise Exception
            except Exception as e:
                # User entered a value that does not correspond to an index
                logger(
                    f'"{answer}" is not a valid answer. '
                    'Try again or type "q" to exit.'
                )
                logger('')
                continue
        except ValueError:
            # Entered value cannot be parsed. Try again
            logger('Not a valid answer. Try again or type "q" to exit.')
            logger('')
        else:
            break
    return selected_option

def _log_forced_default(default_option, logger):
    logger('-'*100)
    logger(f'Automatically selected default option: "{default_option}"{error_up_str}')

def _raise_EOFerror(logger=print):
    logger('*'*100)
    logger(
        '[ERROR]: The terminal cannot get inputs. See Warning above. '
        'To force the default options, run SpotMAX with the "-f" flag, i.e. '
        f'`spotmax -f`.{error_up_str}'
    )
    exit()

def get_user_input(
        question_text: str, options: Iterable[str]=None, 
        default_option: str='', info_txt: str=None, qparent=None, 
        logger=print, dtype=None, format_vertical=False
    ):
    # Default choice in square brackets, choices in brackets after question
    if qparent:
        # Get user input through GUI messagebox
        pass
    else:
        # Ger user input in the cli
        try:
            logger('*'*100)
            answer_txt = _get_user_input_cli(
                question_text, options, default_option=default_option, 
                info_txt=info_txt, logger=logger, dtype=dtype,
                format_vertical=format_vertical
            )
            if answer_txt is not None:
                logger(f'Selected option: "{answer_txt}"')
            else:
                logger(f'Closing SpotMAX...')
            logger('*'*100)
        except EOFError:
            answer_txt = None
            logger(info_txt)
            _raise_EOFerror(logger=logger)
    return answer_txt

def add_spots_coordinates_endname_to_configparser(
        configparser, spots_coordinates_endname: str
    ):
    section = 'File paths and channels'
    option = 'Spots coordinates table end name'
    configparser[section][option] = spots_coordinates_endname
    return configparser

def add_run_number_to_configparser(configparser, run_number: int):
    section = 'File paths and channels'
    option = 'Run number'
    configparser[section][option] = str(run_number)
    return configparser

def add_text_to_append_to_configparser(
        configparser, text_to_append: str
    ):
    section = 'File paths and channels'
    option = 'Text to append at the end of the output files'
    configparser[section][option] = text_to_append
    return configparser

def add_folders_to_analyse_to_configparser(
        configparser, folders_to_analyse: List[os.PathLike]
    ):
    section = 'File paths and channels'
    option = 'Experiment folder path(s) to analyse'
    configparser[section][option] = '\n'.join(folders_to_analyse)
    return configparser

def add_ref_ch_segm_endname_to_configparser(
        configparser, ref_ch_segm_endname: str
    ):
    section = 'File paths and channels'
    option = 'Ref. channel segmentation end name'
    configparser[section][option] = ref_ch_segm_endname
    
    section = 'Reference channel'
    option = 'Save reference channel segmentation masks'
    configparser[section][option] = 'False'
    return configparser

def add_use_default_values_to_configparser(configparser):
    section = 'Configuration'
    option = 'Use default values for missing parameters'
    configparser[section][option] = 'True'
    return configparser

def set_out_files_extension(configparser, ext):
    section = 'File paths and channels'
    option = 'File extension of the output tables'
    configparser[section][option] = ext
    return configparser

def get_existing_run_nums(folderpath, logger_func=print):
    pathScanner = expFolderScanner(
        folderpath, logger_func=logger_func
    )
    pathScanner.getExpPaths(folderpath)
    pathScanner.infoExpPaths(pathScanner.expPaths)
    analysed_run_nums = set()
    for run_num, runInfo in pathScanner.paths.items():
        for exp_path, expInfo in runInfo.items():
            if 'analysisInputs' in expInfo:
                analysed_run_nums.add(run_num)
    return analysed_run_nums

def get_run_number_from_ini_filepath(ini_filepath):
    cp = read_ini(ini_filepath)
    section = 'File paths and channels'
    option = 'Run number'
    run_number = int(cp[section][option])
    return run_number

def get_text_to_append_from_ini_filepath(ini_filepath):
    cp = read_ini(ini_filepath)
    section = 'File paths and channels'
    option = 'Text to append at the end of the output files'
    text_to_append = cp[section][option]
    return text_to_append

def remove_run_number_spotmax_out_files(
        run_number, spotmax_out_path, additional_files=None
    ):
    if additional_files is None:
        additional_files = []
    for file in utils.listdir(spotmax_out_path):
        if file.startswith(f'{run_number}_'):
            try:
                os.remove(os.path.join(spotmax_out_path, file))
            except Exception as err:
                pass
        for additional_file in additional_files:
            if file.startswith(additional_file):
                try:
                    os.remove(os.path.join(spotmax_out_path, file))
                except Exception as err:
                    pass 

def save_df_spots(
        df: pd.DataFrame, folder_path: os.PathLike, filename_no_ext: str, 
        extension: str='.h5', df_with_all_keys=None
    ):
    filename = f'{filename_no_ext}{extension}'
    filepath = os.path.join(folder_path, filename)
    if extension == '.csv':
        df.to_csv(filepath)
    else:
        save_df_spots_to_hdf(
            df, folder_path, filename, df_with_all_keys=df_with_all_keys
        )
    return filepath

def save_df_ref_ch_features(
        df_ref_ch, run_number, images_path, text_to_append=''
    ):
    pos_path = os.path.dirname(images_path)
    spotmax_out_path = os.path.join(pos_path, 'spotMAX_output')
    if not os.path.exists(spotmax_out_path):
        os.mkdir(spotmax_out_path)
    
    if text_to_append:
        if not text_to_append.startswith('_'):
            text_to_append = f'_{text_to_append}'
    
    filename = f'{run_number}_3_ref_channel_features{text_to_append}.csv'
    filepath = os.path.join(spotmax_out_path, filename)
    
    df_ref_ch.to_csv(filepath)

def load_df_ref_ch_features(filepath):
    df = pd.read_csv(filepath, index_col=['frame_i', 'Cell_ID', 'sub_obj_id'])
    return df

def save_df_spots_to_hdf(
        df: pd.DataFrame, folder_path: os.PathLike, filename: str,
        df_with_all_keys: pd.DataFrame | None
    ):    
    temp_dirpath = tempfile.mkdtemp()
    temp_filepath = os.path.join(temp_dirpath, filename)
    store_hdf = pd.HDFStore(temp_filepath, mode='w')
    levels = df.index.levels[0]
    if df_with_all_keys is not None:
        levels = df_with_all_keys.index.levels[0]

    for frame_i in levels:
        key = f'frame_{frame_i}'

        try:
            sub_df = df.xs(frame_i, level='frame_i')
            df_clean = prepare_for_hdf(sub_df)
        except KeyError:
            sub_df = df.iloc[0:0].droplevel(0)
            df_clean = sub_df

        store_hdf.put(key, df_clean, format="fixed")
        
    store_hdf.close()
    dst_filepath = os.path.join(folder_path, filename)
    shutil.move(temp_filepath, dst_filepath)
    shutil.rmtree(temp_dirpath)

def prepare_for_hdf(df):
    df = df.copy()
    
    for col in df.columns:
        dtype = df[col].dtype
        
        if pd.api.types.is_string_dtype(dtype):
            df[col] = df[col].astype(object)
            
        elif pd.api.types.is_integer_dtype(dtype):
            # nullable ints → float (safe fallback) or int if no NaN
            if df[col].isna().any():
                df[col] = df[col].astype(float)
            else:
                df[col] = df[col].astype(np.int64)
                
        elif pd.api.types.is_bool_dtype(dtype):
            df[col] = df[col].astype(object)
    
    return df

def _save_concat_dfs_to_hdf(
        dfs, keys, dst_folderpath, filename, return_concat_df=False, 
        names=get_run_number_from_ini_filepath
    ):
    temp_dirpath = tempfile.mkdtemp()
    temp_filepath = os.path.join(temp_dirpath, filename)
    store_hdf = pd.HDFStore(temp_filepath, mode='w')
    for key, df in zip(keys, dfs):
        if not isinstance(key, str):
            # Convert iterable keys to single string
            key = ';;'.join([str(k) for k in key])
            key = key.replace('\\', '/')
        
        try:
            df = acdc_load.pd_bool_and_float_to_int(df)
        except Exception as err:
            df = acdc_load.pd_bool_and_float_to_int_to_str(df)
            
        key = re.sub(r'[^a-zA-Z0-9_]', "_", key)
        df_clean = prepare_for_hdf(df)
        store_hdf.append(key, df_clean)
    store_hdf.close()
    dst_filepath = os.path.join(dst_folderpath, filename)
    shutil.move(temp_filepath, dst_filepath)
    shutil.rmtree(temp_dirpath)
    
    if not return_concat_df:
        return
    
    df_concat = pd.concat(dfs, keys=keys, names=names)
    
    return df_concat

def _save_concat_dfs_to_csv(dfs, keys, dst_folderpath, filename, names=None):
    filepath = os.path.join(dst_folderpath, filename)
    df = pd.concat(dfs, keys=keys, names=names)
    df.to_csv(filepath)
    return df

def _save_concat_dfs_to_excel(dfs, keys, dst_folderpath, filename, names=None):
    filepath = os.path.join(dst_folderpath, filename)
    df = pd.concat(dfs, keys=keys, names=names)
    df.to_excel(filepath)
    return df
    
def save_concat_dfs(
        dfs, keys, dst_folderpath, filename, ext, names=None, 
        return_concat_df=False
    ):
    if ext == '.h5':
        df = _save_concat_dfs_to_hdf(
            dfs, keys, dst_folderpath, filename, 
            return_concat_df=return_concat_df, 
            names=names
        )
    elif ext == '.csv':
        df = _save_concat_dfs_to_csv(
            dfs, keys, dst_folderpath, filename, names=names,
        )
    elif ext == '.xlsx':
        df = _save_concat_dfs_to_excel(
            dfs, keys, dst_folderpath, filename, names=names, 
        )
    return df

def save_df_agg_to_csv(df: pd.DataFrame, folder_path: os.PathLike, filename: str):
    if df is None:
        return

def save_ref_ch_mask(
        ref_ch_segm_data, 
        images_path, 
        ref_ch_endname, 
        basename, 
        run_number, 
        text_to_append='', 
        pad_width=None, 
        verbose=True, 
        logger_func=print
    ):
    if verbose:
        logger_func(f'Saving reference channel masks...')
        
    if not basename.endswith('_'):
        basename = f'{basename}_'
    ref_ch_segm_filename = (
        f'{basename}run_num{run_number}_{ref_ch_endname}_ref_ch_segm_mask'
    )
    if text_to_append:
        if not text_to_append.startswith('_'):
            text_to_append = f'_{text_to_append}'
        ref_ch_segm_filename = f'{ref_ch_segm_filename}{text_to_append}'
    
    ref_ch_segm_filename = f'{ref_ch_segm_filename}.npz'
    ref_ch_segm_filepath = os.path.join(images_path, ref_ch_segm_filename)

    if pad_width is not None:
        ref_ch_segm_data = np.pad(ref_ch_segm_data, pad_width)
        
    ref_ch_segm_data = np.squeeze(ref_ch_segm_data)      
    np.savez_compressed(ref_ch_segm_filepath, ref_ch_segm_data)
    
    if verbose:
        logger_func(
            f'Reference channel masks saved to "{ref_ch_segm_filepath}"'
        )

def save_spots_masks(
        df_spots, 
        images_path, 
        basename, 
        filename, 
        spots_ch_endname, 
        run_number, 
        sizes_for_spot_masks: dict[str, list]='',
        text_to_append='', 
        mask_shape=None, 
        verbose=True, 
        logger_func=print
    ):
    if verbose:
        logger_func(f'Saving spots masks...')
        
    if not basename.endswith('_'):
        basename = f'{basename}_'
    
    spots_ch_segm_filename = filename.replace('*rn*', f'run_num{run_number}')
    desc = f'_{spots_ch_endname}'
    spots_ch_segm_filename = spots_ch_segm_filename.replace('*desc*', desc)
    
    spots_ch_segm_filename = f'{basename}{spots_ch_segm_filename}_spots_segm_mask'
    if text_to_append:
        if not text_to_append.startswith('_'):
            text_to_append = f'_{text_to_append}'
        spots_ch_segm_filename = f'{spots_ch_segm_filename}{text_to_append}'
    
    spots_ch_segm_filename = f'{spots_ch_segm_filename}.npz'
    spots_ch_segm_filepath = os.path.join(images_path, spots_ch_segm_filename)
    
    spots_mask_data = np.zeros(mask_shape, dtype=np.uint32)
    custom_spots_masks_data = {}
    if sizes_for_spot_masks:
        custom_spots_masks_data = defaultdict(dict)
        for group, features in sizes_for_spot_masks.items():
            for feature in features:
                if group == 'custom':
                    values_text_pixel = re.findall(
                        rf'\(({float_re}), ({float_re}), ({float_re})\) pixel',
                        feature
                    )
                    values_text_fn = (
                        '_'.join(values_text_pixel[0])
                        .replace('.', 'p')
                    )
                    spot_mask_size_colname = f'custom_{values_text_fn}_pixel'
                else:
                    key = f'{group}, {feature}'
                    spot_mask_size_colname = group_feature_to_col_mapper[key]
                custom_spots_masks_data[spot_mask_size_colname] = (
                    np.zeros_like(spots_mask_data)
                )
        
    for frame_i, df_spots_frame_i in df_spots.groupby(level=0):
        df_spots_frame_i = df_spots_frame_i.loc[frame_i]
        spots_lab = spots_mask_data[frame_i]
        spots_lab = transformations.from_df_spots_objs_to_spots_lab(
            df_spots_frame_i, spots_lab.shape, spots_lab=spots_lab
        )
        spots_mask_data[frame_i] = spots_lab
        
        if not sizes_for_spot_masks:
            continue
        
        _add_custom_size_spots_masks(
            custom_spots_masks_data, frame_i, df_spots_frame_i, 
            sizes_for_spot_masks
        )                

    np.savez_compressed(spots_ch_segm_filepath, np.squeeze(spots_mask_data))
    df_spots = df_spots.drop(columns='spot_mask')
    if verbose:
        logger_func(f'Spots masks saved to "{spots_ch_segm_filepath}"')
        
    for colname, custom_spots_masks_data in custom_spots_masks_data.items():
        custom_spots_segm_filepath = spots_ch_segm_filepath.replace(
            '_spots_segm_mask', f'_spots_segm_mask_{colname}'
        )
        np.savez_compressed(
            custom_spots_segm_filepath, np.squeeze(custom_spots_masks_data)
        )
        if verbose:
            logger_func(
                f'Custom size spots masks saved to "{custom_spots_segm_filepath}"'
            ) 
    
    return df_spots

def _add_custom_size_spots_masks(
        custom_spots_masks_data, frame_i, df_spots_frame_i, 
        sizes_for_spot_masks
    ):
    for colname, spots_masks_data in custom_spots_masks_data.items():
        spots_lab = spots_masks_data[frame_i]
        
        spots_lab = transformations.from_df_spots_objs_to_spots_lab(
            df_spots_frame_i, spots_lab.shape, spots_lab=spots_lab,
            spot_mask_size_colname=colname
        ) 
        custom_spots_masks_data[colname][frame_i] = spots_lab

def addToRecentPaths(selectedPath):
    if not os.path.exists(selectedPath):
        return
    recentPaths_path = os.path.join(
        settings_path, 'recentPaths.csv'
    )
    if os.path.exists(recentPaths_path):
        df = pd.read_csv(recentPaths_path, index_col='index')
        recentPaths = df['path'].to_list()
        if 'opened_last_on' in df.columns:
            openedOn = df['opened_last_on'].to_list()
        else:
            openedOn = [np.nan]*len(recentPaths)
        if selectedPath in recentPaths:
            pop_idx = recentPaths.index(selectedPath)
            recentPaths.pop(pop_idx)
            openedOn.pop(pop_idx)
        recentPaths.insert(0, selectedPath)
        openedOn.insert(0, datetime.datetime.now())
        # Keep max 30 recent paths
        if len(recentPaths) > 30:
            recentPaths.pop(-1)
            openedOn.pop(-1)
    else:
        recentPaths = [selectedPath]
        openedOn = [datetime.datetime.now()]
    df = pd.DataFrame({
        'path': recentPaths,
        'opened_last_on': pd.Series(openedOn, dtype='datetime64[ns]')
    })
    df.index.name = 'index'
    df.to_csv(recentPaths_path)

def getInfoPosStatus(expPaths):
    infoPaths = {}
    for exp_path, posFoldernames in expPaths.items():
        posFoldersInfo = {}
        for pos in posFoldernames:
            pos_path = os.path.join(exp_path, pos)
            status = acdc_myutils.get_pos_status(pos_path)
            posFoldersInfo[pos] = status
        infoPaths[exp_path] = posFoldersInfo
    return infoPaths

def is_pos_folder(path):
    foldername = os.path.basename(path)
    return (
        re.search(r'Position_(\d+)$', foldername) is not None
        and os.path.exists(os.path.join(path, 'Images'))
    )

def is_images_folder(path):
    foldername = os.path.basename(path)
    return (
        os.path.isdir(path) and foldername == 'Images'
    )

class PathScanner:
    def __init__(self, guiWin, progressWin):
        self.guiWin = guiWin
        self.progressWin = progressWin
    
    def start(self, selectedPath):
        worker = qtworkers.pathScannerWorker(selectedPath)
        worker.signals.finished.connect(self.pathScannerWorkerFinished)
        worker.signals.progress.connect(self.guiWin.workerProgress)
        worker.signals.initProgressBar.connect(self.guiWin.workerInitProgressbar)
        worker.signals.progressBar.connect(self.guiWin.workerUpdateProgressbar)
        worker.signals.critical.connect(self.guiWin.workerCritical)
        self.guiWin.threadPool.start(worker)
        self._wait()
    
    def _wait(self):
        self._loop = QEventLoop()
        self._loop.exec_()
    
    def pathScannerWorkerFinished(self, pathScanner: expFolderScanner):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        pathScanner.input(
            app=self.guiWin.app, parent=self.guiWin, 
            singleTopLevelItemSelection=False
        )
        self.images_paths = pathScanner.selectedPaths
        self._loop.exit()

def browse_last_used_ini_folderpath():
    with open(last_used_ini_text_filepath, 'r') as txt:
        params_path = txt.read()
    acdc_myutils.showInExplorer(os.path.dirname(params_path))

def save_last_used_ini_filepath(params_path: os.PathLike):
    with open(last_used_ini_text_filepath, 'w') as txt:
        txt.write(params_path)

def download_unet_models():
    from . import unet2D_checkpoint_path, unet3D_checkpoint_path
    unet2D_url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/4dxeHSLDfAbC8dA/download/unet_best.pth'
    unet2D_filename = 'unet_best.pth'
    unet2D_filesize_bytes = 69_124_096
    
    unet3D_url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/eoeFcgsAMDsgTgw/download/best_checkpoint.pytorch'
    unet3D_filename = 'best_checkpoint.pytorch'
    unet3D_filesize_bytes = 49_049_600
    
    unet2D_filepath = os.path.join(unet2D_checkpoint_path, unet2D_filename)
    if not os.path.exists(unet2D_filepath):
        print('Downloading SpotMAX U-Net 2D model...')
        os.makedirs(unet2D_checkpoint_path, exist_ok=True)
        acdc_myutils.download_url(
            unet2D_url, unet2D_filepath, 
            file_size=unet2D_filesize_bytes, 
            desc='SpotMAX U-Net 2D', 
            verbose=False
        )
    
    unet3D_filepath = os.path.join(unet3D_checkpoint_path, unet3D_filename)
    if not os.path.exists(unet3D_filepath):
        print('Downloading SpotMAX U-Net 3D model...')
        os.makedirs(unet3D_checkpoint_path, exist_ok=True)
        acdc_myutils.download_url(
            unet3D_url, unet3D_filepath, 
            file_size=unet3D_filesize_bytes, 
            desc='SpotMAX U-Net 3D', 
            verbose=False
        )


def _raise_missing_param_ini(missing_option, section):
    raise KeyError(
        'The following parameter is missing from the INI configuration file: '
        f'`{missing_option}` (section `[{section}]`). '
        'You can force using default value by setting '
        '`Use default values for missing parameters = True` in the '
        'INI file.'
    )

def nnet_get_defaults_missing_param(section_params, model_module, method):
    missing_params = []
    init_args, segment_args = acdc_myutils.getModelArgSpec(model_module)    
    if method == 'init':
        argspecs = init_args
    elif method == 'segment':
        argspecs = segment_args
        
    for argWidget in argspecs:
        try:
            not_a_param = argWidget.type().not_a_param
            continue
        except Exception as err:
            pass
        option = section_params.get(argWidget.name)
        if option is not None:
            continue
        missing_params.append(
            (argWidget.name, argWidget.default, argWidget.name)
        )
    return missing_params

def nnet_params_from_ini_params(
        ini_params, sections, model_module, use_default_for_missing=False
    ):
    argSpecs = acdc_myutils.getModelArgSpec(model_module)
    params = {section.split('.')[1]: {} for section in sections}
    
    for s, section in enumerate(sections):
        if section not in ini_params:
            continue
        
        key = section.split('.')[1]
        section_params = ini_params[section]
        for argWidget in argSpecs[s]:
            try:
                not_a_param = argWidget.type().not_a_param
                continue
            except Exception as err:
                pass
            option = section_params.get(argWidget.name)
            if option is None:
                # Try older name
                older_name = LEGACY_MODEL_KWARGS.get(argWidget.name)
                if older_name is not None:
                    option = section_params.get(older_name) 
            
            if option is None:
                if use_default_for_missing:
                    continue
                else:
                    _raise_missing_param_ini(argWidget.name, section)
            value = option['loadedVal']
            if not isinstance(argWidget.default, str):
                try:
                    value = utils.to_dtype(value, type(argWidget.default))
                except Exception as err:
                    value = argWidget.default
            params[key][argWidget.name] = value
    return params

def load_df_agg_from_df_spots_filename(
        spotmax_output_folderpath, df_spots_filename
    ):
    filename_no_ext, _ = os.path.splitext(df_spots_filename)
    aggr_filename = f'{filename_no_ext}_aggregated.csv'
    aggr_filepath = os.path.join(spotmax_output_folderpath, aggr_filename)
    df_aggr = pd.read_csv(aggr_filepath, index_col=['frame_i', 'Cell_ID'])
    return df_aggr

def df_spots_filename_parts(df_spots_filename):
    found = re.match(df_spots_filename_parts_pattern, df_spots_filename)
    if not found:
        return ('', '', '', '', '')
    return found.groups()

def get_analysis_params_filepath_from_df_spots_filename(
        spotmax_output_folderpath, df_spots_filename
    ):
    parts = df_spots_filename_parts(df_spots_filename)
    run_num, df_id, df_text, desc, ext = parts
    if df_id.startswith('4_'):
        run_num = f'{run_num}_4'
    analysis_params_filename = f'{run_num}_analysis_parameters{desc}.ini'
    filepath = os.path.join(spotmax_output_folderpath, analysis_params_filename)
    return filepath

def load_analysis_params_from_df_spots_filename(
        spotmax_output_folderpath, df_spots_filename
    ):
    filepath = get_analysis_params_filepath_from_df_spots_filename(
        spotmax_output_folderpath, df_spots_filename
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f'The parameters file linked to analysis file "{df_spots_filename}"'
            f' was not found. Requested parameters file path: "{filepath}"'
        )
    cp = config.ConfigParser()
    cp.read(filepath, encoding="utf-8")
    return cp

def get_temp_ini_filepath():
    temp_dirpath = tempfile.mkdtemp()
    ini_filename = get_new_ini_filename()
    ini_filepath = os.path.join(temp_dirpath, ini_filename)
    return ini_filepath, temp_dirpath

def get_temp_csv_filepath():
    temp_dirpath = tempfile.mkdtemp()
    now = datetime.datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    csv_filename = f'{now}_spotmax_csv_tempfile.csv'
    return os.path.join(temp_dirpath, csv_filename)

def get_new_ini_filename(text_to_add=''):
    now = datetime.datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    if text_to_add:
        if not text_to_add.startswith('_'):
            text_to_add = f'_{text_to_add}'
            
    ini_filename = f'{now}_spotmax_analysis_parameters{text_to_add}.ini'
    return ini_filename

def get_ini_filepath_appdata(text_to_add=''):
    ini_filename = get_new_ini_filename(text_to_add=text_to_add)
    return os.path.join(spotmax_appdata_path, ini_filename)

def get_analysis_params_filepath_from_run(spotmax_out_path, run_num):
    for file in utils.listdir(spotmax_out_path):
        if file.startswith(f'{run_num}_analysis_parameters'):
            return os.path.join(spotmax_out_path, file)

def read_ini(ini_filepath):
    configPars = config.ConfigParser()
    configPars.read(ini_filepath)
    return configPars

def write_to_ini(configparser, ini_filepath):
    with open(ini_filepath, 'w', encoding="utf-8") as file:
        configparser.write(file)

def get_filepath_from_channel_name(
        images_path, channel_name, raise_on_duplicates=True
    ):
    valid_patterns = (
        f'{channel_name}_aligned.npz', 
        f'{channel_name}_aligned.h5', 
        f'{channel_name}.tif', 
        f'{channel_name}.npz', 
        f'{channel_name}.npy', 
        channel_name,
    )
    found_files = defaultdict(list)
    for pattern in valid_patterns:
        for file in utils.listdir(images_path):
            if file.endswith(pattern):
                found_files[pattern].append(file)
    
    if not found_files:
        return ''
    
    first_pattern_found = list(found_files.keys())[0]
    found_files = found_files[first_pattern_found]
    
    if len(found_files) > 1:
        closest_files = []
        basename = os.path.commonprefix(found_files)
        endname = first_pattern_found
        closest_pattern = f'{basename}{endname}'
        for file in found_files:
            file_noext, _ = os.path.splitext(file)
            if file_noext == closest_pattern:
                closest_files.append(file)
            
            if file == closest_pattern:
                closest_files.append(file)    
                  
        found_files = closest_files
        
    if len(found_files) > 1 and raise_on_duplicates:
        raise MultipleFilesFound(
            f'Multiple files ending with the requested pattern found: {found_files}'
        )
    
    return os.path.join(images_path, found_files[0])

def run_num_and_appended_text_from_segm_path(
        segm_path, channel_name: str,
        channel_type: Literal['spots', 'ref_ch']='spots', 
    ):
    pattern = fr'_run_num(\d+)_{channel_name}_{channel_type}_segm_mask(_?.*).npz'
    filename = os.path.basename(segm_path)
    found = re.findall(pattern, filename)
    if not found:
        return
    
    found = found[0]
    if len(found) < 2:
        return
    
    run_num, appended_text = found
    return run_num, appended_text

def get_ref_channel_dfs_features_filename(
        run_num: int, appended_text: str, spotmax_out_folder: os.PathLike
    ):
    files = utils.listdir(spotmax_out_folder)
    file_to_match = f'{run_num}_3_ref_channel_features{appended_text}.csv'
    for file in files:
        if file == file_to_match:
            return file, True
    
    return file_to_match, False