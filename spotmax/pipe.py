from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd

import skimage.measure
import skimage.filters

from . import GUI_INSTALLED
if GUI_INSTALLED:
    from cellacdc.plot import imshow

from . import filters
from . import transformations
from . import printl
from . import ZYX_LOCAL_COLS, ZYX_LOCAL_EXPANDED_COLS, ZYX_GLOBAL_COLS
from . import ZYX_FIT_COLS
from . import features
from . import utils
from . import core

def preprocess_image(
        image, 
        lab=None, 
        do_remove_hot_pixels=False,
        gauss_sigma=0.0,
        use_gpu=True, 
        return_lab=False,
        do_sharpen=False,
        spots_zyx_radii_pxl=None,
        logger_func=print
    ):
    _, image = transformations.reshape_lab_image_to_3D(lab, image)
        
    if do_remove_hot_pixels:
        image = filters.remove_hot_pixels(image)
    else:
        image = image
    
    if gauss_sigma != 0:
        image = filters.gaussian(
            image, gauss_sigma, use_gpu=use_gpu, logger_func=logger_func
        )
    else:
        image = image
    
    if do_sharpen and spots_zyx_radii_pxl is not None:
        image = filters.DoG_spots(
            image, spots_zyx_radii_pxl, use_gpu=use_gpu, 
            logger_func=logger_func, lab=lab
        )
    # elif gauss_sigma != 0:
    #     image = filters.gaussian(
    #         image, gauss_sigma, use_gpu=use_gpu, logger_func=logger_func
    #     )
    else:
        image = image
    
    if return_lab:
        return image, lab
    else:
        return image


def ridge_filter(
        image, 
        lab=None, 
        do_remove_hot_pixels=False, 
        ridge_sigmas=0.0,
        logger_func=print
    ):
    _, image = transformations.reshape_lab_image_to_3D(lab, image)
        
    if do_remove_hot_pixels:
        image = filters.remove_hot_pixels(image)
    else:
        image = image
    
    if ridge_sigmas:
        image = filters.ridge(image, ridge_sigmas)
    else:
        image = image
    return image

def spots_semantic_segmentation(
        image, 
        lab=None,
        gauss_sigma=0.0,
        spots_zyx_radii_pxl=None, 
        do_sharpen=False, 
        do_remove_hot_pixels=False,
        lineage_table=None,
        do_aggregate=True,
        thresh_only_inside_objs_intens=True,
        min_spot_mask_size=5,
        keep_objects_touching_lab_intact=True,
        use_gpu=False,
        logger_func=print,
        thresholding_method=None,
        keep_input_shape=True,
        nnet_model=None,
        nnet_params=None,
        nnet_input_data=None,
        return_nnet_prediction=False,
        bioimageio_model=None,
        bioimageio_params=None,
        spotiflow_model=None,
        spotiflow_params=None,
        do_preprocess=True,
        do_try_all_thresholds=True,
        return_only_segm=False,
        pre_aggregated=False,
        x_slice_idxs=None,
        raw_image=None
    ):  
    """Pipeline to perform semantic segmentation on the spots channel, 
    i.e., determine the areas where spot will be detected.

    Parameters
    ----------
    image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray
        Input 2D or 3D image.
    lab : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        Optional input segmentation image with the masks of the objects, i.e. 
        single cells. Spots will be detected only inside each object. If None, 
        detection will be performed on the entire image. Default is None. 
    gauss_sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel. The standard deviations of 
        the Gaussian filter are given for each axis as a sequence, or as a 
        single number, in which case it is equal for all axes. If 0, no 
        gaussian filter is applied. Default is 0.0
    spots_zyx_radii_pxl : (z, y, x) sequence of floats, optional
        Rough estimation of the expected spot radii in z, y, and x direction
        in pixels. The values are used to determine the sigmas used in the 
        difference-of-gaussians filter that enhances spots-like structures. 
        If None, no filter is applied. Default is None
    do_sharpen : bool, optional
        If True and spots_zyx_radii_pxl is not None, applies a 
        difference-of-gaussians (DoG) filter before segmenting. This filter 
        enhances spots-like structures and it usually improves detection. 
        Default is False.
        For more details, see the parameter `Sharpen spots signal prior 
        detection` at the following webpage: 
        https://spotmax.readthedocs.io/en/latest/parameters/parameters_description.html#confval-Sharpen-spots-signal-prior-detection
    do_remove_hot_pixels : bool, optional
        If True, apply a grayscale morphological opening filter before 
        segmenting. Opening can remove small bright spots (i.e. “salt”, or 
        "hot pixels") and connect small dark cracks. Default is False
    lineage_table : pandas.DataFrame, optional
        Table containing parent-daughter relationships. Default is None
        For more details, see the parameter `Table with lineage info end name` 
        at the following webpage: 
        https://spotmax.readthedocs.io/en/latest/parameters/parameters_description.html#confval-Table-with-lineage-info-end-name
    do_aggregate : bool, optional
        If True, perform segmentation on all the cells at once. Default is True
    thresh_only_inside_objs_intens : bool, optional
        If True, use only the intensities from inside the segmented objects 
        (in `lab`). Default is False
    min_spot_mask_size : int, optional
        Minimum size (in pixels) of the spots masks. Masks with 
        `size < min_spot_mask_size` will be removed. Default is 5.
    keep_objects_touching_lab_intact : bool, optional
        If True, objects that are partially touching any of the segmentation 
        masks present in `lab` will be entirely kept. If False, the part of 
        these objects that is outside of the segmentation masks will be 
        removed. Default is True
    use_gpu : bool, optional
        If True, some steps will run on the GPU, potentially speeding up the 
        computation. Default is False
    logger_func : callable, optional
        Function used to print or log process information. Default is print
    thresholding_method : {'threshol_li', 'threshold_isodata', 'threshold_otsu',
        'threshold_minimum', 'threshold_triangle', 'threshold_mean',
        'threshold_yen'} or callable, optional
        Thresholding method used to obtain semantic segmentation masks of the 
        spots. If None and do_try_all_thresholds is True, the result of every 
        threshold method available is returned. Default is None
    keep_input_shape : bool, optional
        If True, return segmentation array with the same shape of the 
        input image. If False, output shape will depend on whether do_aggregate
        is True or False. Default is True
    nnet_model : Cell-ACDC segmentation model class, optional
        If not None, the output will include the key 'neural_network' with the 
        result of the segmentation using the neural network model. 
        Default is None
    nnet_params : dict with 'segment' key, optional
        Parameters used in the segment method of the nnet_model. Default is None
    nnet_input_data : numpy.ndarray or sequence of arrays, optional
        If not None, run the neural network on this data and not on the 
        pre-processed input image. Default is None
    return_nnet_prediction : bool, optional
        If True, include the key 'neural_network_prediciton' in the returned 
        dictionary. Default is False
    bioimageio_model : Cell-ACDC implementation of any BioImage.IO model, optional
        If not None, the output will include the key 'bioimageio_model' with the 
        result of the segmentation using the BioImage.IO model. 
        Default is None
    bioimageio_params : dict with 'segment' key, optional
        Parameters used in the segment method of the bioimageio_model. 
        Default is None
    spotiflow_model : Cell-ACDC implementation of Spotiflow, optional
        If not None, the output will include the key 'spotiflow' with the 
        result of the segmentation using Spotiflow. Default is None
    spotiflow_params : dict with 'segment' key, optional
        Parameters used in the `segment` method of the spotiflow_model. 
        Default is None
    do_preprocess : bool, optional
        If True, pre-process image before segmentation using the filters 
        'remove hot pixels', 'gaussian', and 'sharpen spots' (if requested). 
        Default is True
    do_try_all_thresholds : bool, optional
        If True and thresholding_method is not None, the result of every 
        threshold method available is returned. Default is True
    return_only_segm : bool, optional
        If True, return only the result of the segmentation as numpy.ndarray 
        with the same shape as the input image. Default is False
    pre_aggregated : bool, optional
        If True and do_aggregate is True, run segmentation on the entire input 
        image without aggregating segmented objects. Default is False
    x_slice_idxs : list, optional
        List of indices along the x-axis of the input image (last axis) where 
        each object in `lab` ends. This is useful if the input image is 
        "aggregated" meaning that the objects in `lab` are concatenated 
        along the x-axis. Default is None
    raw_image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray, optional
        If not None, neural network and BioImage.IO models will segment 
        the raw image. Default is None

    Returns
    -------
    result : dict or numpy.ndarray
        If `return_only_segm` is True, the first value of the output will be 
        the numpy.ndarray with the segmentation result. Note that, in this case,
        if `nnet_model` is not None and `nnet_params['save_prediction_map']` 
        is `True` the second value of the output will be the prediction 
        map from the output of running `nnet_model`. 
        
        If `thresholding_method` is None and do_try_all_thresholds is True, 
        the output will be a dictionary with keys {'threshol_li', 
        'threshold_isodata', 'threshold_otsu', 'threshold_minimum', 
        'threshold_triangle', 'threshold_mean', 'threshold_yen'} and values 
        the result of each thresholding method. 
        
        If `thresholding_method` is not None, the output will be a dictionary 
        with one key {'custom'} and the result of applying the requested 
        thresholding_method. 
        
        If `nnet_model` is not None, the output dictionary will include the 
        'neural_network' key with value the result of running the `nnet_model`
        requested. 
        
        If `return_nnet_prediction` is True and `nnet_model` is not None, the 
        output dictionary will include the 'neural_network_prediciton' key 
        with value the prediction map output of `nnet_model`. 
        
        If bioimageio_model is not None, the output dictionary will include the 
        'bioimageio_model' key with value the result of running the bioimageio_model
        requested. 
        
        If spotiflow_model is not None, the output dictionary will include the 
        'spotiflow' key with value the result of running the spotiflow_model
        requested. 
        
        The output dictionary will also include the key 'input_image' with value 
        the pre-processed image. 
    """   
    lab, image = transformations.reshape_lab_image_to_3D(lab, image)
    
    if raw_image is None:
        raw_image = image.copy()
    else:
        _, raw_image = transformations.reshape_lab_image_to_3D(lab, raw_image)
        
    if do_preprocess:
        image, lab = preprocess_image(
            image, 
            lab=lab, 
            do_remove_hot_pixels=do_remove_hot_pixels, 
            gauss_sigma=gauss_sigma,
            use_gpu=use_gpu, 
            return_lab=True,
            do_sharpen=do_sharpen,
            spots_zyx_radii_pxl=spots_zyx_radii_pxl,
            logger_func=logger_func
        )

    if lab is None:
        lab = np.ones(image.shape, dtype=np.uint8)
    
    if not np.any(lab):
        result = {
            'input_image': image,
            'Segmentation_data_is_empty': np.zeros(image.shape, dtype=np.uint8)
        }
        return result
    
    if nnet_model is not None and nnet_input_data is None:
        # Use raw image as input to neural network if nnet_input_data is None
        nnet_input_data = raw_image
    
    zyx_tolerance = transformations.get_expand_obj_delta_tolerance(
        spots_zyx_radii_pxl
    )
    if do_aggregate:
        result = filters.global_semantic_segmentation(
            image, lab, 
            lineage_table=lineage_table, 
            zyx_tolerance=zyx_tolerance, 
            threshold_func=thresholding_method, 
            logger_func=logger_func, 
            return_image=True,
            keep_input_shape=keep_input_shape,
            keep_objects_touching_lab_intact=keep_objects_touching_lab_intact,
            thresh_only_inside_objs_intens=thresh_only_inside_objs_intens,
            nnet_model=nnet_model,
            nnet_params=nnet_params,
            nnet_input_data=nnet_input_data,
            return_nnet_prediction=return_nnet_prediction,
            do_try_all_thresholds=do_try_all_thresholds,
            return_only_output_mask=return_only_segm,
            pre_aggregated=pre_aggregated,
            x_slice_idxs=x_slice_idxs,
            bioimageio_model=bioimageio_model,
            bioimageio_params=bioimageio_params,
            bioimageio_input_image=raw_image,
            spotiflow_model=spotiflow_model,
            spotiflow_params=spotiflow_params,
            spotiflow_input_image=raw_image,
            min_mask_size=min_spot_mask_size
        )
    else:
        result = filters.local_semantic_segmentation(
            image, lab, 
            threshold_func=thresholding_method, 
            zyx_tolerance=zyx_tolerance, 
            lineage_table=lineage_table, 
            return_image=True,
            keep_objects_touching_lab_intact=keep_objects_touching_lab_intact,
            nnet_model=nnet_model, 
            nnet_params=nnet_params,
            nnet_input_data=nnet_input_data,
            return_nnet_prediction=return_nnet_prediction,
            do_try_all_thresholds=do_try_all_thresholds,
            return_only_output_mask=return_only_segm,
            do_max_proj=True,
            bioimageio_model=bioimageio_model,
            bioimageio_params=bioimageio_params,
            bioimageio_input_image=raw_image,
            spotiflow_model=spotiflow_model,
            spotiflow_params=spotiflow_params,
            spotiflow_input_image=raw_image,
            min_mask_size=min_spot_mask_size
        )
    
    return result

def reference_channel_semantic_segm(
        image, 
        lab=None,
        gauss_sigma=0.0,
        keep_only_largest_obj=False,
        keep_objects_touching_lab_intact=False,
        do_remove_hot_pixels=False,
        lineage_table=None,
        do_aggregate=True,
        use_gpu=False,
        logger_func=print,
        thresholding_method=None,
        ridge_filter_sigmas=0,
        keep_input_shape=True,
        do_preprocess=True,
        return_only_segm=False,
        do_try_all_thresholds=True,
        bioimageio_model=None,
        bioimageio_params=None,
        raw_image=None,
        pre_aggregated=False,
        x_slice_idxs=None,
        show_progress=False
    ):    
    """Pipeline to segment the reference channel.

    Parameters
    ----------
    image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray
        Input 2D or 3D image.
    lab : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints
        Optional input segmentation image with the masks of the objects, i.e. 
        single cells. Default is None. 
    gauss_sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel. The standard deviations of 
        the Gaussian filter are given for each axis as a sequence, or as a 
        single number, in which case it is equal for all axes. If 0, no 
        gaussian filter is applied. Default is 0.0
    keep_only_largest_obj : bool, optional
        If True, keep only the largest object (determined by connected component
        labelling) per segmented object in lab. Default is False
    keep_objects_touching_lab_intact : bool, optional
        If `True`, the segmented objects that are partially outside of the 
        objects in `lab` are kept intact. If `False`, the external part 
        is erased. Default is False
    do_remove_hot_pixels : bool, optional
        If True, apply a grayscale morphological opening filter before 
        segmenting. Opening can remove small bright spots (i.e. “salt”, or 
        "hot pixels") and connect small dark cracks. Default is False
    lineage_table : pandas.DataFrame, optional
        Table containing parent-daughter relationships. Default is None
        For more details, see the parameter `Table with lineage info end name` 
        at the following webpage: 
        https://spotmax.readthedocs.io/en/latest/parameters/parameters_description.html#confval-Table-with-lineage-info-end-name
    do_aggregate : bool, optional
        If True, perform segmentation on all the cells at once. Default is True
    use_gpu : bool, optional
        If True, some steps will run on the GPU, potentially speeding up the 
        computation. Default is False
    logger_func : callable, optional
        Function used to print or log process information. Default is print
    thresholding_method : {'threshol_li', 'threshold_isodata', 'threshold_otsu',
        'threshold_minimum', 'threshold_triangle', 'threshold_mean',
        'threshold_yen'} or callable, optional
        Thresholding method used to obtain semantic segmentation masks of the 
        spots. If None and do_try_all_thresholds is True, the result of every 
        threshold method available is returned. Default is None
    ridge_filter_sigmas : scalar or sequence of scalars, optional
        Sigmas used as scales of filter. If not 0, filter the image with the 
        Sato tubeness filter. This filter can be used to detect continuous 
        ridges, e.g. mitochondrial network. Default is 0
    keep_input_shape : bool, optional
        If True, return segmentation array with the same shape of the 
        input image. If False, output shape will depend on whether do_aggregate
        is True or False. Default is True
    do_preprocess : bool, optional
        If True, pre-process image before segmentation using the filters 
        'remove hot pixels', 'gaussian', and 'sharpen spots' (if requested). 
        Default is True
    do_try_all_thresholds : bool, optional
        If True and thresholding_method is not None, the result of every 
        threshold method available is returned. Default is True
    return_only_segm : bool, optional
        If True, return only the result of the segmentation as numpy.ndarray 
        with the same shape as the input image. Default is False
    bioimageio_model : Cell-ACDC implementation of any BioImage.IO model, optional
        If not None, the output will include the key 'bioimageio_model' with the 
        result of the segmentation using the BioImage.IO model. 
        Default is None
    bioimageio_params : _type_, optional
        Parameters used in the segment method of the bioimageio_model. 
        Default is None
    pre_aggregated : bool, optional
        If True and do_aggregate is True, run segmentation on the entire input 
        image without aggregating segmented objects. Default is False
    x_slice_idxs : list, optional
        List of indices along the x-axis of the input image (last axis) where 
        each object in `lab` ends. This is useful if the input image is 
        "aggregated" meaning that the objects in `lab` are concatenated 
        along the x-axis. Default is None
    raw_image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray, optional
        If not None, neural network and BioImage.IO models will segment 
        the raw image. Default is None
    show_progress : bool, optional
        If True, display progressbars. Default is False

    Returns
    -------
    result : dict or numpy.ndarray
        If return_only_segm is True, the output will the the numpy.ndarray 
        with the segmentation result. 
        
        If thresholding_method is `None` and do_try_all_thresholds is True, 
        the output will be a dictionary with keys {'threshol_li', 
        'threshold_isodata', 'threshold_otsu', 'threshold_minimum', 
        'threshold_triangle', 'threshold_mean', 'threshold_yen'} and values 
        the result of each thresholding method. 
        
        If thresholding_method is not `None`, the output will be a dictionary 
        with key {'custom'} and value the result of applying the requested 
        thresholding_method. 
        
        If bioimageio_model is not `None`, the output dictionary will include the 
        'bioimageio_model' key with value the result of running the bioimageio_model
        requested. 
        
        The output dictionary will also include the key 'input_image' with value 
        the pre-processed image. 
    """    
    if raw_image is None:
        raw_image = image.copy()
        
    if do_preprocess:
        if show_progress:
            logger_func('Pre-processing image...')
        image, lab = preprocess_image(
            image, 
            lab=lab, 
            do_remove_hot_pixels=do_remove_hot_pixels, 
            gauss_sigma=gauss_sigma,
            use_gpu=use_gpu, 
            logger_func=logger_func,
            return_lab=True
        )
    
    if not np.any(lab):
        empty_segm = np.zeros(image.shape, dtype=np.uint8)
        if thresholding_method is not None or return_only_segm:
            return empty_segm
        else:
            result = {
                'input_image': image,
                'Segmentation_data_is_empty': empty_segm
            }
            return result
    
    if do_aggregate:
        result = filters.global_semantic_segmentation(
            image, lab, lineage_table=lineage_table, 
            threshold_func=thresholding_method, 
            logger_func=logger_func, return_image=True,
            keep_input_shape=keep_input_shape,
            keep_objects_touching_lab_intact=keep_objects_touching_lab_intact,
            ridge_filter_sigmas=ridge_filter_sigmas,
            return_only_output_mask=return_only_segm,
            do_try_all_thresholds=do_try_all_thresholds,
            pre_aggregated=pre_aggregated,
            x_slice_idxs=x_slice_idxs,
            bioimageio_model=bioimageio_model,
            bioimageio_params=bioimageio_params,
            bioimageio_input_image=raw_image
        )
    else:
        result = filters.local_semantic_segmentation(
            image, lab, 
            threshold_func=thresholding_method, 
            lineage_table=lineage_table, 
            return_image=True,
            do_max_proj=False, 
            keep_objects_touching_lab_intact=keep_objects_touching_lab_intact,
            ridge_filter_sigmas=ridge_filter_sigmas,
            return_only_output_mask=return_only_segm,
            do_try_all_thresholds=do_try_all_thresholds,
            bioimageio_model=bioimageio_model,
            bioimageio_params=bioimageio_params,
            bioimageio_input_image=raw_image
        )
    
    if not keep_only_largest_obj:
        return result
    
    if not np.any(lab):
        return result
    
    if not return_only_segm:
        input_image = result.pop('input_image')
        result = {
            key:filters.filter_largest_sub_obj_per_obj(img, lab) 
            for key, img in result.items()
        }
        result = {**{'input_image': input_image}, **result}
    else:
        result = filters.filter_largest_sub_obj_per_obj(result, lab)
    
    return result

def reference_channel_quantify(
        ref_ch_segm,
        ref_ch_img,
        lab=None, 
        lab_rp=None,
        calc_rp=True,
        filtering_features_thresholds=None,
        df_agg=None,
        frame_i=0, 
        vox_to_um3=None,
        logger_func=print,
        verbose=True
    ):
    """Calculate reference channel features and filter valid objects

    Parameters
    ----------
    ref_ch_segm : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray of ints
        Segmentation mask of the reference channel
    ref_ch_img : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray
        Input reference channel image.
    lab : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray of ints, optional
        Optional input segmentation image with the masks of the objects, i.e. 
        single cells. Default is None. Default is None
    lab_rp : list of skimage.measure.RegionProperties, optional
        If not None, list of properties of objects in `lab` as returned by 
        skimage.measure.regionprops(lab). If None, this will be computed 
        with `skimage.measure.regionprops(lab)`. Default is None
    calc_rp : bool, optional
        If True, calculate additional regionprops using `skimage.measure.regionprops` and `features.calc_additional_regionprops`. 
        Default is True
    filtering_features_thresholds : dict of {'feature_name': (min_value, max_value)}, optional
        Features and their maximum and minimum values to filter valid reference 
        channel segmented objects. 
        An object is valid when `feature_name` is greater than `min_value` and 
        lower than `max_value`. If a value is None it means there is no minimum 
        or maximum threshold. Default is None
    df_agg : pd.DataFrame, optional
        Optional input DataFrame where to insert default features. The default 
        feautures are `ref_ch_num_fragments`, `ref_ch_vol_vox`, and 
        `ref_ch_vol_um3`. Default is None
    frame_i : int, optional
        Frame index in timelapse data. Default is 0
    vox_to_um3 : float, optional
        Optional factor used to convert voxels to micrometer cubed (equivalent 
        to fL). Default is None
    logger_func : callable, optional
        Function used to print or log process information. Default is print
    verbose : bool, optional
        If True, additional information text will be printed to the terminal. 
        Default is True

    Returns
    -------
    df_agg : pandas.DataFrame
        DataFrame with index `('frame_i', 'Cell_ID')` and three default 
        columns where 'Cell_ID' is the ID of the obejcts in `lab`.
        If input `df_agg` is not None, these columns will be added to it. 
        The default feautures are `ref_ch_num_fragments`, `ref_ch_vol_vox`, and 
        `ref_ch_vol_um3`.
    df_ref_ch : pandas.DataFrame
        DataFrame with index `('frame_i', 'Cell_ID', 'sub_obj_id')` and all the
        features columns. 'Cell_ID' is the ID of the obejcts in `lab`, and
        'sub_obj_id' is the id of each reference channel sub-object per 
        single-cell. 
        If input `filtering_features_thresholds`, is not None, the objects 
        whose features are outside of the requested ranges are dropped from the
        output table.
    filtered_ref_ch_segm : pandas.DataFrame
        _description_
    """    
    if verbose and frame_i == 0:
        print('')
        logger_func('Quantifying reference channel...')
        
    if lab is None:
        lab = np.ones(ref_ch_segm.shape, dtype=np.uint8)
    
    if lab_rp is None:
        lab_rp = skimage.measure.regionprops(lab)
        
    if df_agg is None:
        df_agg = pd.DataFrame({
            'frame_i': [frame_i]*len(lab_rp),
            'Cell_ID': [obj.label for obj in lab_rp]
        })
    
    distribution_metrics_func = features.get_distribution_metrics_func()
    
    dfs_ref_ch = []
    keys = []
    sub_objs = {}
    
    desc = 'Quantifying reference channel'
    pbar = tqdm(
        total=len(lab_rp), ncols=100, desc=desc, position=3, 
        leave=False
    )
    for obj in lab_rp:
        ID = obj.label
        
        ref_ch_lab_local = ref_ch_segm[obj.slice].copy()
        ref_ch_lab_local[ref_ch_lab_local!=obj.label] = 0
        ref_ch_mask_local = ref_ch_lab_local > 0
        
        ref_ch_img_local = ref_ch_img[obj.slice]
        
        ref_ch_lab = skimage.measure.label(ref_ch_mask_local)
        ref_ch_rp = skimage.measure.regionprops(ref_ch_lab)
        
        if len(ref_ch_rp) == 0:
            continue
        
        df_ref_ch = features._init_df_ref_ch(ref_ch_rp)

        # Add num of fragments
        num_fragments = len(ref_ch_rp)
        df_agg.at[(frame_i, ID), 'ref_ch_num_fragments'] = num_fragments        
        df_ref_ch['ref_ch_num_fragments'] = num_fragments
        
        # Add volumes
        vol_voxels = np.count_nonzero(ref_ch_mask_local)
        df_agg.at[(frame_i, ID), 'ref_ch_vol_vox'] = vol_voxels
        df_ref_ch['ref_ch_vol_vox'] = vol_voxels
        if vox_to_um3 is not None:
            vol_um3 = vol_voxels*vox_to_um3
            df_agg.at[(frame_i, ID), 'ref_ch_vol_um3'] = vol_um3
            df_ref_ch['ref_ch_vol_um3'] = vol_um3
        
        # Add background metrics
        bkgr_mask = np.logical_and(~ref_ch_mask_local, obj.image)
        backgr_vals = ref_ch_img_local[bkgr_mask]
        for name, func in distribution_metrics_func.items():
            df_ref_ch.loc[:, f'background_ref_ch_{name}_intensity'] = (
                func(backgr_vals)
            )
        
        # Add intensity metrics
        foregr_vals = ref_ch_img_local[ref_ch_mask_local]
        for name, func in distribution_metrics_func.items():
            df_ref_ch.loc[:, f'ref_ch_{name}_intensity'] = (
                func(foregr_vals)
            )
        
        # Add background corrected metrics
        backgr_val = df_ref_ch['background_ref_ch_median_intensity']
        mean_val = df_ref_ch['ref_ch_mean_intensity']
        backr_corr_mean = mean_val - backgr_val
        df_ref_ch.loc[:, 'ref_ch_backgr_corrected_mean_intensity'] = (
            backr_corr_mean
        )
        df_ref_ch.loc[:, 'ref_ch_backgr_corrected_sum_intensity'] = (
            backr_corr_mean*vol_voxels
        )
        
        if calc_rp:
            df_ref_ch = features.add_regionprops_to_df(
                ref_ch_mask_local, df_ref_ch
            )
        
        desc = 'Quantifying sub-objects in reference channel'
        pbar_subobj = tqdm(
            total=len(ref_ch_rp), 
            ncols=100, 
            desc=desc, 
            leave=False
        )
        for sub_obj in ref_ch_rp:
            sub_vol_vox = np.count_nonzero(sub_obj.image)
            df_ref_ch.at[sub_obj.label, 'sub_obj_vol_vox'] = sub_vol_vox
            
            if vox_to_um3 is not None:
                sub_vol_fl = sub_vol_vox*vox_to_um3 
                df_ref_ch.at[sub_obj.label, 'sub_obj_vol_fl'] = sub_vol_fl
            
            # Add intensity metrics
            sub_foregr_vals = ref_ch_img_local[sub_obj.slice][sub_obj.image]
            for name, func in distribution_metrics_func.items():
                col = f'sub_obj_ref_ch_{name}_intensity'
                df_ref_ch.loc[sub_obj.label, col] = (
                    func(sub_foregr_vals)
                )
            
            # Add background corrected metrics
            try:
                sub_mean_val = (
                    df_ref_ch.loc[sub_obj.label, 'sub_obj_ref_ch_mean_intensity']
                )
                sub_backgr_val = backgr_val.loc[sub_obj.label]
                sub_backr_corr_mean = sub_mean_val - sub_backgr_val
            except Exception as err:
                sub_backr_corr_mean = np.nan
            
            col = 'sub_obj_ref_ch_backgr_corrected_mean_intensity'
            df_ref_ch.loc[sub_obj.label, col] = sub_backr_corr_mean
                
            col = 'sub_obj_ref_ch_backgr_corrected_sum_intensity'
            df_ref_ch.loc[sub_obj.label, col] = sub_backr_corr_mean*sub_vol_vox
            
            sub_objs[(ID, sub_obj.label)] = (obj, sub_obj)
            
            pbar_subobj.update()
        
        pbar_subobj.close()
        
        if calc_rp:
            df_ref_ch = features.add_regionprops_subobj_ref_ch_to_df(
                ref_ch_lab, df_ref_ch
            )
            
        dfs_ref_ch.append(df_ref_ch)
        keys.append((frame_i, ID))
        
        pbar.update()
    pbar.close()
    
    df_ref_ch = pd.concat(
        dfs_ref_ch, keys=keys, names=['frame_i', 'Cell_ID']
    )
    
    if filtering_features_thresholds is None:
        return df_agg, df_ref_ch, ref_ch_segm
    
    df_ref_ch_filtered = filters.filter_df_from_features_thresholds(
        df_ref_ch, 
        filtering_features_thresholds,
        logger_func=logger_func
    )
    df_ref_ch = df_ref_ch_filtered
    # Filter valid segmentation masks
    filtered_ref_ch_segm = np.zeros_like(ref_ch_segm)
    for (ID, id), _ in df_ref_ch.groupby(level=(1, 2)):
        obj, sub_obj = sub_objs[(ID, id)]
        obj, sub_obj = sub_objs[(ID, id)]
        filtered_ref_ch_segm[obj.slice][sub_obj.slice][sub_obj.image] = (
            ref_ch_segm[obj.slice][sub_obj.slice][sub_obj.image]
        )
    
    if np.array_equal(filtered_ref_ch_segm, ref_ch_segm):
        return df_agg, df_ref_ch, ref_ch_segm
    
    # Correct metrics that depends on filtering valid objects (total volume 
    # and background are different after removing some objects)
    for obj in lab_rp:
        ID = obj.label
        idx = (frame_i, obj.label)
        try:
            df_ref_ch_obj = df_ref_ch.loc[idx]
        except KeyError as err:
            continue
        
        ref_ch_lab_local = filtered_ref_ch_segm[obj.slice].copy()
        ref_ch_lab_local[ref_ch_lab_local!=obj.label] = 0
        ref_ch_mask_local = ref_ch_lab_local > 0
        
        ref_ch_img_local = ref_ch_img[obj.slice]
        
        ref_ch_num_fragments = len(df_ref_ch_obj)
        df_ref_ch.loc[idx, 'ref_ch_num_fragments'] = ref_ch_num_fragments
        
        ref_ch_vol_vox = df_ref_ch.loc[idx, 'sub_obj_vol_vox'].sum()
        df_ref_ch.loc[idx, 'ref_ch_vol_vox'] = ref_ch_vol_vox
        if vox_to_um3 is not None:
            vol_fl = ref_ch_vol_vox*vox_to_um3
            df_agg.at[(frame_i, ID), 'ref_ch_vol_um3'] = vol_fl
            df_ref_ch.loc[idx, 'ref_ch_vol_um3'] = vol_fl
        
        # Add background metrics
        bkgr_mask = np.logical_and(~ref_ch_mask_local, obj.image)
        backgr_vals = ref_ch_img_local[bkgr_mask]
        for name, func in distribution_metrics_func.items():
            df_ref_ch.loc[idx, f'background_ref_ch_{name}_intensity'] = (
                func(backgr_vals)
            )
        
        # Add background corrected metrics
        backgr_colname = 'background_ref_ch_median_intensity'
        backgr_val = df_ref_ch.loc[idx, backgr_colname]
        
        mean_colname = 'ref_ch_mean_intensity'
        mean_val = df_ref_ch.loc[idx, mean_colname]
        backr_corr_mean = mean_val - backgr_val
        df_ref_ch.loc[idx, 'ref_ch_backgr_corrected_mean_intensity'] = (
            backr_corr_mean
        )
        df_ref_ch.loc[idx, 'ref_ch_backgr_corrected_sum_intensity'] = (
            backr_corr_mean*vol_voxels
        )
        
        # Correct sub-object background corrected metrics        
        mean_corr_col = 'sub_obj_ref_ch_backgr_corrected_mean_intensity'
        df_ref_ch.loc[idx, mean_corr_col] -= backgr_val
    
        sum_corr_col = 'sub_obj_ref_ch_backgr_corrected_sum_intensity'
        df_ref_ch.loc[idx, sum_corr_col] = (
            df_ref_ch.loc[idx, mean_corr_col]
            * df_ref_ch.loc[idx, 'sub_obj_vol_vox']
        )
        
        ref_ch_local_lab = skimage.measure.label(ref_ch_mask_local)
        ref_ch_local_rp = skimage.measure.regionprops(ref_ch_local_lab)
        sub_obj_ids = [sub_obj.label for sub_obj in ref_ch_local_rp]
        df_ref_ch.loc[idx, 'sub_obj_id'] = sub_obj_ids
    
    return df_agg, df_ref_ch, filtered_ref_ch_segm

def _add_spot_vs_ref_location(ref_ch_mask, zyx_center, df, idx):
    if ref_ch_mask is None:
        return
    is_spot_in_ref_ch = int(ref_ch_mask[zyx_center] > 0)
    df.at[idx, 'is_spot_inside_ref_ch'] = is_spot_in_ref_ch
    _, dist_from_ref_ch = core.nearest_nonzero(ref_ch_mask, zyx_center)
    df.at[idx, 'spot_distance_from_ref_ch'] = dist_from_ref_ch

def _debug_compute_obj_spots_features(
        row, raw_spots_img_obj, zyx_center, sharp_spot_obj_z, 
        backgr_mask_z_spot, spheroids_mask, local_spot_bkgr_mask_z, ID=1
    ):
    print('')
    zyx_local = tuple(
        [getattr(row, col) for col in ZYX_LOCAL_COLS]
    )
    zyx_global = tuple(
        [getattr(row, col) for col in ZYX_GLOBAL_COLS]
    )
    print(f'Local coordinates = {zyx_local}')
    print(f'Global coordinates = {zyx_global}')
    print(f'Spot raw intensity at center = {raw_spots_img_obj[zyx_center]}')
    from ._debug import _compute_obj_spots_metrics
    win = _compute_obj_spots_metrics(
        sharp_spot_obj_z, backgr_mask_z_spot, 
        spheroids_mask[zyx_center[0]], 
        zyx_center[1:], local_spot_bkgr_mask_z, ID=ID, block=True
    )
    
def _compute_obj_spots_features(
        spots_img_obj, 
        df_obj_spots, 
        obj_mask, 
        sharp_spots_img_obj, 
        raw_spots_img_obj=None, 
        min_size_spheroid_mask=None, 
        zyx_voxel_size=None,
        dist_transform_spheroid=None,
        local_background_ring_width_pixel=5,
        optimise_for_high_spot_density=False,
        ref_ch_mask_obj=None, 
        ref_ch_img_obj=None, 
        zyx_resolution_limit_pxl=None, 
        get_backgr_from_inside_ref_ch_mask=False,
        custom_combined_measurements=None,
        logger_func=print,
        logger_warning_report=print,
        show_progress=True,
        debug=False,
        _ID=1
    ):
    """Compute spots features in the parent object.

    Parameters
    ----------
    spots_img_obj : (Z, Y, X) ndarray
        Spots' signal 3D z-stack image sliced at the segmentation object
        level. Note that this is the preprocessed image, i.e., after 
        gaussian filtering, but NOT after sharpening. Sharpening is used 
        only to improve detection. The first dimension must be 
        the number of z-slices.
    df_obj_spots : pandas.DataFrame
        Pandas DataFrame with `spot_id` as index.
    obj_mask : (Z, Y, X) ndarray of dtype bool
        Boolean mask of the segmentation object.
    sharp_spots_img_obj : (Z, Y, X) ndarray
        Spots' signal 3D z-stack image sliced at the segmentation object
        level. Note that this is the preprocessed image, i.e., after 
        gaussian filtering, sharpening etc. It is used to determine the 
        threshold for peak detection and for filtering against background. 
        The first dimension must be the number of z-slices.   
    raw_spots_img_obj : (Z, Y, X) ndarray or None, optional
        Raw spots' signal 3D z-stack image sliced at the segmentation
        object level. Note that this is the raw, unprocessed signal. 
        The first dimension must be  the number of z-slices. 
        If None, the features from the raw signal will not be computed.
    min_size_spheroid_mask : (Z, Y, X) ndarray of bools or pandas.Series of arrays, optional
        The boolean mask of the smallest spot expected. 
        This is pre-computed using the resolution limit equations and the 
        pixel size. If None, this will be computed from 
        `zyx_resolution_limit_pxl`. You can also pass a pandas.Series with 
        the same index as `df_obj_spots` with one mask for each spot.
        Default is None. 
    zyx_voxel_size : (z, y, x) sequence
        Voxel size in z-, y-, and x- directions in μm/pixel. Default is None
    dist_transform_spheroid : (Z, Y, X) ndarray, optional
        A distance transform of the `min_size_spheroid_mask`. This will be 
        multiplied by the spots intensities to reduce the skewing effect of 
        neighbouring peaks. 
        It must have the same shape of `min_size_spheroid_mask`.
        If None, normalisation will not be performed.
    local_background_ring_width_pixel : int, optional
        Width of the ring in pixels around each spot used to determine the 
        local effect sizes.
    optimise_for_high_spot_density : bool, optional
        If True and `dist_transform_spheroid` is None, then 
        `dist_transform_spheroid` will be initialized with the euclidean 
        distance transform of `min_size_spheroid_mask`.
    ref_ch_mask_obj : (Z, Y, X) ndarray of dtype bool or None, optional
        Boolean mask of the reference channel, e.g., obtained by 
        thresholding. The first dimension must be  the number of z-slices.
        If not None, it is used to compute background metrics, filter 
        and localise spots compared to the reference channel, etc.
    ref_ch_img_obj : (Z, Y, X) ndarray or None, optional
        Reference channel's signal 3D z-stack image sliced at the 
        segmentation object level. Note that this is the preprocessed image,
        i.e., after gaussian filtering, sharpening etc. 
        The first dimension must be the number of z-slices.
        If None, the features from the reference channel signal will not 
        be computed.
    get_backgr_from_inside_ref_ch_mask : bool, optional by default False
        If True, the background mask are made of the pixels that are inside 
        the segmented object but outside of the reference channel mask.
    zyx_resolution_limit_pxl : (z, y, x) tuple or None, optional
        Resolution limit in (z, y, x) direction in pixels. Default is None. 
        If `min_size_spheroid_mask` is None, this will be used to computed 
        the boolean mask of the smallest spot expected.
    custom_combined_measurements : dict or None, optional
        If not None, this is a dictionary of new column names as keys and 
        mathematical expressions that combines standard single-spot features. 
        The expression can be any text that can be evaluated by `pandas.eval`. 
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    logger_func : callable, optional
        Function used to print or log process information. Default is print
    logger_warning_report : callable, optional
        Additional function used by the SpotMAX cli Kernel to log 
        warnings in the report file. Default is print
    debug : bool, optional
        If True, displays intermediate results. Requires GUI libraries. 
        Default is False.
    """ 
    
    distribution_metrics_func = features.get_distribution_metrics_func()
    
    local_peaks_coords = df_obj_spots[ZYX_LOCAL_EXPANDED_COLS].to_numpy()
    result = transformations.get_spheroids_maks(
        local_peaks_coords, obj_mask.shape, 
        min_size_spheroid_mask=min_size_spheroid_mask, 
        zyx_radii_pxl=zyx_resolution_limit_pxl,
        return_spheroids_lab=True,
        debug=debug
    )
    spheroids_mask, spheroids_lab, min_size_spheroid_mask = result
    
    obj_rp = skimage.measure.regionprops(obj_mask.astype(np.uint8))[0]
    obj_centroid = obj_rp.centroid
    
    vox_to_fl = 1
    if zyx_voxel_size is not None:
        vox_to_fl = np.prod(zyx_voxel_size)
    
    # Get local background labels
    expand_dist = zyx_voxel_size[1]*local_background_ring_width_pixel
    spheroids_local_bkgr_lab = transformations.expand_labels(
        spheroids_lab, distance=expand_dist, zyx_vox_size=zyx_voxel_size
    )
    spheroids_local_bkgr_lab[spheroids_mask] = 0
    spheroids_local_bkgr_lab[~obj_mask] = 0

    # Check if spots_img needs to be normalised
    if get_backgr_from_inside_ref_ch_mask:
        backgr_mask = np.logical_and(ref_ch_mask_obj, ~spheroids_mask)
        spheroids_local_bkgr_lab[~ref_ch_mask_obj] = 0
        normalised_result = transformations.normalise_img(
            ref_ch_img_obj, backgr_mask, raise_if_norm_zero=False, 
            logger_func=logger_func, 
            logger_warning_report=logger_warning_report
        )
        normalised_ref_ch_img_obj, ref_ch_norm_value = normalised_result
        df_obj_spots.loc[:, 'ref_ch_normalising_value'] = ref_ch_norm_value
        normalised_result = transformations.normalise_img(
            spots_img_obj, backgr_mask, raise_if_norm_zero=True, 
            logger_func=logger_func, 
            logger_warning_report=logger_warning_report
        )
        normalised_spots_img_obj, spots_norm_value = normalised_result
        df_obj_spots.loc[:, 'spots_normalising_value'] = spots_norm_value
    else:
        backgr_mask = np.logical_and(obj_mask, ~spheroids_mask)
        normalised_spots_img_obj = spots_img_obj
        normalised_ref_ch_img_obj = ref_ch_img_obj
    
    # Calculate background metrics
    backgr_vals = sharp_spots_img_obj[backgr_mask]
    for name, func in distribution_metrics_func.items():
        df_obj_spots.loc[:, f'background_{name}_spot_detection_image'] = (
            func(backgr_vals)
        )
    
    if raw_spots_img_obj is not None:
        backgr_vals = raw_spots_img_obj[backgr_mask]
        for name, func in distribution_metrics_func.items():
            df_obj_spots.loc[:, f'background_{name}_raw_image'] = (
                func(backgr_vals)
            )
    
    backgr_vals = spots_img_obj[backgr_mask]
    for name, func in distribution_metrics_func.items():
        df_obj_spots.loc[:, f'background_{name}_preproc_image'] = (
            func(backgr_vals)
        )
    
    if raw_spots_img_obj is None:
        raw_spots_img_obj = spots_img_obj
    
    if show_progress:
        pbar_desc = 'Computing spots features'
        pbar = tqdm(
            total=len(df_obj_spots), ncols=100, desc=pbar_desc, position=3, 
            leave=False
        )
    
    spot_ids_to_drop = []
    for spot_idx, row in enumerate(df_obj_spots.itertuples()):
        spot_id = row.Index
        if isinstance(min_size_spheroid_mask, pd.Series):
            spot_mask = min_size_spheroid_mask.loc[spot_id]
        else:
            spot_mask = min_size_spheroid_mask
        zyx_center = tuple(
            [getattr(row, col) for col in ZYX_LOCAL_EXPANDED_COLS]
        )

        slices = transformations.get_slices_local_into_global_3D_arr(
            zyx_center, spots_img_obj.shape, spot_mask.shape
        )
        slice_global_to_local, slice_crop_local = slices
        
        # Intensity image at spot center z-slice 
        sharp_spot_obj_z = sharp_spots_img_obj[zyx_center[0]]
        preproc_spot_obj_z = spots_img_obj[zyx_center[0]]
        raw_spot_obj_z = raw_spots_img_obj[zyx_center[0]]
        
        # Background values at spot z-slice
        backgr_mask_z_spot = backgr_mask[zyx_center[0]]
        backgr_vals_z_spot = sharp_spot_obj_z[backgr_mask_z_spot]
        
        if len(backgr_vals_z_spot) == 0:
            # This is most likely because the reference channel mask at 
            # center z-slice is smaller than the spot resulting 
            # in no background mask (since the background is outside of 
            # the spot but inside the ref. ch. mask) --> there is not 
            # enough ref. channel to consider this a valid spot.
            spot_ids_to_drop.append(spot_id)
            continue
        
        local_spot_bkgr_lab_z = spheroids_local_bkgr_lab[zyx_center[0]]
        local_spot_bkgr_mask_z = local_spot_bkgr_lab_z==(spot_idx+1)
        local_sharp_spot_bkgr_vals = sharp_spot_obj_z[local_spot_bkgr_mask_z]
        local_preproc_spot_bkgr_vals = preproc_spot_obj_z[local_spot_bkgr_mask_z]
        local_raw_spot_bkgr_vals = raw_spot_obj_z[local_spot_bkgr_mask_z]
        
        if debug:
            _debug_compute_obj_spots_features(
                row, raw_spots_img_obj, zyx_center, sharp_spot_obj_z, 
                backgr_mask_z_spot, spheroids_mask, local_spot_bkgr_mask_z, 
                ID=_ID
            )
            import pdb; pdb.set_trace()

        # Add spot volume from mask
        spot_mask_vol = np.count_nonzero(spot_mask)
        df_obj_spots.at[spot_id, 'spot_mask_volume_voxel'] = spot_mask_vol
        spot_mask_vol_fl = spot_mask_vol*vox_to_fl
        df_obj_spots.at[spot_id, 'spot_mask_volume_fl'] = spot_mask_vol_fl
            
        # Add background metrics at center z-slice
        sharp_spot_bkgr_values_z = sharp_spot_obj_z[backgr_mask_z_spot]
        features.add_distribution_metrics(
            sharp_spot_bkgr_values_z, df_obj_spots, spot_id, 
            col_name='background_*name_z_slice_spot_detection_image'
        )
        
        zc_spot = zyx_center[0]
        spot_bkgr_values_z = spots_img_obj[zc_spot, backgr_mask_z_spot]
        features.add_distribution_metrics(
            spot_bkgr_values_z, df_obj_spots, spot_id, 
            col_name='background_*name_z_slice_preproc_image'
        )
        
        if raw_spots_img_obj is not None:
            raw_bkgr_values_z = raw_spots_img_obj[zc_spot, backgr_mask_z_spot]
            features.add_distribution_metrics(
                raw_bkgr_values_z, df_obj_spots, spot_id, 
                col_name='background_*name_z_slice_raw_image'
            )
            
        # Crop masks
        spheroid_mask = spot_mask[slice_crop_local]
        spot_slice_local = spots_img_obj[slice_global_to_local]

        # Get the sharp spot sliced
        sharp_spot_slice_z = sharp_spot_obj_z[slice_global_to_local[-2:]]
        
        if optimise_for_high_spot_density:
            dist_transform_spheroid = (
                transformations.norm_distance_transform_edt(spot_mask)
            )
        
        if dist_transform_spheroid is None:
            # Do not optimise for high spot density
            sharp_spot_slice_z_transf = sharp_spot_slice_z
        else:
            dist_transf = dist_transform_spheroid[slice_crop_local]
            sharp_spot_slice_z_transf = (
                transformations.normalise_spot_by_dist_transf(
                    sharp_spot_slice_z, dist_transf.max(axis=0),
                    backgr_vals_z_spot, how='range', 
                    debug=False # spot_id==32
            ))

        # Get spot intensities
        spot_intensities = spot_slice_local[spheroid_mask]
        spheroid_mask_proj = spheroid_mask.max(axis=0)
        sharp_spot_intensities_z_edt = (
            sharp_spot_slice_z_transf[spheroid_mask_proj]
        )
        
        # Get local background intensities
        local_sharp_bkgr_vals = local_spot_bkgr_mask_z

        value = spots_img_obj[zyx_center]
        df_obj_spots.at[spot_id, 'spot_center_preproc_intensity'] = value
        features.add_distribution_metrics(
            spot_intensities, df_obj_spots, spot_id, 
            col_name='spot_preproc_*name_in_spot_minimumsize_vol',
            add_bkgr_corrected_metrics=True, 
            logger_warning_report=logger_warning_report, 
            logger_func=logger_func, iter_idx=spot_idx
        )
        
        if raw_spots_img_obj is None:
            raw_spot_intensities = spot_intensities
        else:
            raw_spot_intensities = (
                raw_spots_img_obj[slice_global_to_local][spheroid_mask]
            )
            value = raw_spots_img_obj[zyx_center]
            df_obj_spots.at[spot_id, 'spot_center_raw_intensity'] = value

            features.add_distribution_metrics(
                raw_spot_intensities, df_obj_spots, spot_id, 
                col_name='spot_raw_*name_in_spot_minimumsize_vol',
                add_bkgr_corrected_metrics=True, 
                logger_warning_report=logger_warning_report, 
                logger_func=logger_func, iter_idx=spot_idx
            )

        # Intensities metrics from background around the spots (local)
        features.add_distribution_metrics(
            local_sharp_spot_bkgr_vals, df_obj_spots, spot_id, 
            col_name='background_local_*name_z_slice_spot_detection_image',
            add_bkgr_corrected_metrics=False
        )
        features.add_distribution_metrics(
            local_preproc_spot_bkgr_vals, df_obj_spots, spot_id, 
            col_name='background_local_*name_z_slice_preproc_image',
            add_bkgr_corrected_metrics=False
        )
        features.add_distribution_metrics(
            local_raw_spot_bkgr_vals, df_obj_spots, spot_id, 
            col_name='background_local_*name_z_slice_raw_image',
            add_bkgr_corrected_metrics=False
        )
        
        # When comparing to the background we use the sharpened image 
        # at the center z-slice of the spot
        features.add_ttest_values(
            sharp_spot_intensities_z_edt, backgr_vals_z_spot, 
            df_obj_spots, spot_id, name='spot_vs_backgr',
            logger_func=logger_func
        )
        
        features.add_effect_sizes(
            sharp_spot_intensities_z_edt, backgr_vals_z_spot, 
            df_obj_spots, spot_id, name='spot_vs_backgr',
            debug=debug, logger_warning_report=logger_warning_report, 
            logger_func=logger_func
        )
        
        features.add_effect_sizes(
            sharp_spot_intensities_z_edt, local_sharp_spot_bkgr_vals, 
            df_obj_spots, spot_id, name='spot_vs_local_backgr',
            debug=debug, logger_warning_report=logger_warning_report,
            logger_func=logger_func
        )
        
        # if spot_id == 32:
        #     import pdb; pdb.set_trace()
        
        features.add_spot_localization_metrics(
            df_obj_spots, spot_id, zyx_center, obj_centroid,
            voxel_size=zyx_voxel_size,
            logger_warning_report=logger_warning_report,
            logger_func=logger_func
        )
        
        if ref_ch_img_obj is None:
            # Raw reference channel not present --> continue
            if show_progress:
                pbar.update()
            continue

        normalised_spot_intensities, normalised_ref_ch_intensities = (
            features.get_normalised_spot_ref_ch_intensities(
                normalised_spots_img_obj, normalised_ref_ch_img_obj,
                spheroid_mask, slice_global_to_local
            )
        )
        features.add_ttest_values(
            normalised_spot_intensities, normalised_ref_ch_intensities, 
            df_obj_spots, spot_id, name='spot_vs_ref_ch',
            logger_func=logger_func
        )
        features.add_effect_sizes(
            normalised_spot_intensities, normalised_ref_ch_intensities, 
            df_obj_spots, spot_id, name='spot_vs_ref_ch', 
            logger_warning_report=logger_warning_report, 
            logger_func=logger_func
        )
        _add_spot_vs_ref_location(
            ref_ch_mask_obj, zyx_center, df_obj_spots, spot_id
        )                
        
        value = ref_ch_img_obj[zyx_center]
        df_obj_spots.at[spot_id, 'ref_ch_raw_intensity_at_center'] = value

        ref_ch_intensities = (
            ref_ch_img_obj[slice_global_to_local][spheroid_mask]
        )
        features.add_distribution_metrics(
            ref_ch_intensities, df_obj_spots, spot_id, 
            col_name='ref_ch_raw_*name_in_spot_minimumsize_vol'
        )
        if show_progress:
            pbar.update()
    if show_progress:
        pbar.close()
    
    if custom_combined_measurements is not None:
        df_obj_spots = features.add_custom_combined_measurements(
            df_obj_spots, logger_func=logger_func, 
            **custom_combined_measurements,   
        )
    
    return df_obj_spots, spot_ids_to_drop

def spot_detection(
        image,
        spots_segmantic_segm=None,
        detection_method='peak_local_max',
        spot_footprint=None,
        spots_zyx_radii_pxl=None,
        return_spots_mask=False,
        lab=None,
        return_df=False,
        logger_func=None,
        validate=False,
        debug=False
    ):
    """Detect spots and return their coordinates

    Parameters
    ----------
    image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray
        Input 2D or 3D image.
    spots_segmantic_segm : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        If not None and detection_method is 'peak_local_max', peaks will be 
        searched only where spots_segmantic_segm > 0. Default is None
    detection_method : {'peak_local_max', 'label_prediction_mask'}, optional
        Method used to detect the peaks. Default is 'peak_local_max'
        For more details, see the parameter `Spots detection method` at the 
        following webpage: 
        https://spotmax.readthedocs.io/en/latest/parameters/parameters_description.html#confval-Spots-detection-method
    spot_footprint : numpy.ndarray of bools, optional
        If not None, only one peak is searched in the footprint at every point 
        in the image. Default is None
    spots_zyx_radii_pxl : (z, y, x) sequence of floats, optional
        Rough estimation of the expected spot radii in z, y, and x direction
        in pixels. The values are used to determine the spot footprint if 
        spot_footprint is not provided. Default is None
    return_spots_mask : bool, optional
        This is forced to be True if `detection_method` is equal to 
        'label_prediction_mask'.
        If True, the second element returned will be a list of region properties 
        (see scikit-image `skimage.measure.regionprops`) with an additional 
        attribute called `zyx_local_center`. Default is False
    lab : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        Optional input segmentation image with the masks of the objects, i.e. 
        single cells. It will be used to create the pandas.DataFrame with 
        spots coordinates per object (if `return_df` is True).
        If None, it will be generated with one object covering the entire image. 
        Default is None.
    return_df : bool, optional
        If True, returns a pandas DataFrame. More details on the returned 
        items section below. Default is False
    logger_func : callable, optional
        If not None, this is the function used to print or log process information. 
        Default is None
    validate : bool, optional
        If True, it checks if there are spots detected outside of segmented objects  
        and returns an addtional False if that happens

    Returns
    -------
    spots_coords : (N, 3) numpy.ndarray of ints
        (N, 3) array of integers where each row is the (z, y, x) coordinates 
        of one peak. Returned only if `return_df` is `False`.
    
    df_coords : pandas.DataFrame 
        DataFrame with Cell_ID as index and columns 
        {'z', 'y', 'x'} with the detected spots coordinates.
        Returned only if `return_df` is `True`.
    
    spots_masks : list of region properties or None
        List of spots masks as boolean arrays. None if `return_spots_mask` 
        is `False`.
    
    See also
    --------
    `skimage.measure.regionprops <https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops>`__
    """        
    if spot_footprint is None and spots_zyx_radii_pxl is not None:
        spot_footprint = features.get_peak_footprint(
            spots_zyx_radii_pxl, image 
        )
        
    if spots_zyx_radii_pxl is None:
        spots_zyx_radii_pxl = np.array([1, 1, 1])
    
    if spots_segmantic_segm is None:
        spots_segmantic_segm = np.ones(image.shape, int)
    
    is_zstack = True
    if image.ndim==2 or (image.ndim==3 and image.shape[0]==1):
        is_zstack = False
    
    if spot_footprint is not None and not is_zstack and spot_footprint.ndim==3:
        # Make sure that spot_footprint is 2D with 2D input image
        spot_footprint = spot_footprint.max(axis=0)
    
    spots_masks = None
    
    if logger_func is not None:
        logger_func(f'Detecting spots with method `{detection_method}`')
    
    if detection_method == 'peak_local_max':
        detect_image = np.squeeze(image)
        labels = np.squeeze(spots_segmantic_segm)
        min_distance = spots_zyx_radii_pxl
        if not is_zstack and len(min_distance) == 3:
            # Make sure that min_distance is 2 values for 2D images
            min_distance = spots_zyx_radii_pxl[-2:]
        
        spots_coords = features.find_local_peaks(
            detect_image, 
            min_distance=min_distance,
            footprint=spot_footprint, 
            labels=labels,
            debug=debug
        )
        if return_spots_mask:
            spots_masks = transformations.from_spots_coords_to_spots_masks(
                spots_coords, spots_zyx_radii_pxl, debug=debug
            )
    elif detection_method == 'label_prediction_mask':
        prediction_lab = skimage.measure.label(spots_segmantic_segm>0)
        prediction_lab, _ = transformations.reshape_lab_image_to_3D(
            prediction_lab, image
        )
        prediction_lab_rp = skimage.measure.regionprops(prediction_lab)
        num_spots = len(prediction_lab_rp)
        spots_coords = np.zeros((num_spots, 3), dtype=int)
        spots_masks = []
        for s, spot_obj in enumerate(prediction_lab_rp):
            zyx_coords = tuple([round(c) for c in spot_obj.centroid])
            spots_coords[s] = zyx_coords
            spots_masks.append(spot_obj.image)
    
    if return_df:
        if lab is None:
            lab = np.ones(image.shape, dtype=np.uint8)
        else:
            lab, _ = transformations.reshape_lab_image_to_3D(lab, image)
        df_coords = transformations.from_spots_coords_arr_to_df(
            spots_coords, lab
        )
        out = df_coords, spots_masks
    else:
        out = spots_coords, spots_masks
    
    if validate:
        out = (*out, True)
    
    if lab is None:
        return out
    
    if np.all(lab[tuple(spots_coords.transpose())]):
        return out
    
    return (*out[:-1], False)
    

def _replace_None_with_empty_dfs(dfs_spots_gop):
    """Replace Nones in `dfs_spots_gop` with empty DataFrames

    Parameters
    ----------
    dfs_spots_gop : list
        List of DataFrames as calculated in `pipe.spots_calc_features_and_filter` 

    Returns
    -------
    list
        List of DataFrames as calculated in `pipe.spots_calc_features_and_filter`
        with Nones replaced with empty DataFrames
    
    Notes
    -----
    When the spot center does not lie on a segmented object but its mask still 
    touches it we keep those spots with Cell_ID = 0. However, we keep them 
    only in the detected spots and we dropped them for the valid spots. 
    To achieve this, in `pipe.spots_calc_features_and_filter` we temporarily 
    place None in the `dfs_spots_gop` and we replace the Nones with empty 
    DataFrames in order to use the same keys for both `dfs_spots_gop` and 
    `dfs_spots_det` (where `dfs_spots_det` keys might be (frame_i, 0))
    
    """    
    None_idxs = []
    df_template = None
    for d, df in enumerate(dfs_spots_gop):
        if df is not None:
            df_template = df
        else:
            None_idxs.append(d)
    
    if not None_idxs:
        return dfs_spots_gop
    
    if df_template is None:
        return dfs_spots_gop
        
    empty_df = pd.DataFrame({
        col:pd.Series(dtype=df_template[col].dtype) 
        for col in df_template.columns}
    )
    for i in None_idxs:
        dfs_spots_gop[i] = empty_df
    
    return dfs_spots_gop
    

def _init_df_spots_IDs_0(
        df_spots_coords, lab, rp, delta_tol, spots_zyx_radii_pxl
    ):
    closest_IDs = df_spots_coords.loc[[0], 'closest_ID'].unique()
    IDs = [obj.label for obj in rp]
    dfs_spots_IDs_0 = {}
    for closest_ID in closest_IDs:
        df_spots_closest_ID = (
            df_spots_coords[df_spots_coords['closest_ID']==closest_ID]
        )
        closest_ID_idx = IDs.index(closest_ID)
        obj_closest_ID = rp[closest_ID_idx]
        expanded_obj_closest_ID = transformations.get_expanded_obj_slice_image(
            obj_closest_ID, delta_tol, lab
        )
        _, _, crop_obj_start_closest_ID = expanded_obj_closest_ID
        df_spots_IDs_0, _, _ = transformations.init_df_features(
            df_spots_closest_ID, obj_closest_ID, crop_obj_start_closest_ID, 
            spots_zyx_radii_pxl, ID=0, tot_num_spots=len(df_spots_coords)
        )
        if df_spots_IDs_0 is None:
            continue

        dfs_spots_IDs_0[closest_ID] = (
            df_spots_IDs_0.set_index('spot_id').sort_index()
        )
    return dfs_spots_IDs_0

def spots_calc_features_and_filter(
        image, 
        spots_zyx_radii_pxl,
        df_spots_coords,
        frame_i=0,
        sharp_spots_image=None,
        lab=None,
        rp=None,
        gop_filtering_thresholds=None,
        delta_tol=None,   
        raw_image=None,
        ref_ch_mask_or_labels=None, 
        ref_ch_img=None,   
        keep_only_spots_in_ref_ch=False,
        remove_spots_in_ref_ch=False,
        use_spots_segm_masks=False,
        min_size_spheroid_mask=None,
        zyx_voxel_size=None,
        optimise_for_high_spot_density=False,
        dist_transform_spheroid=None,
        local_background_ring_width='5 pixel',
        get_backgr_from_inside_ref_ch_mask=False,
        custom_combined_measurements=None,
        show_progress=True,
        verbose=True,
        logger_func=print,
        logger_warning_report=print
    ):
    """Calculate spots features and filter valid spots based on 
    `gop_filtering_thresholds`.

    Parameters
    ----------
    image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray
        Input 2D or 3D image.
    spots_zyx_radii_pxl : (z, y, x) sequence of floats, optional
        Rough estimation of the expected spot radii in z, y, and x direction
        in pixels. The values are used to build the ellipsoid mask centered at 
        each spot. The volume of the ellipsoid is then used for those aggregated 
        metrics like the mean intensity in the spot.
    df_spots_coords : pandas.DataFrame
        DataFrame with Cell_ID as index and the columns {'z', 'y', 'x'} which 
        are the coordinates of the spots in `image`. 
    frame_i : int, optional
        Frame index in timelapse data. Default is 0
    sharp_spots_image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray, optional
        Optional image that was filtered to enhance the spots (e.g., using 
        spotmax.filters.DoG_spots). This image will be used for those features 
        that requires comparing the spot's signal to a reference signal 
        (background or reference channel). If None, `image` will be used 
        instead. Default is None
    lab : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        Optional input segmentation image with the masks of the objects, i.e. 
        single cells. If None, it will be generated with one object covering 
        the entire image. Default is None.
    rp : list of skimage.measure.RegionProperties, optional
        If not None, list of properties of objects in `lab` as returned by 
        skimage.measure.regionprops(lab). If None, this will be computed 
        with `skimage.measure.regionprops(lab)`. Default is None
    gop_filtering_thresholds : dict of {'feature_name': (min_value, max_value)}, optional
        Features and their maximum and minimum values to filter valid spots. 
        A spot is valid when `feature_name` is greater than `min_value` and 
        lower than `max_value`. If a value is None it means there is no minimum 
        or maximum threshold. Default is None
    delta_tol : (z, y, x) sequence of floats, optional
        If not None, these values will be used to enlarge the segmented objects. 
        It will prevent clipping the spots masks for those spots whose intensities 
        bleed outside of the object (e.g., single cell). Default is None
    raw_image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray, optional
        Optional image to calculate features from. The name 
        of these features will have the text '_raw_'. Default is None
    ref_ch_mask_or_labels : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        Instance or semantic segmentation of the reference channel. If not None, 
        this is used to calculate the background intensity inside the segmented 
        object from `lab` but outside of the reference channel mask. 
        Default is None
    ref_ch_img : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray, optional
        Reference channel image. Default is None
    keep_only_spots_in_ref_ch : bool, optional
        If True, drops the spots that are outside of the reference channel mask. 
        Default is False
    remove_spots_in_ref_ch : bool, optional
        If True, removes the spots that are inside the reference channel mask.
        Default is False
    use_spots_segm_masks : bool, optional
        If True and `df_spots_coords` has a column called 'spot_maks' with one 
        spot mask for each spot then `min_size_spheroid_mask` is ignored and 
        the spot mask from `df_spots_coords` will be used.
        Default is False
    min_size_spheroid_mask : (M, N) numpy.ndarray or (K, M, N) numpy.ndarray or bools, optional
        Spheroid mask used to calculate those aggregated features like the 
        mean intensity in each spot. If this value is None, it will be created 
        from `spots_zyx_radii_pxl`. Note that if `use_spots_segm_masks` is
        True, this parameter might be ignored. Default is None      
    zyx_voxel_size : (z, y, x) sequence
        Voxel size in z-, y-, and x- directions in μm/pixel. If None, this will 
        be initialize to [1, 1, 1]. Default is None
    optimise_for_high_spot_density : bool, optional
        If True and `dist_transform_spheroid` is None, then `dist_transform_spheroid`
        will be initialized with the euclidean distance transform of 
        `min_size_spheroid_mask`.
    dist_transform_spheroid : (M, N) numpy.ndarray or (K, M, N) numpy.ndarray of floats, optional
        Optional probability map that will be multiplicated to each spot's 
        intensities. An example is the euclidean distance tranform 
        (normalised to the range 0-1). This is useful to reduce the influence 
        of bright neighbouring spots on dimmer spots since the intensities of the 
        bright spot can bleed into the edges of the dimmer spot skewing its 
        metrics like the mean intensity. 
        If None and `optimise_for_high_spot_density` is True, this will be 
        initialized with the euclidean distance transform of 
        `min_size_spheroid_mask`. Default is None
    local_background_ring_width : '<value> pixel' or '<value> micrometre'
        Width of the ring around each spot used to determine the local effect 
        sizes. It can be specified in pixel or micrometre, e.g. '5 pixel' or 
        '0.4 micrometre'. If the unit is 'micrometre', then the value will 
        be converted to 'pixel' using `zyx_voxel_size[-1]` and rounded to the 
        nearest integer. Default is '5 pixel'
    get_backgr_from_inside_ref_ch_mask : bool, optional
        If True, the background will be determined from the pixels that are
        outside of the spots, but inside the reference channel mask. 
        Default is False
    custom_combined_measurements : dict or None, optional
        If not None, this is a dictionary of new column names as keys and 
        mathematical expressions that combines standard single-spot features. 
        The expression can be any text that can be evaluated by `pandas.eval`. 
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    show_progress : bool, optional
        If True, display progressbars. Default is False
    verbose : bool, optional
        If True, additional information text will be printed to the terminal. 
        Default is True
    logger_func : callable, optional
        Function used to print or log process information. Default is print
    logger_warning_report : callable, optional
        Additional function used by the SpotMAX cli Kernel to log 
        warnings in the report file. Default is print

    Returns
    -------
    keys : list of 2-tuple (int, int) 
        List of keys that can be used to concatenate the 
        dataframes with 
        `pandas.concat(dfs_spots_gop, keys=keys, names=['frame_i', 'Cell_ID'])` 
    dfs_spots_det : list of pandas.DataFrames
        List of DataFrames with the features columns 
        for each frame and ID of the segmented objects in `lab` 
    dfs_spots_gop : list of pandas.DataFrames
        Same as `dfs_spots_det` but with only the valid spots 
    
    See also
    --------
    `skimage.measure.regionprops <https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops>`__
    """    
    if verbose and len(df_spots_coords) > 0:
        print('')
        logger_func('Quantifying spots...')

    if gop_filtering_thresholds is None:
        gop_filtering_thresholds = {}
    
    if zyx_voxel_size is None:
        zyx_voxel_size = np.array([1, 1, 1])
    
    local_backgr_ring_width_pixel = utils.get_local_backgr_ring_width_pixel(
        local_background_ring_width, zyx_voxel_size[-1]
    )
    
    if lab is None:
        lab = np.ones(image.shape, dtype=np.uint8)
    
    lab, image = transformations.reshape_lab_image_to_3D(lab, image)
    
    if rp is None:
        rp = skimage.measure.regionprops(lab)
    
    if delta_tol is None:
        delta_tol = transformations.get_expand_obj_delta_tolerance(
            spots_zyx_radii_pxl
        )
    
    if use_spots_segm_masks and 'spot_mask' not in df_spots_coords.columns:
        use_spots_segm_masks = False
    
    calc_spheroid_mask = (
        min_size_spheroid_mask is None 
        and not use_spots_segm_masks 
    )
    if calc_spheroid_mask:
        min_size_spheroid_mask = transformations.get_local_spheroid_mask(
            spots_zyx_radii_pxl
        )
    
    calc_dist_spheroid_mask = (
        optimise_for_high_spot_density
        and dist_transform_spheroid is None 
        and not use_spots_segm_masks 
    )
    if calc_dist_spheroid_mask:
        dist_transform_spheroid = transformations.norm_distance_transform_edt(
            min_size_spheroid_mask
        )
    
    if use_spots_segm_masks:
        # Force recalc of dist transform in _compute_obj_spots_features
        min_size_spheroid_mask = None
        dist_transform_spheroid = None
    
    if sharp_spots_image is None:
        sharp_spots_image = image
    
    if show_progress:
        desc = 'Quantifying spots'
        pbar = tqdm(
            total=len(rp), ncols=100, desc=desc, position=3, leave=False
        )
    
    keys = []
    dfs_spots_det = []
    dfs_spots_gop = []
    dfs_spots_IDs_0 = None
    if 0 in df_spots_coords.index and 'closest_ID' in df_spots_coords.columns:
        dfs_spots_IDs_0 = _init_df_spots_IDs_0(
            df_spots_coords, lab, rp, delta_tol, spots_zyx_radii_pxl
        )
        keys.extend([(frame_i, 0)]*len(dfs_spots_IDs_0))
        dfs_spots_det.extend(dfs_spots_IDs_0.values())
        dfs_spots_gop.extend([None]*len(dfs_spots_IDs_0))
    
    last_spot_id = 0
    filtered_spots_info = defaultdict(dict)
    obj_idx = len(keys)
    for obj in rp:
        df_spots_coords = transformations.add_zyx_local_coords_if_not_valid(
            df_spots_coords, obj
        )
        
        expanded_obj = transformations.get_expanded_obj_slice_image(
            obj, delta_tol, lab
        )
        obj_slice, obj_image, crop_obj_start = expanded_obj

        local_spots_img = image[obj_slice]
        local_sharp_spots_img = sharp_spots_image[obj_slice]

        result = transformations.init_df_features(
            df_spots_coords, obj, crop_obj_start, spots_zyx_radii_pxl
        )
        df_obj_spots_det, expanded_obj_coords, do_increment_spot_id = result
            
        if df_obj_spots_det is None:
            filtered_spots_info[obj.label]['start_num_spots'] = 0
            filtered_spots_info[obj.label]['end_num_spots'] = 0
            filtered_spots_info[obj.label]['num_iter'] = 0
            continue
        
        # Increment spot_id with previous object
        if do_increment_spot_id:
            df_obj_spots_det['spot_id'] += last_spot_id
            
        df_obj_spots_det = df_obj_spots_det.set_index('spot_id').sort_index()
        
        if use_spots_segm_masks:
            min_size_spheroid_mask = df_obj_spots_det['spot_mask']
        
        keys.append((frame_i, obj.label))
        num_spots_detected = len(df_obj_spots_det)
        last_spot_id += num_spots_detected

        if ref_ch_mask_or_labels is not None:
            local_ref_ch_mask = ref_ch_mask_or_labels[obj_slice]>0
            local_ref_ch_mask = np.logical_and(local_ref_ch_mask, obj_image)
        else:
            local_ref_ch_mask = None

        if ref_ch_img is not None:
            local_ref_ch_img = ref_ch_img[obj_slice]
        else:
            local_ref_ch_img = None
        
        raw_spots_img_obj = None
        if raw_image is not None:
            raw_spots_img_obj = raw_image[obj_slice]

        dfs_spots_det.append(df_obj_spots_det.copy())
        df_obj_spots_gop = df_obj_spots_det.copy()
        
        do_filter_spots_with_ref_ch_masks = (
            keep_only_spots_in_ref_ch or remove_spots_in_ref_ch
        )
        
        start_num_spots = len(df_obj_spots_det)
        filtered_spots_info[obj.label]['start_num_spots'] = start_num_spots
        debug = False # obj.label == 41 or obj.label == 44
        i = 0
        while True:     
            num_spots_prev = len(df_obj_spots_gop)
            if num_spots_prev == 0:
                num_spots_filtered = 0
                break
            
            bkgr_from_in_reg_ch = get_backgr_from_inside_ref_ch_mask
            df_obj_spots_gop, spot_ids_to_drop = _compute_obj_spots_features(
                local_spots_img, 
                df_obj_spots_gop.copy(), 
                obj_image, 
                local_sharp_spots_img, 
                raw_spots_img_obj=raw_spots_img_obj,
                min_size_spheroid_mask=min_size_spheroid_mask, 
                zyx_voxel_size=zyx_voxel_size,
                dist_transform_spheroid=dist_transform_spheroid,
                local_background_ring_width_pixel=local_backgr_ring_width_pixel,
                optimise_for_high_spot_density=optimise_for_high_spot_density,
                ref_ch_mask_obj=local_ref_ch_mask, 
                ref_ch_img_obj=local_ref_ch_img,
                get_backgr_from_inside_ref_ch_mask=bkgr_from_in_reg_ch,
                zyx_resolution_limit_pxl=spots_zyx_radii_pxl,
                custom_combined_measurements=custom_combined_measurements,
                debug=debug, 
                logger_func=logger_func,
                logger_warning_report=logger_warning_report,
                show_progress=show_progress,
                _ID=obj.label
            )
            if i == 0:
                # Store metrics at first iteration
                dfs_spots_det[obj_idx] = df_obj_spots_gop.copy()
            else:
                # Update metrics in detect df
                dfs_spots_det[obj_idx].loc[df_obj_spots_gop.index] = (
                    df_obj_spots_gop
                )
            
            if do_filter_spots_with_ref_ch_masks and i == 0:
                # Filter spots with ref_ch_masks at first iteration
                df_obj_spots_gop = filters.filter_spots_with_ref_ch_masks(
                    df_obj_spots_gop, local_ref_ch_mask, 
                    expanded_obj_coords, 
                    keep_inside=keep_only_spots_in_ref_ch, 
                    remove_inside=remove_spots_in_ref_ch,
                )
            
            if spot_ids_to_drop:
                df_obj_spots_gop = df_obj_spots_gop.drop(
                    index=spot_ids_to_drop, errors='ignore'
                )
            
            # from . import _debug
            # _debug._spots_filtering(
            #     local_spots_img, df_obj_spots_gop, obj, obj_image
            # )

            df_obj_spots_gop = filter_spots_from_features_thresholds(
                df_obj_spots_gop, gop_filtering_thresholds,
                is_spotfit=False, 
                debug=False, # obj.label==6,
                logger_func=logger_func, 
                verbose=False
            )
            num_spots_filtered = len(df_obj_spots_gop)   

            if num_spots_filtered == num_spots_prev or num_spots_filtered == 0:
                # Number of filtered spots stopped decreasing --> stop loop
                break

            i += 1

        filtered_spots_info[obj.label]['end_num_spots'] = num_spots_filtered
        filtered_spots_info[obj.label]['num_iter'] = i
        
        dfs_spots_gop.append(df_obj_spots_gop)
        
        obj_idx += 1

        if show_progress:
            pbar.update()
    if show_progress:
        pbar.close()
    
    dfs_spots_gop = _replace_None_with_empty_dfs(dfs_spots_gop)
    
    _log_filtered_number_spots(
        verbose, frame_i, filtered_spots_info, logger_func, 
        category='valid spots based on features'
    )
        
    return keys, dfs_spots_det, dfs_spots_gop

def _log_filtered_number_spots(
        verbose, frame_i, filtered_spots_info, logger_func, 
        category='valid spots'
    ):
    if not verbose:
        return
    
    are_all_objs_with_0_spots = True
    num_spots_filtered_log = []
    for ID, info_ID in filtered_spots_info.items():
        start_num_spots = info_ID['start_num_spots']
        end_num_spots = info_ID['end_num_spots']
        num_iter = info_ID['num_iter']
        if start_num_spots != 0:
            are_all_objs_with_0_spots = False
            
        if start_num_spots == end_num_spots:
            continue
        txt = (
            f'  * Object ID {ID} = {start_num_spots} --> {end_num_spots} '
            f'({num_iter} iterations)'
        )
        num_spots_filtered_log.append(txt)
    
    if are_all_objs_with_0_spots:
        return
    
    if num_spots_filtered_log:
        info = '\n'.join(num_spots_filtered_log)
    else:
        info = 'All spots are valid'
        
    print('')
    header = f'Frame n. {frame_i+1}: number of spots after filtering {category}:'
    print('*'*len(header))
    logger_func(f'{header}\n\n{info}')
    print('-'*len(header))

def spotfit(
        kernel,
        spots_img, 
        df_spots, 
        zyx_voxel_size=None, 
        zyx_spot_min_vol_um=None,
        spots_zyx_radii_pxl=None,
        delta_tol=None,
        rp=None, 
        lab=None, 
        frame_i=0, 
        ref_ch_mask_or_labels=None, 
        spots_masks_check_merge=None,
        drop_peaks_too_close=False,
        return_df=False,
        use_gpu=False,
        show_progress=True,
        verbose=True,
        logger_func=print,
        custom_combined_measurements=None,
        max_number_pairs_check_merge=11,
        xy_center_half_interval_val=0.1, 
        z_center_half_interval_val=0.2, 
        sigma_x_min_max_expr=('0.5', 'spotsize_yx_radius_pxl'),
        sigma_y_min_max_expr=('0.5', 'spotsize_yx_radius_pxl'),
        sigma_z_min_max_expr=('0.5', 'spotsize_z_radius_pxl'),
        A_min_max_expr=('0.0', 'spotsize_A_max'),
        B_min_max_expr=('spot_B_min', 'inf'),
        sigma_x_guess_expr='spotsize_initial_radius_yx_pixel',
        sigma_y_guess_expr='spotsize_initial_radius_yx_pixel',
        sigma_z_guess_expr='spotsize_initial_radius_z_pixel',
        A_guess_expr='spotsize_A_max',
        B_guess_expr='spotsize_surface_median',  
    ):
    """Run spotFIT (fitting 3D gaussian curves) and get the related features

    Parameters
    ----------
    kernel : spotmax.core.SpotFIT
        Initialized SpoFIT class defined in spotmax.core.SpotFIT
    spots_img : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray
        Input 2D or 3D image.
    df_spots : pandas.DataFrame
        DataFrame with Cell_ID as index and the columns {'z', 'y', 'x'} which 
        are the coordinates of the spots in `spots_img` to fit. 
    zyx_voxel_size : sequence of 3 floats (z, y, x), optional
        Voxel size in μm/pixel. If `None` this will be initialized to 
        (1, 1, 1). Default is None
    zyx_spot_min_vol_um : (z, y, x) sequence of floats, optional
        Rough estimation of the expected spot radii in z, y, and x direction
        in μm. The values are used to build starting masks for the spotSIZE step.
        The spotSIZE step will determine the extent of each spot, i.e., the pixels 
        that will be the input for the fitting procedure. 
        If `None` this will be calculated from `spots_zyx_radii_pxl` and 
        `zyx_voxel_size`. Default is None
    spots_zyx_radii_pxl : (z, y, x) sequence of floats, optional
        Minimum distance between peaks in z, y, and x direction in pixels. 
        Used only if `drop_peaks_too_close` is True. If None and 
        `drop_peaks_too_close` is True then this will be calculated from 
        `zyx_spot_min_vol_um` and `zyx_voxel_size`. Default is None
    delta_tol : (z, y, x) sequence of floats, optional
        If not None, these values will be used to enlarge the segmented objects. 
        It will enable correct fitting of those spots whose intensities 
        bleed outside of the object (e.g., single cell). Default is None
    rp : list of skimage.measure.RegionProperties, optional
        If not None, list of properties of objects in `lab` as returned by 
        skimage.measure.regionprops(lab). If None, this will be computed 
        with `skimage.measure.regionprops(lab)`. Default is None
    lab : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        Optional input segmentation image with the masks of the objects, i.e. 
        single cells. Default is None. 
    frame_i : int, optional
        Frame index in timelapse data. Default is 0
    ref_ch_mask_or_labels : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        Instance or semantic segmentation of the reference channel. 
        Default is None
    spots_masks_check_merge : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        If not `None`, for each pair of touching spots in this array check 
        if one gaussian peak fits better than two. If yes, merge the two spots 
        before running final fitting procedure.
    drop_peaks_too_close : bool, optional
        If True, when two or more peaks are within the same ellipsoid with 
        radii = `spots_zyx_radii_pxl` only the brightest peak is kepts. 
        The center of the peaks is the one determined by the fitting procedure. 
        Default is False
    return_df : bool, optional
        If True, returns a pandas DataFrame. More details on the returned 
        items section below. Default is False
    use_gpu : bool, optional
        If True, some steps will run on the GPU, potentially speeding up the 
        computation. Default is False
    show_progress : bool, optional
        If True, display progressbars. Default is False
    verbose : bool, optional
        If True, additional information text will be printed to the terminal. 
        Default is True
    logger_func : callable, optional
        Function used to print or log process information. Default is print
    custom_combined_measurements : dict or None, optional
        If not None, this is a dictionary of new column names as keys and 
        mathematical expressions that combines standard single-spot features. 
        The expression can be any text that can be evaluated by `pandas.eval`. 
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    max_number_pairs_check_merge : int, optional
        If `spots_masks_check_merge` is not None, SpotMAX will test  
        `max_number_pairs_check_merge` number of pairs to check if 
        they require merging. The pairs are determined from all those peaks 
        that lie on the same spot mask and are within `spots_zyx_radii_pxl` 
        distance between each other. To test all pairs set this value to -1. 
        Default is 11 (just a random lucky number :D) 
    xy_center_half_interval_val : float, optional
        Half interval width for bounds on x and y center coordinates during fit. 
        Default is 0.1
    z_center_half_interval_val : float, optional
        Half interval width for bounds on z center coordinate during fit. 
        Default is 0.2
    sigma_x_min_max_expr : 2-tuple of str, optional
        Expressions to evaluate with `pandas.eval` to determine minimum and 
        maximum values for bounds on `sigma_x_fit` fitting paramter. The expression 
        can be any text that can be evaluated by `pandas.eval`. 
        Default is ('0.5', 'spotsize_yx_radius_pxl').
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    sigma_y_min_max_expr : 2-tuple of str, optional
        Expressions to evaluate with `pandas.eval` to determine minimum and 
        maximum values for bounds on `sigma_y_fit` fitting paramter. The expression 
        can be any text that can be evaluated by `pandas.eval`. 
        Default is ('0.5', 'spotsize_yx_radius_pxl').
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
        Default is ('0.5', 'spotsize_yx_radius_pxl').
    sigma_z_min_max_expr : 2-tuple of str, optional
        Expressions to evaluate with `pandas.eval` to determine minimum and 
        maximum values for bounds on `sigma_z_fit` fitting paramter. The expression 
        can be any text that can be evaluated by `pandas.eval`. 
        Default is ('0.5', 'spotsize_z_radius_pxl').
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    A_min_max_expr : 2-tuple of str, optional
        Expressions to evaluate with `pandas.eval` to determine minimum and 
        maximum values for bounds on `A_fit` (peak amplitude) fitting paramter. 
        The expression can be any text that can be evaluated by `pandas.eval`. 
        Default is ('0.0', 'spotsize_A_max').
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    B_min_max_expr : 2-tuple of str, optional
        Expressions to evaluate with `pandas.eval` to determine minimum and 
        maximum values for bounds on `B_fit` (background) fitting paramter. 
        The expression can be any text that can be evaluated by `pandas.eval`. 
        Default is ('spot_B_min', 'inf').
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    sigma_x_guess_expr : str, optional
        Expressions to evaluate with `pandas.eval` to determine the initial 
        guess for the `sigma_x_fit` fitting paramter. 
        The expression can be any text that can be evaluated by `pandas.eval`. 
        Default is 'spotsize_initial_radius_yx_pixel'.
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    sigma_y_guess_expr : str, optional
        Expressions to evaluate with `pandas.eval` to determine the initial 
        guess for the `sigma_y_fit` fitting paramter. 
        The expression can be any text that can be evaluated by `pandas.eval`. 
        Default is 'spotsize_initial_radius_yx_pixel'.
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    sigma_z_guess_expr : str, optional
        Expressions to evaluate with `pandas.eval` to determine the initial 
        guess for the `sigma_z_fit` fitting paramter. 
        The expression can be any text that can be evaluated by `pandas.eval`. 
        Default is 'spotsize_initial_radius_z_pixel'.
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    A_guess_expr : str, optional
        Expressions to evaluate with `pandas.eval` to determine the initial 
        guess for the `A_fit` (amplitude) fitting paramter. 
        The expression can be any text that can be evaluated by `pandas.eval`. 
        Default is 'spotsize_A_max'.
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    B_guess_expr : str, optional
        Expressions to evaluate with `pandas.eval` to determine the initial 
        guess for the `B_fit` (background) fitting paramter. 
        The expression can be any text that can be evaluated by `pandas.eval`. 
        Default is 'spotsize_surface_median'.
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
        
    Returns
    -------
    keys : list of 2-tuple (int, int) 
        List of keys that can be used to concatenate the 
        dataframes with 
        `pandas.concat(dfs_spots_spotfit, keys=keys, names=['frame_i', 'Cell_ID'])`
        Returned only if `return_df` is False
    
    dfs_spots_spotfit : list of pandas.DataFrames
        List of DataFrames with additional spotFIT features columns 
        for each frame and ID of the segmented objects in `lab`. 
        If `drop_peaks_too_close` is True, each DataFrame in this list 
        will contain only the valid spots. Returned only if `return_df` is False
    
    dfs_spots_spotfit_iter0 : list of pandas.DataFrames
        List of DataFrames with additional spotFIT features columns 
        for each frame and ID of the segmented objects in `lab`. No matter 
        the value of `drop_peaks_too_close`, each DataFrame in this list 
        will contain all the input spots. Returned only if `return_df` is False
    
    df_spotfit : pandas.DataFrame 
        DataFrame with Cell_ID as index and all the spotFIT features as columns.
        If `drop_peaks_too_close` is True this DataFrame will contain only 
        valid spots. Returned only if `return_df` is True 
    
    df_spotfit_iter0 : pandas.DataFrame 
        DataFrame with Cell_ID as index and all the spotFIT features as columns.
        No matter the value of `drop_peaks_too_close` this DataFrame will 
        contain all the input spots. Returned only if `return_df` is True 
        
    See also
    --------
    `skimage.measure.regionprops <https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops>`__
    """    
    if lab is None:
        lab = np.ones(spots_img.shape, dtype=np.uint8)

    if rp is None:
        rp = skimage.measure.regionprops(lab)
    
    if delta_tol is None:
        delta_tol = transformations.get_expand_obj_delta_tolerance(
            spots_zyx_radii_pxl
        )
    
    if zyx_voxel_size is None:
        zyx_voxel_size = np.array((1, 1, 1))
    
    if zyx_spot_min_vol_um is None:
        zyx_spot_min_vol_um = np.array(
            [v*s for v, s in zip(spots_zyx_radii_pxl, zyx_voxel_size)]
        )
    
    if spots_zyx_radii_pxl is None and drop_peaks_too_close:
        spots_zyx_radii_pxl = np.array(
            [v/s for v, s in zip(zyx_spot_min_vol_um, zyx_voxel_size)]
        )
    
    dfs_spots_spotfit = []
    dfs_spots_spotfit_iter0 = []
    keys = []
    
    if show_progress:
        desc = 'Measuring spots'
        pbar = tqdm(
            total=len(rp), ncols=100, desc=desc, position=3, leave=False
        )
    # df_spots_spotfit = df_spots.drop(columns=['spot_mask'], errors='ignore')
    df_spots_spotfit = df_spots.copy()
    non_spotfit_cols = df_spots_spotfit.columns.to_list()
    filtered_spots_info = defaultdict(dict)
    for obj in rp:
        if obj.label not in df_spots.index:
            continue
        expanded_obj = transformations.get_expanded_obj(obj, delta_tol, lab)
        df_spots_obj = df_spots_spotfit.loc[obj.label].copy()
        start_num_spots = len(df_spots_obj)
        filtered_spots_info[obj.label]['start_num_spots'] = start_num_spots
        i = 0
        while True:                
            kernel.set_args(
                expanded_obj, 
                spots_img, 
                df_spots_obj, 
                zyx_voxel_size, 
                zyx_spot_min_vol_um, 
                xy_center_half_interval_val=xy_center_half_interval_val, 
                z_center_half_interval_val=z_center_half_interval_val, 
                sigma_x_min_max_expr=sigma_x_min_max_expr,
                sigma_y_min_max_expr=sigma_y_min_max_expr,
                sigma_z_min_max_expr=sigma_z_min_max_expr,
                A_min_max_expr=A_min_max_expr,
                B_min_max_expr=B_min_max_expr,
                sigma_x_guess_expr=sigma_x_guess_expr,
                sigma_y_guess_expr=sigma_y_guess_expr,
                sigma_z_guess_expr=sigma_z_guess_expr,
                A_guess_expr=A_guess_expr,
                B_guess_expr=B_guess_expr,
                spots_masks_check_merge=spots_masks_check_merge,
                max_number_pairs_check_merge=max_number_pairs_check_merge,
                ref_ch_mask_or_labels=ref_ch_mask_or_labels,
                use_gpu=use_gpu, 
                logger_func=logger_func, 
                show_progress=show_progress
            )
            kernel.fit()
            prev_num_spots = len(kernel.df_spotFIT_ID)
            
            if custom_combined_measurements is not None:
                kernel.add_custom_combined_features(
                    **custom_combined_measurements
                )
            
            if i == 0:
                # Store all features at first iteration
                dfs_spots_spotfit_iter0.append(kernel.df_spotFIT_ID.copy())
            
            if not drop_peaks_too_close: 
                num_spots = prev_num_spots
                break
            
            df_spotfit = kernel.df_spotFIT_ID
            
            fit_coords = df_spotfit[ZYX_FIT_COLS].to_numpy()
            fit_coords_int = np.round(fit_coords).astype(int)
            intensities = spots_img[tuple(fit_coords_int.transpose())]
            
            valid_fit_coords = filters.filter_valid_points_min_distance(
                fit_coords, spots_zyx_radii_pxl, intensities=intensities, 
            )
            
            if 'do_not_drop' in df_spotfit.columns:
                undroppable_coords = (
                    df_spotfit[df_spotfit['do_not_drop'] > 0]
                    [ZYX_FIT_COLS].to_numpy()
                )
                valid_fit_coords = np.unique(
                    np.vstack((valid_fit_coords, undroppable_coords)), 
                    axis=0
                )

            num_spots = len(valid_fit_coords)
            if num_spots == prev_num_spots:
                # All spots are valid --> break loop
                break
            
            if num_spots == 0:
                kernel.df_spotFIT_ID = kernel.df_spotFIT_ID[0:0]
                break
            
            index_names = kernel.df_spotFIT_ID.index.names
            filter_zyx_index = pd.MultiIndex.from_arrays(
                tuple(valid_fit_coords.transpose())
            )
            df_spots_obj = (
                kernel.df_spotFIT_ID.reset_index()
                .set_index(ZYX_FIT_COLS)
                .loc[filter_zyx_index]
                .reset_index()
                .set_index(index_names)
                .sort_index()
                [non_spotfit_cols]
            )
            prev_num_spots = num_spots      
            i += 1

        filtered_spots_info[obj.label]['end_num_spots'] = num_spots
        filtered_spots_info[obj.label]['num_iter'] = i
        
        dfs_spots_spotfit.append(kernel.df_spotFIT_ID)
        keys.append((frame_i, obj.label))
        if show_progress:
            pbar.update()
    if show_progress:
        pbar.close()
    
    _log_filtered_number_spots(
        verbose, frame_i, filtered_spots_info, logger_func, 
        category='valid spots according to spotFIT'
    )
    if not return_df:
        return keys, dfs_spots_spotfit, dfs_spots_spotfit_iter0
    else:
        df_spots_spotfit = pd.concat(
            dfs_spots_spotfit, keys=keys, names=['frame_i', 'Cell_ID']
        )
        df_spots_spotfit_iter0 = pd.concat(
            dfs_spots_spotfit_iter0, keys=keys, names=['frame_i', 'Cell_ID']
        )
        return df_spots_spotfit, df_spots_spotfit_iter0

def filter_spots_from_features_thresholds(
        df_features: pd.DataFrame, 
        features_thresholds: dict, 
        is_spotfit=False,
        frame_i=0,
        debug=False,
        logger_func=print, 
        verbose=True,   
    ):
    """Filter valid spots based on features ranges

    Parameters
    ----------
    df_features : pandas.DataFrame
        Pandas DataFrame with 'spot_id' or ('Cell_ID', 'spot_id') as index and 
        the features as columns.
    features_thresholds : dict
        A dictionary of features and thresholds to use for filtering. The 
        keys are the feature names that mush coincide with one of the columns'
        names. The values are a tuple of `(min, max)` thresholds.
        For example, for filtering spots that have the t-statistic of the 
        t-test spot vs reference channel > 0 and the p-value < 0.025 
        (i.e. spots are significantly brighter than reference channel) 
        we pass the following dictionary:
		
        .. code-block:: python
        
            features_thresholds = {
                'spot_vs_ref_ch_ttest_pvalue': (None,0.025),
                'spot_vs_ref_ch_ttest_tstat': (0, None)
            }
		
        where `None` indicates the absence of maximum or minimum.
    is_spotfit : bool, optional
        If False, features ending with '_fit' will be ignored. Default is False
    verbose : bool, optional
        If True, additional information text will be printed to the terminal. 
        Default is True
    debug : bool, optional
        If True, it can be used for debugging like printing additional 
        internal steps or visualize intermediate results.
    logger_func : callable, optional
        Function used to print or log process information. Default is print

    Returns
    -------
    pandas.DataFrame
        The filtered DataFrame
    """
    if verbose:
        print('')
        logger_func('Filtering valid spots...')
    
    if df_features.empty:
        return df_features
    
    df_filtered = filters.filter_df_from_features_thresholds(
        df_features, 
        features_thresholds,
        is_spotfit=is_spotfit, 
        debug=debug,
        logger_func=logger_func
    )
    if verbose:
        _log_filtered_number_spots_from_dfs(
            df_features, df_filtered, frame_i, logger_func=logger_func
        )
    return df_filtered

def _log_filtered_number_spots_from_dfs(
        start_df, end_df, frame_i, logger_func=print, 
        category='valid spots based on spotFIT features', 
        objects_name='spots', index_names=('Cell_ID', 'spot_id')
    ):
    start_num_spots_df = (
        start_df.reset_index()
        [list(index_names)]
        .groupby('Cell_ID')
        .count()
        .rename(columns={index_names[-1]: 'Before filtering'})
    )
    end_num_spots_df = (
        end_df.reset_index()
        [list(index_names)]
        .groupby('Cell_ID')
        .count()
        .rename(columns={index_names[-1]: 'After filtering'})
    )
    
    dfs = [start_num_spots_df, end_num_spots_df]
    start_end_df = pd.concat(dfs, axis=1).fillna(0)
    
    different_nums_mask = (
        start_end_df['Before filtering'] 
        != start_end_df['After filtering'] 
    )
    different_nums_df = start_end_df[different_nums_mask]
    
    if different_nums_df.empty:
        info = f'All {objects_name} are valid'
    else:
        info = different_nums_df
    
    print('')
    if frame_i is not None:
        header = (
            f'Frame n. {frame_i+1}: number of {objects_name} '
            f'after filtering {category}:'
        )
    else:
        header = (
            f'Number of {objects_name} after filtering {category}:'
        )
    print('*'*len(header))
    logger_func(f'{header}\n\n{info}')
    print('-'*len(header))

class Model:
    """End-to-end spot detection and optional feature extraction pipeline.

    Parameters
    ----------
    spots_zyx_radii_pxl : sequence of 3 floats
        Expected spot radii in z, y, and x in pixels. Used for detection,
        feature measurement, and fitting.
    zyx_voxel_size : sequence of 3 floats, optional
        Voxel size in z, y, and x in micrometers per pixel. Defaults to
        (1, 1, 1) through the underlying pipeline functions.
    zyx_spot_min_vol_um : sequence of 3 floats, optional
        Minimum spot radii in micrometers for SpotSIZE. If omitted, these are
        derived from spots_zyx_radii_pxl and zyx_voxel_size.
    segmentation_kwargs : dict, optional
        Keyword arguments for `pipe.spots_semantic_segmentation`. The default
        thresholding method is threshold_li. Preprocessing settings such as
        gauss_sigma, do_remove_hot_pixels, and do_sharpen are also read here.
    detection_kwargs : dict, optional
        Additional keyword arguments for spot_detection, such as
        detection_method and spot_footprint.
    features_kwargs : dict, optional
        Additional keyword arguments for spots_calc_features_and_filter.
    spotfit_kwargs : dict, optional
        Additional keyword arguments for spotfit.
    spotfit_check_merge : bool, optional
        If True, spotfit will use the mask from the segmentation step to 
        check if spots have been merged (according to fit). This is ignored 
        if `spotfit_kwargs['spots_masks_check_merge']` is not None.
        Default is False.
    spotfit_kernel : spotmax.core.SpotFIT, optional
        Initialized SpotFIT kernel. A new kernel is created for each call to
        predict when this is omitted.
    frame_i : int, optional
        Frame index used by the feature and fitting pipeline. Defaults to 0.

    Notes
    -----
    By default, predict returns only the coordinate DataFrame. If masks or
    preprocessed images are requested, predict returns a tuple containing the
    DataFrame and a dictionary with the requested outputs.
    """

    def __init__(
            self,
            spots_zyx_radii_pxl,
            zyx_voxel_size=None,
            zyx_spot_min_vol_um=None,
            segmentation_kwargs=None,
            detection_kwargs=None,
            features_kwargs=None,
            spotfit_kwargs=None,
            spotfit_kernel=None,
            spotfit_check_merge=False,
            frame_i=0
        ):
        self.spots_zyx_radii_pxl = np.asarray(
            spots_zyx_radii_pxl, dtype=float
        )
        if self.spots_zyx_radii_pxl.shape != (3,):
            raise ValueError(
                "spots_zyx_radii_pxl must contain three values (z, y, x)."
            )

        self.zyx_voxel_size = (
            None if zyx_voxel_size is None
            else np.asarray(zyx_voxel_size, dtype=float)
        )
        if (self.zyx_voxel_size is not None
                and self.zyx_voxel_size.shape != (3,)):
            raise ValueError("zyx_voxel_size must contain three values.")

        self.zyx_spot_min_vol_um = (
            None if zyx_spot_min_vol_um is None
            else np.asarray(zyx_spot_min_vol_um, dtype=float)
        )
        if (self.zyx_spot_min_vol_um is not None
                and self.zyx_spot_min_vol_um.shape != (3,)):
            raise ValueError("zyx_spot_min_vol_um must contain three values.")

        self.segmentation_kwargs = dict(segmentation_kwargs or {})
        self.detection_kwargs = dict(detection_kwargs or {})
        self.features_kwargs = dict(features_kwargs or {})
        self.spotfit_kwargs = dict(spotfit_kwargs or {})
        self.spotfit_kernel = spotfit_kernel
        self.spotfit_check_merge = spotfit_check_merge
        self.frame_i = frame_i

    @staticmethod
    def _get_segmentation_mask(result, image_shape):
        if isinstance(result, np.ndarray):
            return result

        if not isinstance(result, dict):
            raise TypeError(
                "spots_semantic_segmentation must return an array or dict."
            )

        candidates = []
        for key, value in result.items():
            if key == "input_image" or "prediction" in key.lower():
                continue
            if isinstance(value, np.ndarray):
                candidates.append((key, value))

        matching = [
            (key, value) for key, value in candidates
            if value.shape == image_shape
        ]
        if len(matching) == 1:
            return matching[0][1]
        if len(matching) == 0 and len(candidates) == 1:
            return candidates[0][1]

        candidate_names = [key for key, _ in candidates]
        raise ValueError(
            "Expected one segmentation mask with the input image shape; "
            f"found candidate outputs {candidate_names}. Configure one "
            "segmentation method in segmentation_kwargs."
        )

    @staticmethod
    def _without_reserved_kwargs(kwargs, reserved):
        return {
            key: value for key, value in kwargs.items()
            if key not in reserved
        }

    def predict(
            self,
            image,
            lab=None,
            return_features=False,
            return_masks=False,
            return_preprocessed=False
        ):
        """Detect spots and optionally calculate features and return outputs.

        Parameters
        ----------
        image : (Y, X) or (Z, Y, X) numpy.ndarray
            Raw spots-channel image.
        lab : (Y, X) or (Z, Y, X) numpy.ndarray, optional
            Object labels used to constrain detection and organize output rows.
        return_features : bool, optional
            If True, compute detection features and SpotFIT features and include
            them in the returned DataFrame. Defaults to False.
        return_masks : bool, optional
            If True, return the selected segmentation mask and fitted spot mask
            and labels. Spot fitting is run to generate the fitted labels.
            Defaults to False.
        return_preprocessed : bool, optional
            If True, also return the preprocessed image and the detection image.
            The detection image is the DoG-sharpened image when do_sharpen is
            enabled; otherwise it is the preprocessed image. Defaults to False.

        Returns
        -------
        pandas.DataFrame or tuple
            By default, a DataFrame with spot coordinates. If return_features
            is True, the DataFrame also contains detection and SpotFIT features.
            If return_masks or return_preprocessed is True, returns
            (dataframe, extras), where extras is a dictionary containing the
            requested "masks" and/or "preprocessed_images" entries.
        """
        input_is_2d = image.ndim == 2
        if image.ndim not in (2, 3):
            raise ValueError("image must be a 2D or 3D array.")
        if input_is_2d:
            image = image[np.newaxis]
        if lab is not None and lab.ndim == 2:
            lab = np.repeat(lab[np.newaxis], image.shape[0], axis=0)

        segmentation_kwargs = dict(self.segmentation_kwargs)
        do_remove_hot_pixels = segmentation_kwargs.get(
            "do_remove_hot_pixels", False
        )
        gauss_sigma = segmentation_kwargs.get("gauss_sigma", 0.0)
        do_sharpen = segmentation_kwargs.get("do_sharpen", False)
        use_gpu = segmentation_kwargs.get("use_gpu", False)
        logger_func = segmentation_kwargs.get("logger_func", print)

        preprocessed_image = preprocess_image(
            image,
            lab=lab,
            do_remove_hot_pixels=do_remove_hot_pixels,
            gauss_sigma=gauss_sigma,
            use_gpu=use_gpu,
            do_sharpen=False,
            logger_func=logger_func
        )
        if do_sharpen:
            detection_image = filters.DoG_spots(
                preprocessed_image,
                self.spots_zyx_radii_pxl,
                use_gpu=use_gpu,
                logger_func=logger_func,
                lab=lab
            )
        else:
            detection_image = preprocessed_image

        segmentation_kwargs = self._without_reserved_kwargs(
            segmentation_kwargs,
            reserved={
                "do_remove_hot_pixels", "gauss_sigma", "do_sharpen",
                "do_preprocess", "do_try_all_thresholds", "return_only_segm",
                "keep_input_shape", "lab", "raw_image",
                "spots_zyx_radii_pxl"
            }
        )
        if segmentation_kwargs.get("thresholding_method") is None:
            segmentation_kwargs["thresholding_method"] = "threshold_li"
        segmentation_kwargs.update({
            "lab": lab,
            "raw_image": image,
            "spots_zyx_radii_pxl": self.spots_zyx_radii_pxl,
            "do_preprocess": False,
            "do_try_all_thresholds": False,
            "return_only_segm": False,
            "keep_input_shape": True
        })
        segmentation_result = spots_semantic_segmentation(
            detection_image, **segmentation_kwargs
        )
        segmentation_mask = self._get_segmentation_mask(
            segmentation_result, detection_image.shape
        )
        if input_is_2d and segmentation_mask.ndim == 3:
            segmentation_mask = segmentation_mask[0]

        detection_kwargs = self._without_reserved_kwargs(
            self.detection_kwargs,
            reserved={
                "image", "spots_segmantic_segm", "spots_zyx_radii_pxl",
                "return_spots_mask", "lab", "return_df"
            }
        )
        df_coords, _ = spot_detection(
            detection_image,
            spots_segmantic_segm=segmentation_mask,
            spots_zyx_radii_pxl=self.spots_zyx_radii_pxl,
            return_spots_mask=False,
            lab=lab,
            return_df=True,
            **detection_kwargs
        )

        output_df = df_coords
        fitted_df = None
        need_fit = return_features or return_masks
        if need_fit and len(df_coords):
            features_kwargs = self._without_reserved_kwargs(
                self.features_kwargs,
                reserved={
                    "image", "spots_zyx_radii_pxl", "df_spots_coords",
                    "frame_i", "sharp_spots_image", "lab", "zyx_voxel_size"
                }
            )
            features_kwargs.setdefault("show_progress", False)
            features_kwargs.setdefault("verbose", False)
            keys, _, dfs_spots_gop = spots_calc_features_and_filter(
                image,
                self.spots_zyx_radii_pxl,
                df_coords,
                frame_i=self.frame_i,
                sharp_spots_image=detection_image,
                lab=lab,
                zyx_voxel_size=self.zyx_voxel_size,
                **features_kwargs
            )

            if dfs_spots_gop:
                df_spots_all = pd.concat(
                    dfs_spots_gop,
                    keys=keys,
                    names=["frame_i", "Cell_ID"]
                )
                if len(df_spots_all):
                    df_spots = df_spots_all.xs(
                        self.frame_i, level="frame_i", drop_level=True
                    )
                else:
                    df_spots = df_spots_all.droplevel("frame_i")
            else:
                df_spots = df_coords.iloc[0:0].copy()

            if len(df_spots):
                fit_kwargs = self._without_reserved_kwargs(
                    self.spotfit_kwargs,
                    reserved={
                        "kernel", "spots_img", "df_spots", "zyx_voxel_size",
                        "zyx_spot_min_vol_um", "spots_zyx_radii_pxl",
                        "lab", "frame_i", "return_df"
                    }
                )
                kernel = (
                    self.spotfit_kernel
                    if self.spotfit_kernel is not None
                    else core.SpotFIT()
                )
                fit_kwargs.setdefault("show_progress", False)
                fit_kwargs.setdefault("verbose", False)
                if (
                        fit_kwargs.get('spots_masks_check_merge') is None
                        and self.spotfit_check_merge
                    ):
                    fit_kwargs['spots_masks_check_merge'] = (
                        segmentation_mask
                    )

                df_fit_all, _ = spotfit(
                    kernel,
                    image,
                    df_spots,
                    zyx_voxel_size=self.zyx_voxel_size,
                    zyx_spot_min_vol_um=self.zyx_spot_min_vol_um,
                    spots_zyx_radii_pxl=self.spots_zyx_radii_pxl,
                    lab=lab,
                    frame_i=self.frame_i,
                    return_df=True,
                    **fit_kwargs
                )
                if (isinstance(df_fit_all.index, pd.MultiIndex)
                        and "frame_i" in df_fit_all.index.names):
                    fitted_df = df_fit_all.xs(
                        self.frame_i, level="frame_i", drop_level=True
                    )
                else:
                    fitted_df = df_fit_all
            else:
                fitted_df = df_spots

            if return_features:
                output_df = fitted_df

        extras = {}
        if return_masks:
            if fitted_df is not None and len(fitted_df):
                _, fitted_mask, fitted_labels = utils.get_spotfit_image(
                    fitted_df, image.shape
                )
            else:
                fitted_mask = np.zeros(image.shape, dtype=bool)
                fitted_labels = np.zeros(image.shape, dtype=np.uint32)
            if input_is_2d:
                fitted_mask = fitted_mask[0]
                fitted_labels = fitted_labels[0]

            extras["masks"] = {
                "segmentation": segmentation_mask,
                "fitted": {
                    "mask": fitted_mask,
                    "labels": fitted_labels
                }
            }

        if return_preprocessed:
            if input_is_2d:
                preprocessed_image = preprocessed_image[0]
                detection_image = detection_image[0]
            extras["preprocessed_images"] = {
                "preprocessed": preprocessed_image,
                "detection": detection_image
            }

        if extras:
            return output_df, extras
        return output_df
