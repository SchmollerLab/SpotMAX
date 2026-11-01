import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from cellacdc.plot import imshow

import spotmax as sm

spots_radii = np.array([4, 6, 6])

# Generate synthetic data
spots_img, spots_gt_mask, spots_coords, df_spots_coords  = (
    sm.data.synthetic_spots(
        num_spots=20,
        shape=(25, 256, 256), 
        spots_radii=spots_radii, 
        noise_scale=0.05,
        noise_shape=0.03, 
        return_df=True,
        rng_seed=11
    )
)

# Initialise model
from spotmax import Model

spots_zyx_radii_pxl = spots_radii

# Keyword arguments for `spotmax.pipe.spots_semantic_segmentation`
segmentation_kwargs = {
    'thresholding_method': 'threshold_li',
    'do_sharpen': True
}

# Keyword arguments for `spotmax.pipe.spot_detection`
detection_kwargs = {}

# Keyword arguments for `spotmax.pipe.spots_calc_features_and_filter`
gop_filtering_thresholds = sm.config.get_features_thresholds_filter(
    'spot_vs_backgr_effect_size_glass, 0.2'
)
features_kwargs = {
    'gop_filtering_thresholds': gop_filtering_thresholds,
    'optimise_for_high_spot_density': True,
    'show_progress': False,
    'verbose': True,
}

# Keyword arguments for `spotmax.pipe.spotfit`
spotfit_kwargs = {
    'verbose': True
}

model = Model(
    spots_zyx_radii_pxl,
    zyx_voxel_size=None,
    segmentation_kwargs=segmentation_kwargs,
    detection_kwargs=detection_kwargs,
    features_kwargs=features_kwargs,
    spotfit_kwargs=spotfit_kwargs,
    spotfit_check_merge=True,
)

# Run the prediction
df_spots_spotfit, extras = model.predict(
    spots_img,
    return_features=True,
    return_masks=True,
    return_preprocessed=True,
)

spots_pred_mask = extras['masks']['segmentation']
spotfit_mask = extras['masks']['fitted']['mask']
spotfit_labels = extras['masks']['fitted']['labels']
preprocessed_image = extras['preprocessed_images']['preprocessed']
detection_image = extras['preprocessed_images']['detection']

columns = [
    'z', 'y', 'x', 'spot_center_raw_intensity', 
    'spot_vs_backgr_effect_size_glass'
]

print(df_spots_spotfit[columns].head(10))

display_images = {
    'Raw image': spots_img,
    'Preprocessed image': preprocessed_image,
    'Detection image': detection_image,
    'Spots semantic segmentation': spots_pred_mask,
    'SpotFIT masks': spotfit_mask,
    'SpotFIT labels': spotfit_labels
}

imshow(
    display_images,
    points_coords_df=df_spots_spotfit,
    window_title='SpotMAX - Synthetic image simple - Results',
    annotate_labels_idxs=[5],
    max_ncols=3
)