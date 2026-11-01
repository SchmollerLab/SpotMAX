import os

import numpy as np
import pandas as pd

from cellacdc.data import _Data
from cellacdc import load

from . import ZYX_LOCAL_COLS, ZYX_GLOBAL_COLS
from . import data_path, core


def _generate_syntetic_spots_img(img, zyx_spots, zyx_sigmas):
    model = core.GaussianModel()
    SizeZ, SizeY, SizeX = img.shape
    zz, yy, xx = np.ogrid[0:SizeZ, 0:SizeY, 0:SizeX]
    num_spots = len(zyx_spots)

    for zyx_spot in zyx_spots:
        coeffs = (*zyx_spot, *zyx_sigmas, 1)
        spot = model.func(zz, yy, xx, coeffs)
        img += spot
    
    return img

def get_random_coords(shape, num_points, rng_seed=11, rng=None):
    if len(shape) == 2:
        shape = (1, *shape)
    
    if rng is None:
        rng = np.random.default_rng(rng_seed)
    
    Z, Y, X = shape
    
    zz = rng.integers(0, Z, num_points)
    yy = rng.integers(0, Y, num_points)
    xx = rng.integers(0, X, num_points)
    
    zyx_coords = np.column_stack((zz, yy, xx))
    return zyx_coords
    
def add_random_coords_df(
        df, shape, num_points, lab=None, rng_seed=11, rng=None
    ):
    zyx_coords = get_random_coords(
        shape, num_points, rng_seed=rng_seed, rng=rng
    )
    
    columns = [*ZYX_GLOBAL_COLS, *ZYX_LOCAL_COLS]
    if lab is None:
        zyx_coords = np.hstack((zyx_coords, zyx_coords))
        start_new_spot_id = df.index.get_level_values(1).max() + 1
        stop_new_spot_id = start_new_spot_id + num_points
        new_spot_ids = range(start_new_spot_id, stop_new_spot_id)
        idx_names = df.index.names
        index = pd.MultiIndex.from_product(([1], new_spot_ids), names=idx_names)
        df_new_coords = pd.DataFrame(
            data=zyx_coords, index=index, columns=columns
        )
    
    df_with_new_coords = pd.concat([df, df_new_coords])
    return df_with_new_coords
        
def add_point_df_spots(df, point, ID, spot_id):
    columns = [*ZYX_GLOBAL_COLS, *ZYX_LOCAL_COLS]
    idx_names = df.index.names
    index = pd.MultiIndex.from_product(([ID], [spot_id]), names=idx_names)
    data = np.array([[*point, *point]])
    df_point = pd.DataFrame(data=data, index=index, columns=columns)
    df = pd.concat([df, df_point])
    return df

def synthetic_spots(
        num_spots=20,
        shape=(25, 256, 256), 
        spots_radii=(2, 4, 4),
        noise_scale=0.05, 
        noise_shape=0.03, 
        return_df=False,
        rng_seed=11
    ):
    """_summary_

    Parameters
    ----------
    num_spots : int, optional
        Number of spots in the image. Default is 20
    shape : (SizeY, SizeX) or (SizeZ, SizeY, SizeX) tuple of ints, optional
        Shape of the image. Default is (25, 256, 256)
    spots_radii : (y, x) or (z, y, x) tuple of ints, optional
        Radii of the spots in pixels. Default is (2, 4, 4)
    noise_scale : float, optional
        Scale of the gamma distribution used additive noise. Default is 0.05
    noise_shape : float, optional
        Shape of the gamma distribution used additive noise. Default is 0.03
    rng_seed : int, optional
        Seed for random generator to ensure reproducibility. Default is 11

    Returns
    -------
    img : (Y, X) or (Z, Y, X) numpy.ndarray of floats in the [0, 1] range
        Generated image with spots modelled by a gaussian function.
    
    mask : (Y, X) or (Z, Y, X) numpy.ndarray of bools
        Semantic segmentation masks of the generated spots.
    
    zyx_spots_coords : (`num_spots`, 2) or (`num_spots`, 3) numpy.ndarray of ints
        The coordinates of the spots
    """    
    rng = np.random.default_rng(rng_seed)
    
    if len(shape) == 2:
        shape = (1, *shape)
    
    SizeZ, SizeY, SizeX = shape
    
    if len(spots_radii) == 2:
        spots_radii = (1, *spots_radii)
              
    sz, sy, sx = spots_radii
    if SizeZ > 1:
        low_z = int(np.ceil(sz))
        high_z = int(np.floor(SizeZ-sz))
        zz_spots = rng.integers(low_z, high_z, num_spots)
    else:
        zz_spots = [0]*num_spots  
    
    low_y = int(np.ceil(sy))
    high_y = int(np.floor(SizeY-sy))
    yy_spots = rng.integers(low_y, high_y, num_spots)
    
    low_x = int(np.ceil(sy))
    high_x = int(np.floor(SizeY-sy))
    xx_spots = rng.integers(low_x, high_x, num_spots)
    
    zyx_spots_coords = np.column_stack((zz_spots, yy_spots, xx_spots))
    
    img = np.zeros(shape)
    spheroid = core.Spheroid(img, show_progress=False)
    
    mask = spheroid.get_spots_mask(
        0, (1, 1, 1), spots_radii, zyx_spots_coords, 
    )
    
    zyx_sigmas = np.array(spots_radii)/2
    img = _generate_syntetic_spots_img(img, zyx_spots_coords, zyx_sigmas)
    
    noise = rng.gamma(noise_scale, noise_shape, size=shape)
    
    img += noise
    img /= img.max()
    
    if not return_df:
        return img, mask, zyx_spots_coords
    
    df_spots_coords = pd.DataFrame({
        'Cell_ID': [1]*len(zyx_spots_coords),
        'spot_id': np.arange(1, len(zyx_spots_coords)+1),
        'z': zyx_spots_coords[:, 0],
        'y': zyx_spots_coords[:, 1],
        'x': zyx_spots_coords[:, 2],
        'z_local': zyx_spots_coords[:, 0],
        'y_local': zyx_spots_coords[:, 1],
        'x_local': zyx_spots_coords[:, 2],
    }).set_index(['Cell_ID', 'spot_id'])

    return img, mask, zyx_spots_coords, df_spots_coords

class _SpotMaxData(_Data):
    def __init__(
            self, images_path, intensity_image_path, spots_image_path, 
            acdc_df_path, segm_path, basename
        ):
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path,
            basename
        )
        self.spots_image_path = spots_image_path
    
    def spots_image_data(self):
        return load.load_image_file(self.spots_image_path)
        
class MitoDataSnapshot(_SpotMaxData):
    def __init__(self):
        images_path = os.path.join(
            data_path, 'test_multi_pos_analyse_single_pos', 'Position_15', 
            'Images'
        )
        intensity_image_path = os.path.join(
            images_path, 'ASY15-1_15nM-15_s15_phase_contr.tif'
        )
        spots_image_path = os.path.join(
            images_path, 'ASY15-1_15nM-15_s15_mNeon.tif'
        )
        acdc_df_path = os.path.join(
            images_path, 'ASY15-1_15nM-15_s15_acdc_output.csv'
        )
        segm_path = os.path.join(
            images_path, 'ASY15-1_15nM-15_s15_segm.npz'
        )
        basename = 'ASY15-1_15nM-15_s15_'
        super().__init__(
            images_path, intensity_image_path, spots_image_path, 
            acdc_df_path, segm_path, basename
        )
    