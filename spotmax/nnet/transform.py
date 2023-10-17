from matplotlib.pyplot import sca
import skimage.filters as filters
import numpy as np
from skimage.transform import rescale
from sklearn.preprocessing import MinMaxScaler
import cv2
import numba


@numba.jit(nopython=True)
def crop_single(img: np.ndarray, final_size: tuple) -> np.ndarray:
    """Function to crop an image to a final size.

    Args:
        img (np.ndarray): Image to crop
        final_size (tuple): Final size of the image

    Returns:
        np:ndarray: Cropped image
    """

    x_size = (img.shape[0] - final_size[0]) // 2
    y_size = (img.shape[1] - final_size[1]) // 2
    if img.shape[0] % 2 == 1:
        return img[y_size : -y_size - 1, x_size : -x_size - 1]
    else:
        return img[y_size:-y_size, x_size:-x_size]


@numba.jit(nopython=True)
def _get_xy_pads(final_size: tuple, img: np.ndarray) -> tuple:
    """Function to get x and y pads.

    Args:
        final_size (tuple): Final size of the image
        img (np.ndarray): Image to crop

    Returns:
        tuple: x and y pads, and constant values
    """
    x_pad_size = (final_size[0] - img.shape[0]) // 2
    y_pad_size = (final_size[1] - img.shape[1]) // 2
    constant_values = np.amin(img)
    if x_pad_size * 2 + img.shape[0] != final_size[0]:
        x_pad = (x_pad_size + 1, x_pad_size)
        y_pad = (y_pad_size + 1, y_pad_size)
    else:
        x_pad = (x_pad_size, x_pad_size)
        y_pad = (y_pad_size, y_pad_size)
    return x_pad, y_pad, constant_values


def pad_single(img: np.ndarray, final_size: tuple, mode: str = "constant") -> np.ndarray:
    """Function to pad an image to a final size.

    Args:
        img (np.ndarray): Image to pad
        final_size (tuple): Final size of the image
        mode (str, optional): Mode to use when padding. Defaults to "constant".

    Returns:
        np.ndarray: _description_
    """
    x_pad, y_pad, constant_values = _get_xy_pads(final_size=final_size, img=img)
    return np.pad(img, (y_pad, x_pad), mode, constant_values=constant_values)


def _convert_to_numpy(images: np.ndarray) -> np.ndarray:
    """Convert images to numpy array.

    Args:
        images (np.ndarray): Images to convert

    Returns:
        np.ndarray: Converted images
    """
    if not type(images) == np.ndarray:
        images = np.array(images)
    return images


def _pad(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to pad an array of images.

    Args:
        images (np.ndarray): Images to pad

    Returns:
        np.ndarray: Padded images
    """
    final_size = kwargs.get("final_size", (512, 512))
    mode = kwargs.get("mode", "constant")
    pad_images = [pad_single(img, final_size, mode) for img in images]
    return np.asarray(pad_images)


def _crop(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to crop an array of images.

    Args:
        images (np.ndarray): Images to crop

    Returns:
        np.ndarray: Cropped images
    """
    final_size = kwargs.get("final_size", (512, 512))
    return np.asarray([crop_single(img, final_size) for img in images])


def _pad_or_crop(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to pad or crop an array of images.

    Args:
        images (np.ndarray): Images to pad or crop

    Returns:
        np.ndarray: Padded or cropped images
    """
    final_size = kwargs.get("final_size", (512, 512))
    if images.shape == final_size:
        return images
    if images.shape[-1] > final_size[-1]:
        images = _crop(images, **kwargs)
    else:
        images = _pad(images, **kwargs)
    return images


def _rescale(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to rescale an array of images.

    Args:
        images (np.ndarray): Images to rescale

    Returns:
        np.ndarray: Rescaled images
    """
    scale = kwargs.get("scale", 1.0)
    order = kwargs.get("order", 1)
    anti_aliasing = kwargs.get("anti_aliasing", False)
    final_size = kwargs.get("final_size", None)

    # Rescale in 2D
    scaled_images = [rescale(img, scale, anti_aliasing=anti_aliasing, preserve_range=True, order=order) for img in images]
    # Rescale in 3D
    #scaled_images = rescale(images, scale, anti_aliasing=anti_aliasing, preserve_range=True, order=order)
    scaled_images = np.asarray(scaled_images)
    if final_size is not None:
        return _pad_or_crop(scaled_images, **kwargs)
    return scaled_images


def _normalize(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to normalize an array of images.

    Args:
        images (np.ndarray): Images to normalize

    Returns:
        np.ndarray: Normalized images
    """
    percentile = kwargs.get("percentile", 100)
    initial_shape = images.shape
    scaler = MinMaxScaler(feature_range=(-1, 1))
    images[images > np.percentile(images, percentile)] = np.percentile(images, percentile)
    images = scaler.fit_transform(images.reshape(-1, 1))
    images = images.reshape(initial_shape)
    return images


def _opening(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to applying opening filter to an array of images.

    Args:
        images (np.ndarray): Images where apply opening filter

    Returns:
        np.ndarray: Images after applying opening filter
    """
    kernel = kwargs.get("kernel", (2, 2))
    return np.asarray([cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) for img in images])


def _gaussian_filter(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to applying gaussian filter to an array of images.

    Args:
        images (np.ndarray): Images where apply gaussian filter

    Returns:
        np.ndarray: Images after applying gaussian filter
    """
    sigma = kwargs.get("sigma", 0.90)
    return np.asarray([filters.gaussian(img, sigma=sigma, preserve_range=True) for img in images])


def _to_bool(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to convert an array of images to boolean.

    Args:
        images (np.ndarray): Images to convert

    Returns:
        np.ndarray: Converted images
    """
    return images.astype(np.bool8)


def _to_integer(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to convert an array of images to integer.

    Args:
        images (np.ndarray): Images to convert

    Returns:
        np.ndarray: Converted images
    """
    return images.astype(int)


def _to_uint8(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to convert an array of images to uint8.

    Args:
        images (np.ndarray): Images to convert

    Returns:
        np.ndarray: Converted images
    """
    return images.astype(np.uint8)

def _to_uint16(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to convert an array of images to uint16.

    Args:
        images (np.ndarray): Images to convert

    Returns:
        np.ndarray: Converted images
    """
    return images.astype(np.uint16)


def _to_int64(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to convert an array of images to uint64.

    Args:
        images (np.ndarray): Images to convert

    Returns:
        np.ndarray: Converted images
    """
    return images.astype(np.int64)


def _to_float_32(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to convert an array of images to float.

    Args:
        images (np.ndarray): Images to convert

    Returns:
        np.ndarray: Converted images
    """
    return images.astype(np.float32)


@numba.jit(nopython=True)
def _remove_padding(image):
    """Function to remove padding from an image.

    Args:
        image (np.ndarray): Image to remove padding

    Returns:
        np.ndarray: Image without padding
    """
    for i in range(image.shape[0]):
        if np.all(image[0, :] == image[0, 0]):
            image = image[1:, :]
        else:
            break

    for i in range(image.shape[1]):
        if np.all(image[:, 0] == image[0, 0]):
            image = image[:, 1:]
        else:
            break

    for i in range(image.shape[0]):
        if np.all(image[-1, :] == image[-1, 0]):
            image = image[:-1, :]
        else:
            break

    for i in range(image.shape[1]):
        if np.all(image[:, -1] == image[0, -1]):
            image = image[:, :-1]
        else:
            break

    return np.asarray(image[1:-1, 1:-1])

def _remove_padding_wrapper(images, **kwargs):
    """Function to remove padding from an array of images.

    Args:
        images (np.ndarray): Images to remove padding

    Returns:
        np.ndarray: Images without padding
    """
    return np.asarray([_remove_padding(image) for image in images])

def _continuos_to_discrete(images: np.ndarray, **kwargs) -> np.ndarray:
    """Function to convert an array of images to discrete.

    Args:
        images (np.ndarray): Images to convert

    Returns:
        np.ndarray: Converted images
    """
    return ((images - images.min()) * (1/(images.max() - images.min()) * 255)).astype('uint8')


class ImageTransformer(object):
    """Class to transform images."""

    def __init__(self, logs=False) -> None:
        self.logs = logs

        self.pipeline = [
            _normalize,
        ]

    def __str__(self) -> str:
        print("The preprocess pipeline is: ", self.pipeline)

    def set_pipeline(self, pipeline: list):
        """Set the preprocessing pipeline.

        Args:
            pipeline (list): List of functions to apply to images
        """
        self.pipeline = pipeline

    def transform(self, images, **kwargs) -> np.ndarray:
        """Function to preprocess an array of images.

        Args:
            images (list|np.ndarray): Images to preprocess

        Returns:
            (np.ndarray): Preprocessed images
        """

        images = _convert_to_numpy(images)
        if len(images.shape) == 2:
            images = images.reshape((1, images.shape[0], images.shape[1]))

        if self.logs:
            print("The images dtype is: ", images.dtype)
            print("Before the preprocess step the maximum is {:4.3f}".format(np.amax(images)))
            print("Before the preprocess step the minimum is {:4.3f}".format(np.amin(images)))

        for func in self.pipeline:
            if self.logs:
                print("The preprocess step is: ", func.__name__)
            images = func(images, **kwargs)

        if self.logs:
            print("After the preprocess step the maximum is {:4.3f}".format(np.amax(images)))
            print("After the preprocess step the minimum is {:4.3f}".format(np.amin(images)))

        return images