# -*- coding: utf-8 -*-
"""
Data Loading and Image Normalization Routines for VENTSEG.
"""
import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None


def apply_normalization(image, normalization_type):
    """
    Apply intensity normalization to an input image.

    :param image: (np.ndarray) Input image array
    :param normalization_type: (str) Normalization strategy: 'none', 'reescale', or 'standardize'
    :return: (np.ndarray) Normalized image array
    """
    if normalization_type == "none" or normalization_type is None:
        return image.astype(np.float32)

    elif normalization_type == "reescale":
        image_min = float(image.min())
        image_max = float(image.max())
        if (image_max - image_min) > 1e-7:
            return ((image - image_min) / (image_max - image_min)).astype(np.float32)
        else:
            return np.zeros_like(image, dtype=np.float32)

    elif normalization_type == "standardize":
        image_mean = float(image.mean())
        image_std = float(image.std())
        if image_std > 1e-7:
            return ((image - image_mean) / image_std).astype(np.float32)
        else:
            return np.zeros_like(image, dtype=np.float32)

    else:
        raise ValueError(f"Unknown normalization policy: {normalization_type}")


def load_nii(path):
    """
    Load a NIfTI medical image file.

    :param path: (str) Path to .nii or .nii.gz file
    :return: (np.ndarray) Image array as float32
    """
    if nib is None:
        raise ImportError("The 'nibabel' library is required to load NIfTI images.")
    nii_img = nib.load(path)
    return nii_img.get_fdata().astype(np.float32)
