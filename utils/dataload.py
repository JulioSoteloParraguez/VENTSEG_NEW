# -*- coding: utf-8 -*-
#
# Copyright 2026 Julio Sotelo, Departamento de Informática, Universidad Técnica Federico Santa María
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
