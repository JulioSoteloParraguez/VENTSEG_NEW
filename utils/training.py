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
Training and Inference Utility Functions for VENTSEG Cardiac Segmentation.
"""
import math
import numpy as np
import torch


def convert_multiclass_mask(mask):
    """
    Transform multiclass probability mask [batch, num_classes, h, w] to class label index [batch, h, w].

    :param mask: Multi-class tensor or array [B, C, H, W]
    :return: 1D class label map [B, H, W]
    """
    if isinstance(mask, torch.Tensor):
        return mask.max(1)[1]
    elif isinstance(mask, np.ndarray):
        return np.argmax(mask, axis=1)
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")


def reshape_masks(ndarray, to_shape):
    """
    Reshape a center-cropped (or center-padded) mask array back to original spatial dimensions.

    :param ndarray: (np.ndarray) 2D mask array to reshape
    :param to_shape: (tuple) Target output shape (h_out, w_out)
    :return: (np.ndarray) Reshaped array matching target dimensions
    """
    h_in, w_in = ndarray.shape
    h_out, w_out = to_shape

    # Vertical adjustment (height)
    if h_in > h_out:
        h_offset = math.ceil((h_in - h_out) / 2)
        ndarray = ndarray[h_offset:h_offset + h_out, :]
    elif h_in < h_out:
        h_offset = math.ceil((h_out - h_in) / 2)
        pad = np.zeros([h_out, w_in], dtype=ndarray.dtype)
        pad[h_offset:h_offset + h_in, :] = ndarray
        ndarray = pad

    # Horizontal adjustment (width)
    if w_in > w_out:
        w_offset = math.ceil((w_in - w_out) / 2)
        ndarray = ndarray[:, w_offset:w_offset + w_out]
    elif w_in < w_out:
        w_offset = math.ceil((w_out - w_in) / 2)
        pad = np.zeros([ndarray.shape[0], w_out], dtype=ndarray.dtype)
        pad[:, w_offset:w_offset + w_in] = ndarray
        ndarray = pad

    return ndarray
