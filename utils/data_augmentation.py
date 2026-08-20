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
Test-Time Preprocessing and Augmentation Routines for VENTSEG.
"""
import albumentations


def common_test_augmentation(img_size):
    """
    Standard test-time transformation pipeline for model inference.
    Pads to minimum square dimensions, center-crops, and resizes to target network input resolution (e.g., 224x224).

    :param img_size: (int) Target square image dimension (e.g. 224)
    :return: List of Albumentations transformation objects
    """
    return [
        albumentations.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True),
        albumentations.CenterCrop(height=img_size, width=img_size, always_apply=True),
        albumentations.Resize(height=img_size, width=img_size, always_apply=True)
    ]
