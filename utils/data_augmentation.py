# -*- coding: utf-8 -*-
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
