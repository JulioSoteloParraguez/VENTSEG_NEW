# -*- coding: utf-8 -*-
"""
VENTSEG Deep Learning Models Package.
Provides model architectures and selectors for cardiac image segmentation.
"""
from .resnet import resnet_model_selector, ResUnetv4


def model_selector(model_name, num_classes=4, in_channels=1):
    """
    Select and instantiate neural network architecture for cardiac segmentation.

    :param model_name: Name of the model architecture (e.g., 'resnet34_unet_scratch')
    :param num_classes: Number of segmentation classes (default: 4)
    :param in_channels: Number of input channels (default: 1)
    :return: Instantiated PyTorch model
    """
    classification = False
    if "classification" in model_name:
        classification = True

    if "resnet34" in model_name or "resnet18" in model_name:
        return resnet_model_selector(model_name, num_classes, classification, in_channels)

    raise ValueError(f"Unknown or unsupported model architecture: {model_name}")
