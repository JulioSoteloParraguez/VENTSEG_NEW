#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VENTSEG - 4D Medical Image Viewer & Cardiac Volumetric Explorer
================================================================================
Interactive workstation and analysis suite for 4D cardiac medical imaging:
  - Dimension 0: Rows (Height / Y)
  - Dimension 1: Columns (Width / X)
  - Dimension 2: Slices (Z-Axis / Long/Short-Axis Stack)
  - Dimension 3: Cardiac Phases (Time / Cardiac Cycle / T)

Supported Formats:
  - MATLAB .mat (image_SA.mat, segmentation_all_phases.mat, segmentation_ED.mat,
                 segmentation_ES.mat, original.mat, resultado.mat)
  - NIfTI .nii / .nii.gz
  - NumPy .npy / .npz

Key Capabilities:
  - Integrated deep learning inference engine (ResNet34-UNet) for direct automatic segmentation.
  - Real-time cardiac cine player with configurable frame rate and transport controls.
  - Synchronized 3D volume and 2D slice area curves for Left Ventricle (LV), Right Ventricle (RV),
    and Myocardium (MYO) in milliliters (mL / cm³) and cm².
  - Automatic detection and visualization of End-Diastole (ED) and End-Systole (ES).
  - Quantitative clinical ventricular function parameters: Ejection Fraction (EF),
    Stroke Volume (SV), End-Diastolic Volume (EDV), End-Systolic Volume (ESV), and Myocardial Mass.
  - High-occupancy multi-view cardiac mosaic (Cardiac Cycle Phase Grid, Z-Stack, and ED vs ES Comparator).
  - Visual ED/ES key-phase markers and synchronized interactive navigation.
  - Multi-class segmentation overlay (LV: Red, MYO: Green, RV: Blue) with opacity control.
  - Window/Level (W/L) brightness & contrast presets, medical colormaps, and high-resolution export (PNG, GIF, CSV, TXT, MAT).
================================================================================
"""

import sys
import os
import time
import re
import numpy as np
import scipy.io

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    import imageio.v2 as imageio
except ImportError:
    try:
        import imageio
    except ImportError:
        imageio = None

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QFileDialog, QMessageBox, QGroupBox, QSplitter,
    QTabWidget, QToolBar, QStatusBar, QFrame, QCheckBox, QSizePolicy,
    QProgressBar, QScrollArea, QRadioButton, QButtonGroup, QDialog,
    QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QThread
from PyQt6.QtGui import QFont, QKeySequence, QColor, QPalette, QAction, QShortcut, QIcon, QPixmap


def get_app_icon():
    """Retrieve the application icon from images/Logo.ico or images/Logo.png."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for icon_name in ("Logo.ico", "Logo.png"):
        path = os.path.join(base_dir, "images", icon_name)
        if os.path.exists(path):
            icon = QIcon(path)
            if not icon.isNull():
                return icon
    return QIcon()



# ==============================================================================
# STYLES AND THEMES (Dark Medical UI)
# ==============================================================================
DARK_STYLE_SHEET = """
QMainWindow {
    background-color: #1a1b22;
    color: #e0e6ed;
}

QWidget {
    background-color: #1a1b22;
    color: #e0e6ed;
    font-family: 'Segoe UI', 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #2d313e;
    border-radius: 8px;
    margin-top: 1.2em;
    padding-top: 12px;
    padding-left: 8px;
    padding-right: 8px;
    padding-bottom: 8px;
    font-weight: bold;
    color: #4da8da;
    background-color: #21242d;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: #21242d;
}

QPushButton {
    background-color: #2e3440;
    color: #eceff4;
    border: 1px solid #3b4252;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #3b4252;
    border-color: #4da8da;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #007acc;
    border-color: #005999;
}

QPushButton:disabled {
    background-color: #1e2129;
    color: #5a6273;
    border-color: #282c37;
}

QPushButton#accentButton {
    background-color: #007acc;
    color: #ffffff;
    border: 1px solid #0098ff;
    font-weight: bold;
}

QPushButton#accentButton:hover {
    background-color: #0098ff;
    border-color: #33aaff;
}

QPushButton#edButton {
    background-color: #1e4620;
    color: #74c0fc;
    border: 1px solid #2b8a3e;
    font-weight: bold;
}

QPushButton#edButton:hover {
    background-color: #2b8a3e;
    color: #ffffff;
}

QPushButton#esButton {
    background-color: #491217;
    color: #ffa8a8;
    border: 1px solid #c92a2a;
    font-weight: bold;
}

QPushButton#esButton:hover {
    background-color: #c92a2a;
    color: #ffffff;
}

QPushButton#playButton {
    background-color: #2e7d32;
    color: #ffffff;
    border: 1px solid #4caf50;
    font-weight: bold;
}

QPushButton#playButton:hover {
    background-color: #388e3c;
}

QPushButton#pauseButton {
    background-color: #c62828;
    color: #ffffff;
    border: 1px solid #ef5350;
    font-weight: bold;
}

QPushButton#pauseButton:hover {
    background-color: #d32f2f;
}

QPushButton#aiButton {
    background-color: #4c1d95;
    color: #f3e8ff;
    border: 1px solid #7c3aed;
    font-weight: bold;
}

QPushButton#aiButton:hover {
    background-color: #6d28d9;
    border-color: #a78bfa;
    color: #ffffff;
}

QPushButton#aiButton:pressed {
    background-color: #7c3aed;
}

QDialog {
    background-color: #1a1b22;
    color: #e0e6ed;
}

QProgressBar {
    border: 1px solid #3b4252;
    border-radius: 6px;
    text-align: center;
    background-color: #21242d;
    color: #ffffff;
    font-weight: bold;
    height: 22px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #00adb5);
    border-radius: 5px;
}

QSlider::groove:horizontal {
    border: 1px solid #3b4252;
    height: 8px;
    background: #282c37;
    border-radius: 4px;
}

QSlider::sub-page:horizontal {
    background: #007acc;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #007acc;
    width: 18px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #e0f2fe;
    border-color: #0098ff;
}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #282c37;
    border: 1px solid #3b4252;
    border-radius: 6px;
    padding: 4px 8px;
    color: #eceff4;
    min-height: 22px;
}

QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #4da8da;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 0px;
}

QComboBox QAbstractItemView {
    background-color: #21242d;
    color: #eceff4;
    selection-background-color: #007acc;
    selection-color: #ffffff;
    border: 1px solid #3b4252;
}

QTabWidget::pane {
    border: 1px solid #2d313e;
    border-radius: 6px;
    background-color: #1a1b22;
}

QTabBar::tab {
    background: #21242d;
    color: #a0aec0;
    border: 1px solid #2d313e;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    margin-right: 2px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background: #1a1b22;
    color: #4da8da;
    border-bottom: 2px solid #007acc;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background: #282c37;
    color: #e0e6ed;
}

QStatusBar {
    background-color: #13141a;
    color: #8f9ba8;
    border-top: 1px solid #21242d;
}

QToolBar {
    background-color: #21242d;
    border-bottom: 1px solid #2d313e;
    spacing: 6px;
    padding: 4px;
}

QLabel#badgeLabel {
    background-color: #007acc;
    color: #ffffff;
    font-weight: bold;
    border-radius: 4px;
    padding: 2px 8px;
}

QLabel#edBadge {
    background-color: #2b8a3e;
    color: #ffffff;
    font-weight: bold;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 11px;
}

QLabel#esBadge {
    background-color: #c92a2a;
    color: #ffffff;
    font-weight: bold;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 11px;
}

QLabel#metricCard {
    background-color: #21242d;
    border: 1px solid #2d313e;
    border-radius: 6px;
    padding: 6px 8px;
}

QCheckBox, QRadioButton {
    spacing: 8px;
    color: #e0e6ed;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #3b4252;
    background-color: #282c37;
}

QCheckBox::indicator:checked {
    background-color: #007acc;
    border-color: #0098ff;
}

QScrollArea {
    border: none;
    background-color: transparent;
}
"""


# ==============================================================================
# 4D MEDICAL IMAGE DATA MODEL WITH CARDIAC QUANTIFICATION
# ==============================================================================
class Medical4DImage:
    """Represents a 4D medical image (Rows, Cols, Slices, Phases), its spatial calibration (voxel_size), and segmentation masks."""

    def __init__(self, data=None, filename="", var_name="", voxel_size=None):
        self.filename = filename
        self.var_name = var_name
        self.raw_data = None
        self.data_4d = None  # Always normalized to shape (Rows, Cols, Slices, Phases)
        self.shape_original = None
        self.min_val = 0.0
        self.max_val = 1.0
        self.mean_val = 0.0
        self.std_val = 0.0
        self.dtype_str = "float32"

        # 4D Segmentation Mask: (Rows, Cols, Slices, Phases)
        self.mask_4d = None
        self.mask_filename = ""

        # Physical Voxel Size: [dx, dy, dz] in mm (e.g. [0.5, 0.5, 8.0])
        # dx: horizontal column spacing (X), dy: vertical row spacing (Y), dz: slice thickness (Z)
        self.voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.has_voxel_size_metadata = False
        if voxel_size is not None:
            self.set_voxel_size(voxel_size, mark_as_metadata=True)

        # Computed cardiac volume curves and clinical metrics
        self.has_mask = False
        self.cardiac_metrics = {}
        self.volume_curves_ml = {}          # 'lv', 'myo', 'rv' -> arrays in mL (cm³)
        self.volume_curves_voxels = {}      # 'lv', 'myo', 'rv' -> arrays in voxels
        self.volume_curves = {}             # Points to volume_curves_ml by default
        self.slice_area_curves_cm2 = {}     # slice_idx -> {'lv', 'myo', 'rv'} in cm²
        self.slice_area_curves_pixels = {}  # slice_idx -> {'lv', 'myo', 'rv'} in pixels
        self.slice_area_curves = {}         # Points to slice_area_curves_cm2 by default
        self.ed_phase = 0                   # End-Diastole phase index (0-indexed)
        self.es_phase = 0                   # End-Systole phase index (0-indexed)

        if data is not None:
            self.set_data(data, filename, var_name)

    def set_voxel_size(self, voxel_size, mark_as_metadata=True):
        """Set physical voxel size [dx, dy, dz] in mm and recompute quantitative volumetric curves."""
        v = np.squeeze(np.asarray(voxel_size, dtype=np.float64))
        if v.size >= 3:
            self.voxel_size = np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
        elif v.size == 2:
            self.voxel_size = np.array([float(v[0]), float(v[1]), 1.0], dtype=np.float64)
        elif v.size == 1:
            val = float(v)
            self.voxel_size = np.array([val, val, val], dtype=np.float64)
        else:
            self.voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        if mark_as_metadata:
            self.has_voxel_size_metadata = True

        if self.mask_4d is not None:
            self.recompute_metrics()

    @property
    def voxel_volume_mm3(self):
        """Volume of a single voxel in mm³."""
        return float(np.prod(self.voxel_size))

    @property
    def voxel_volume_ml(self):
        """Volume of a single voxel in mL (1 mL = 1 cm³ = 1000 mm³)."""
        return self.voxel_volume_mm3 / 1000.0

    @property
    def pixel_area_mm2(self):
        """2D in-plane pixel area in mm²."""
        return float(self.voxel_size[0] * self.voxel_size[1])

    @property
    def pixel_area_cm2(self):
        """2D in-plane pixel area in cm² (1 cm² = 100 mm²)."""
        return self.pixel_area_mm2 / 100.0

    def set_data(self, data, filename="", var_name=""):
        self.raw_data = np.asarray(data)
        self.filename = filename
        self.var_name = var_name
        self.shape_original = self.raw_data.shape

        # Normalize dimensions to 4D: (Rows, Cols, Slices, Phases)
        arr = self.raw_data
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis, np.newaxis]
        elif arr.ndim == 3:
            arr = arr[:, :, :, np.newaxis]
        elif arr.ndim == 4:
            arr = arr
        else:
            raise ValueError(f"Unsupported image dimension: {arr.ndim}D. Expected 2D, 3D, or 4D image.")

        self.data_4d = arr
        self.dtype_str = str(self.data_4d.dtype)
        self.min_val = float(np.min(self.data_4d))
        self.max_val = float(np.max(self.data_4d))
        self.mean_val = float(np.mean(self.data_4d))
        self.std_val = float(np.std(self.data_4d))

        if self.mask_4d is not None:
            self.recompute_metrics()

    @property
    def num_rows(self):
        return self.data_4d.shape[0] if self.data_4d is not None else 0

    @property
    def num_cols(self):
        return self.data_4d.shape[1] if self.data_4d is not None else 0

    @property
    def num_slices(self):
        return self.data_4d.shape[2] if self.data_4d is not None else 0

    @property
    def num_phases(self):
        return self.data_4d.shape[3] if self.data_4d is not None else 0

    def get_slice_2d(self, slice_idx, phase_idx):
        """Retrieve the 2D cross-sectional slice for the specified slice index and cardiac phase."""
        if self.data_4d is None:
            return None
        z = max(0, min(slice_idx, self.num_slices - 1))
        t = max(0, min(phase_idx, self.num_phases - 1))
        return self.data_4d[:, :, z, t]

    def get_mask_slice_2d(self, slice_idx, phase_idx):
        """Retrieve the corresponding 2D segmentation mask slice if available."""
        if self.mask_4d is None:
            return None
        z = max(0, min(slice_idx, self.mask_4d.shape[2] - 1))
        t = max(0, min(phase_idx, self.mask_4d.shape[3] - 1))
        return self.mask_4d[:, :, z, t]

    def set_mask(self, mask_data, mask_filename=""):
        """Assign a multi-class segmentation mask and calculate volumetric curves and clinical parameters."""
        m = np.asarray(mask_data)
        self.mask_filename = mask_filename

        # Adjust dimensions to match the medical volume
        if m.ndim == 2:
            m = np.repeat(np.repeat(m[:, :, np.newaxis, np.newaxis], self.num_slices, axis=2), self.num_phases, axis=3)
        elif m.ndim == 3:
            if m.shape == (self.num_rows, self.num_cols, self.num_slices):
                m = np.repeat(m[:, :, :, np.newaxis], self.num_phases, axis=3)
            elif m.shape == (self.num_rows, self.num_cols, self.num_phases):
                m_full = np.zeros((self.num_rows, self.num_cols, self.num_slices, self.num_phases), dtype=m.dtype)
                mid_slice = self.num_slices // 2
                m_full[:, :, mid_slice, :] = m
                m = m_full
            else:
                m = m[:, :, :, np.newaxis]
                if m.shape[2] < self.num_slices:
                    m_full = np.zeros((self.num_rows, self.num_cols, self.num_slices, self.num_phases), dtype=m.dtype)
                    m_full[:, :, :m.shape[2], :m.shape[3]] = m
                    m = m_full
        elif m.ndim == 4:
            if m.shape != (self.num_rows, self.num_cols, self.num_slices, self.num_phases):
                m_adj = np.zeros((self.num_rows, self.num_cols, self.num_slices, self.num_phases), dtype=m.dtype)
                min_r = min(self.num_rows, m.shape[0])
                min_c = min(self.num_cols, m.shape[1])
                min_z = min(self.num_slices, m.shape[2])
                min_t = min(self.num_phases, m.shape[3])
                m_adj[:min_r, :min_c, :min_z, :min_t] = m[:min_r, :min_c, :min_z, :min_t]
                m = m_adj

        self.mask_4d = m
        self.has_mask = (self.mask_4d is not None and np.max(self.mask_4d) > 0)
        self.recompute_metrics()

    def recompute_metrics(self):
        """Compute 3D volume curves (in mL and voxels) and 2D area curves (in cm² and pixels) for LV (1), MYO (2), RV (3), and detect ED/ES."""
        if not self.has_mask or self.mask_4d is None or self.num_phases == 0:
            self.volume_curves_ml = {'lv': np.zeros(self.num_phases), 'myo': np.zeros(self.num_phases), 'rv': np.zeros(self.num_phases)}
            self.volume_curves_voxels = {'lv': np.zeros(self.num_phases), 'myo': np.zeros(self.num_phases), 'rv': np.zeros(self.num_phases)}
            self.volume_curves = self.volume_curves_ml
            self.slice_area_curves_cm2 = {}
            self.slice_area_curves_pixels = {}
            self.slice_area_curves = {}
            self.ed_phase = 0
            self.es_phase = 0
            self.cardiac_metrics = {}
            return

        n_phases = self.num_phases
        n_slices = self.num_slices

        lv_vol_vx = np.zeros(n_phases, dtype=np.float64)
        myo_vol_vx = np.zeros(n_phases, dtype=np.float64)
        rv_vol_vx = np.zeros(n_phases, dtype=np.float64)

        self.slice_area_curves_pixels = {}
        self.slice_area_curves_cm2 = {}
        for s in range(n_slices):
            self.slice_area_curves_pixels[s] = {
                'lv': np.zeros(n_phases, dtype=np.float64),
                'myo': np.zeros(n_phases, dtype=np.float64),
                'rv': np.zeros(n_phases, dtype=np.float64)
            }
            self.slice_area_curves_cm2[s] = {
                'lv': np.zeros(n_phases, dtype=np.float64),
                'myo': np.zeros(n_phases, dtype=np.float64),
                'rv': np.zeros(n_phases, dtype=np.float64)
            }

        for t in range(n_phases):
            mask_t = self.mask_4d[:, :, :, t]
            lv_vol_vx[t] = np.sum(mask_t == 1)
            myo_vol_vx[t] = np.sum(mask_t == 2)
            rv_vol_vx[t] = np.sum(mask_t == 3)

            for s in range(n_slices):
                slice_mask = mask_t[:, :, s]
                lv_p = np.sum(slice_mask == 1)
                myo_p = np.sum(slice_mask == 2)
                rv_p = np.sum(slice_mask == 3)

                self.slice_area_curves_pixels[s]['lv'][t] = lv_p
                self.slice_area_curves_pixels[s]['myo'][t] = myo_p
                self.slice_area_curves_pixels[s]['rv'][t] = rv_p

                self.slice_area_curves_cm2[s]['lv'][t] = lv_p * self.pixel_area_cm2
                self.slice_area_curves_cm2[s]['myo'][t] = myo_p * self.pixel_area_cm2
                self.slice_area_curves_cm2[s]['rv'][t] = rv_p * self.pixel_area_cm2

        # Convert to physiological volumes in mL (1 mL = 1 cm³ = 1000 mm³)
        lv_vol_ml = lv_vol_vx * self.voxel_volume_ml
        myo_vol_ml = myo_vol_vx * self.voxel_volume_ml
        rv_vol_ml = rv_vol_vx * self.voxel_volume_ml

        self.volume_curves_ml = {'lv': lv_vol_ml, 'myo': myo_vol_ml, 'rv': rv_vol_ml}
        self.volume_curves_voxels = {'lv': lv_vol_vx, 'myo': myo_vol_vx, 'rv': rv_vol_vx}
        self.volume_curves = self.volume_curves_ml
        self.slice_area_curves = self.slice_area_curves_cm2

        # Determine End-Diastole (ED) and End-Systole (ES)
        # ED = Maximum cavity volume of Left Ventricle (LV)
        # ES = Minimum cavity volume of Left Ventricle (LV)
        if np.max(lv_vol_vx) > 0:
            self.ed_phase = int(np.argmax(lv_vol_vx))
            self.es_phase = int(np.argmin(lv_vol_vx))
            if self.ed_phase == self.es_phase and n_phases > 1:
                self.es_phase = (self.ed_phase + n_phases // 2) % n_phases
        elif np.max(myo_vol_vx) > 0:
            self.ed_phase = int(np.argmax(myo_vol_vx))
            self.es_phase = int(np.argmin(myo_vol_vx))
        elif np.max(rv_vol_vx) > 0:
            self.ed_phase = int(np.argmax(rv_vol_vx))
            self.es_phase = int(np.argmin(rv_vol_vx))
        else:
            self.ed_phase = 0
            self.es_phase = min(1, n_phases - 1)

        # Quantitative parameters in mL and voxels
        edv_ml = lv_vol_ml[self.ed_phase]
        esv_ml = lv_vol_ml[self.es_phase]
        sv_ml = max(0.0, edv_ml - esv_ml)
        ef = (sv_ml / edv_ml * 100.0) if edv_ml > 0 else 0.0

        edv_vx = lv_vol_vx[self.ed_phase]
        esv_vx = lv_vol_vx[self.es_phase]
        sv_vx = max(0.0, edv_vx - esv_vx)

        myo_ed_ml = myo_vol_ml[self.ed_phase] if self.ed_phase < n_phases else 0.0
        myo_es_ml = myo_vol_ml[self.es_phase] if self.es_phase < n_phases else 0.0
        myo_ed_vx = myo_vol_vx[self.ed_phase] if self.ed_phase < n_phases else 0.0
        myo_es_vx = myo_vol_vx[self.es_phase] if self.es_phase < n_phases else 0.0
        myo_mass_g = myo_ed_ml * 1.05  # Standard clinical myocardial tissue density: 1.05 g/mL

        rv_edv_ml = rv_vol_ml[self.ed_phase] if self.ed_phase < n_phases else 0.0
        rv_esv_ml = rv_vol_ml[self.es_phase] if self.es_phase < n_phases else 0.0
        rv_sv_ml = max(0.0, rv_edv_ml - rv_esv_ml)
        rv_ef = (rv_sv_ml / rv_edv_ml * 100.0) if rv_edv_ml > 0 else 0.0

        rv_edv_vx = rv_vol_vx[self.ed_phase] if self.ed_phase < n_phases else 0.0
        rv_esv_vx = rv_vol_vx[self.es_phase] if self.es_phase < n_phases else 0.0
        rv_sv_vx = max(0.0, rv_edv_vx - rv_esv_vx)

        self.cardiac_metrics = {
            'ed_phase': self.ed_phase,
            'es_phase': self.es_phase,
            'lv_edv': edv_ml,
            'lv_esv': esv_ml,
            'lv_sv': sv_ml,
            'lv_ef': ef,
            'lv_edv_ml': edv_ml,
            'lv_esv_ml': esv_ml,
            'lv_sv_ml': sv_ml,
            'lv_edv_vx': edv_vx,
            'lv_esv_vx': esv_vx,
            'lv_sv_vx': sv_vx,
            'myo_ed': myo_ed_ml,
            'myo_es': myo_es_ml,
            'myo_ed_ml': myo_ed_ml,
            'myo_es_ml': myo_es_ml,
            'myo_ed_vx': myo_ed_vx,
            'myo_es_vx': myo_es_vx,
            'myo_mass_g': myo_mass_g,
            'rv_edv': rv_edv_ml,
            'rv_esv': rv_esv_ml,
            'rv_sv': rv_sv_ml,
            'rv_ef': rv_ef,
            'rv_edv_ml': rv_edv_ml,
            'rv_esv_ml': rv_esv_ml,
            'rv_sv_ml': rv_sv_ml,
            'rv_edv_vx': rv_edv_vx,
            'rv_esv_vx': rv_esv_vx,
            'rv_sv_vx': rv_sv_vx,
            'voxel_size': self.voxel_size.copy(),
            'voxel_volume_ml': self.voxel_volume_ml,
            'voxel_volume_mm3': self.voxel_volume_mm3,
            'pixel_area_cm2': self.pixel_area_cm2,
            'pixel_area_mm2': self.pixel_area_mm2
        }

    def get_time_series_at_pixel(self, row, col, slice_idx):
        """Extract intensity time-series across all cardiac phases at the specified voxel coordinate."""
        if self.data_4d is None:
            return np.array([])
        r = max(0, min(row, self.num_rows - 1))
        c = max(0, min(col, self.num_cols - 1))
        z = max(0, min(slice_idx, self.num_slices - 1))
        return self.data_4d[r, c, z, :]


def load_file_4d(filepath):
    """Load a 4D medical image (.mat, .nii, .nii.gz, .npy, .npz) and extract voxel dimensions."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    base = os.path.basename(filepath)
    base_dir = os.path.dirname(os.path.abspath(filepath))
    voxel_keys = ['voxel_size', 'voxelsize', 'voxel_spacing', 'spacing', 'pixel_spacing', 'pixdim', 'voxdim']

    if ext == '.mat':
        mat_contents = scipy.io.loadmat(filepath)
        candidate_keys = [
            'image_SA', 'image_sa', 'imagesa', 'imageSA',
            'segmentation_all_phases', 'segmentation_ED', 'segmentation_ES',
            'segmentation_ed', 'segmentation_es', 'segmentation_phase',
            'original', 'resultado', 'middle', 'phase', 'img', 'data', 'images', 'volume'
        ]
        selected_key = None
        for k in candidate_keys:
            if k in mat_contents and hasattr(mat_contents[k], 'shape'):
                selected_key = k
                break

        if selected_key is None:
            non_dunder = [k for k in mat_contents.keys() if not k.startswith('__') and hasattr(mat_contents[k], 'shape')]
            if non_dunder:
                selected_key = non_dunder[0]
            else:
                raise ValueError(f"No numeric arrays found in MATLAB file {base}.")

        arr = mat_contents[selected_key]

        # Extract voxel_size if present
        voxel_size = None
        for vk in voxel_keys:
            if vk in mat_contents:
                v = np.squeeze(np.asarray(mat_contents[vk], dtype=np.float64))
                if v.size >= 3:
                    voxel_size = np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
                    break
                elif v.size == 2:
                    voxel_size = np.array([float(v[0]), float(v[1]), 1.0], dtype=np.float64)
                    break
                elif v.size == 1:
                    val = float(v)
                    voxel_size = np.array([val, val, val], dtype=np.float64)
                    break

        # Fallback to companion files if missing
        if voxel_size is None and base_dir:
            for fallback_name in ['image_SA.mat', 'original.mat']:
                fallback_path = os.path.join(base_dir, fallback_name)
                if os.path.exists(fallback_path):
                    try:
                        fallback_mat = scipy.io.loadmat(fallback_path)
                        for vk in voxel_keys:
                            if vk in fallback_mat:
                                v = np.squeeze(np.asarray(fallback_mat[vk], dtype=np.float64))
                                if v.size >= 3:
                                    voxel_size = np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
                                    break
                                elif v.size == 2:
                                    voxel_size = np.array([float(v[0]), float(v[1]), 1.0], dtype=np.float64)
                                    break
                                elif v.size == 1:
                                    val = float(v)
                                    voxel_size = np.array([val, val, val], dtype=np.float64)
                                    break
                        if voxel_size is not None:
                            break
                    except Exception:
                        pass

        return Medical4DImage(arr, filename=filepath, var_name=selected_key, voxel_size=voxel_size)

    elif ext in ('.nii', '.gz') or filepath.endswith('.nii.gz'):
        if nib is None:
            raise ImportError("The 'nibabel' library is required to read NIfTI files.")
        nii_obj = nib.load(filepath)
        arr = nii_obj.get_fdata()
        voxel_size = None
        try:
            zooms = nii_obj.header.get_zooms()
            if zooms and len(zooms) >= 3:
                voxel_size = np.array([float(zooms[0]), float(zooms[1]), float(zooms[2])], dtype=np.float64)
        except Exception:
            pass
        return Medical4DImage(arr, filename=filepath, var_name="NIfTI", voxel_size=voxel_size)

    elif ext == '.npy':
        arr = np.load(filepath)
        voxel_size = None
        if base_dir:
            for fallback_name in ['image_SA.mat', 'original.mat']:
                fallback_path = os.path.join(base_dir, fallback_name)
                if os.path.exists(fallback_path):
                    try:
                        fallback_mat = scipy.io.loadmat(fallback_path)
                        for vk in voxel_keys:
                            if vk in fallback_mat:
                                v = np.squeeze(np.asarray(fallback_mat[vk], dtype=np.float64))
                                if v.size >= 3:
                                    voxel_size = np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
                                    break
                                elif v.size == 2:
                                    voxel_size = np.array([float(v[0]), float(v[1]), 1.0], dtype=np.float64)
                                    break
                                elif v.size == 1:
                                    val = float(v)
                                    voxel_size = np.array([val, val, val], dtype=np.float64)
                                    break
                        if voxel_size is not None:
                            break
                    except Exception:
                        pass
        return Medical4DImage(arr, filename=filepath, var_name="numpy_array", voxel_size=voxel_size)

    elif ext == '.npz':
        npz_obj = np.load(filepath)
        keys = list(npz_obj.keys())
        if not keys:
            raise ValueError(f"Empty NPZ archive: {base}")
        selected_key = 'image_SA' if 'image_SA' in keys else ('original' if 'original' in keys else keys[0])
        arr = npz_obj[selected_key]
        voxel_size = None
        if 'voxel_size' in keys:
            v = np.squeeze(np.asarray(npz_obj['voxel_size'], dtype=np.float64))
            if v.size >= 3:
                voxel_size = np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
        return Medical4DImage(arr, filename=filepath, var_name=selected_key, voxel_size=voxel_size)

    else:
        raise ValueError(f"Unsupported format: {ext}. Supported formats are .mat, .nii, .nii.gz, .npy, and .npz.")


# ==============================================================================
# INTERACTIVE MATPLOTLIB CANVAS (Single View & Volumetric Curves)
# ==============================================================================
class InteractiveViewerCanvas(FigureCanvasQTAgg):
    """Main viewer canvas displaying 2D anatomical slice and synchronized cardiac volume/area curves."""

    pixel_clicked = pyqtSignal(int, int)        # row, col
    phase_clicked = pyqtSignal(int)             # phase index (0-based)
    slice_scroll_requested = pyqtSignal(int)    # delta (+1 or -1)
    phase_scroll_requested = pyqtSignal(int)    # delta (+1 or -1)

    def __init__(self, parent=None):
        self.fig = Figure(facecolor='#1a1b22', tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

        gs = self.fig.add_gridspec(3, 1, height_ratios=[3.6, 0.15, 1.45])
        self.ax_img = self.fig.add_subplot(gs[0])
        self.ax_curve = self.fig.add_subplot(gs[2])

        self.img_artist = None
        self.mask_artist = None
        self.crosshair_v = None
        self.crosshair_h = None

        self.selected_row = None
        self.selected_col = None

        self.curve_mode = "volume_3d"  # "volume_3d" or "area_2d"
        self.unit_mode = "ml"          # "ml" or "voxels"

        self.show_empty_placeholder()

        self.mpl_connect('button_press_event', self.on_mouse_click)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('scroll_event', self.on_scroll)

        self.status_callback = None

    def show_empty_placeholder(self):
        self.ax_img.clear()
        self.ax_img.set_facecolor('#13141a')
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        for spine in self.ax_img.spines.values():
            spine.set_color('#2d313e')
        self.ax_img.text(
            0.5, 0.5,
            "VENTSEG 4D - Cardiac Image Viewer & Quantification Suite\n\n"
            "Open a 4D medical dataset (.mat, .nii, .npy)\n"
            "via 'Open Image' (Ctrl+O) to begin.\n\n"
            "Automatically quantify and segment using ResNet34-UNet\n"
            "and save all clinical outputs to any folder (Ctrl+Shift+S).",
            horizontalalignment='center', verticalalignment='center',
            transform=self.ax_img.transAxes, color='#8f9ba8', fontsize=10.5,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#21242d', edgecolor='#3b4252', alpha=0.9)
        )
        self.ax_img.set_title("2D Cardiac Slice (No Image Loaded)", color='#4da8da', fontsize=11, pad=6, fontweight='bold')

        self.ax_curve.clear()
        self.ax_curve.set_facecolor('#13141a')
        self.ax_curve.set_xticks([])
        self.ax_curve.set_yticks([])
        for spine in self.ax_curve.spines.values():
            spine.set_color('#2d313e')
        self.ax_curve.text(
            0.5, 0.5,
            "Waiting for image and segmentation to plot physiological volume curves in mL...",
            horizontalalignment='center', verticalalignment='center',
            transform=self.ax_curve.transAxes, color='#6b7280', fontsize=9
        )
        self.ax_curve.set_title("Cardiac Volume Curves", color='#6b7280', fontsize=9.5, pad=4)
        self.draw_idle()

    def setup_axes(self):
        self.ax_img.set_facecolor('#13141a')
        self.ax_img.tick_params(colors='#8f9ba8', labelsize=9)
        for spine in self.ax_img.spines.values():
            spine.set_color('#2d313e')
        self.ax_img.set_title("2D Cardiac Slice", color='#4da8da', fontsize=12, pad=6, fontweight='bold')

        self.ax_curve.set_facecolor('#13141a')
        self.ax_curve.tick_params(colors='#8f9ba8', labelsize=8)
        for spine in self.ax_curve.spines.values():
            spine.set_color('#2d313e')
        self.ax_curve.set_xlabel("Cardiac Phase (Time / Frame)", color='#8f9ba8', fontsize=9)
        self.ax_curve.set_ylabel("Volume (mL)", color='#8f9ba8', fontsize=9)
        self.ax_curve.grid(True, color='#2d313e', linestyle='--', alpha=0.6)
        self.ax_curve.set_title("Cardiac Volume Curves (LV, MYO, RV)", color='#00adb5', fontsize=10, pad=4)

    def update_display(self, slice_data, vmin, vmax, cmap_name, mask_slice=None, mask_alpha=0.45,
                       show_mask=False, current_slice=0, total_slices=1, current_phase=0, total_phases=1,
                       med_image=None):
        if slice_data is None:
            return

        # 1. Update 2D Anatomical Slice
        is_ed = (med_image is not None and med_image.has_mask and current_phase == med_image.ed_phase)
        is_es = (med_image is not None and med_image.has_mask and current_phase == med_image.es_phase)

        phase_tag = ""
        if is_ed:
            phase_tag = "  |  [END-DIASTOLE - ED]"
        elif is_es:
            phase_tag = "  |  [END-SYSTOLE - ES]"

        title_text = f"Slice: {current_slice + 1}/{total_slices}  |  Cardiac Phase: {current_phase + 1}/{total_phases}{phase_tag}"

        if self.img_artist is None:
            self.ax_img.clear()
            self.setup_axes()
            self.img_artist = self.ax_img.imshow(
                slice_data, cmap=cmap_name, vmin=vmin, vmax=vmax,
                interpolation='bilinear', origin='upper'
            )
            self.ax_img.set_title(title_text, color='#4da8da' if not (is_ed or is_es) else ('#51cf66' if is_ed else '#ff6b6b'),
                                  fontsize=11, fontweight='bold')
        else:
            self.img_artist.set_data(slice_data)
            self.img_artist.set_clim(vmin=vmin, vmax=vmax)
            self.img_artist.set_cmap(cmap_name)
            self.ax_img.set_title(title_text, color='#4da8da' if not (is_ed or is_es) else ('#51cf66' if is_ed else '#ff6b6b'),
                                  fontsize=11, fontweight='bold')

        # Segmentation Mask Overlay
        if show_mask and mask_slice is not None:
            # 0: Background, 1: LV Endo (Red), 2: Myocardium (Green), 3: RV (Blue)
            seg_colors = [
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 0.25, 0.25, mask_alpha),
                (0.2, 0.85, 0.35, mask_alpha),
                (0.2, 0.6, 1.0, mask_alpha)
            ]
            seg_cmap = ListedColormap(seg_colors)
            if self.mask_artist is None:
                self.mask_artist = self.ax_img.imshow(
                    mask_slice, cmap=seg_cmap, vmin=0, vmax=3,
                    interpolation='nearest', origin='upper'
                )
            else:
                self.mask_artist.set_data(mask_slice)
                self.mask_artist.set_cmap(seg_cmap)
                self.mask_artist.set_visible(True)
        else:
            if self.mask_artist is not None:
                self.mask_artist.set_visible(False)

        # Crosshair cursor
        if self.selected_row is not None and self.selected_col is not None:
            if self.crosshair_v is not None:
                self.crosshair_v.remove()
                self.crosshair_h.remove()
            self.crosshair_v = self.ax_img.axvline(x=self.selected_col, color='#00ffcc', linestyle=':', alpha=0.7, linewidth=1)
            self.crosshair_h = self.ax_img.axhline(y=self.selected_row, color='#00ffcc', linestyle=':', alpha=0.7, linewidth=1)

        # 2. Update Cardiac Volume / Area Curves
        self.ax_curve.clear()
        self.ax_curve.set_facecolor('#13141a')
        self.ax_curve.tick_params(colors='#8f9ba8', labelsize=8)
        for spine in self.ax_curve.spines.values():
            spine.set_color('#2d313e')
        self.ax_curve.grid(True, color='#2d313e', linestyle='--', alpha=0.6)
        self.ax_curve.set_xlabel("Cardiac Phase (Time / Frame)", color='#8f9ba8', fontsize=9)

        if med_image is not None and med_image.has_mask and total_phases > 0:
            phases_x = np.arange(1, total_phases + 1)

            if self.curve_mode == "volume_3d":
                if self.unit_mode == "ml":
                    lv_data = med_image.volume_curves_ml.get('lv', np.zeros(total_phases))
                    myo_data = med_image.volume_curves_ml.get('myo', np.zeros(total_phases))
                    rv_data = med_image.volume_curves_ml.get('rv', np.zeros(total_phases))
                    unit_str = "Total 3D Volume (mL)"
                    mode_title = "3D Cardiac Volume (mL)"
                    val_fmt = "{:.2f} mL"
                else:
                    lv_data = med_image.volume_curves_voxels.get('lv', np.zeros(total_phases))
                    myo_data = med_image.volume_curves_voxels.get('myo', np.zeros(total_phases))
                    rv_data = med_image.volume_curves_voxels.get('rv', np.zeros(total_phases))
                    unit_str = "Total 3D Volume (Voxels)"
                    mode_title = "3D Cardiac Volume (Voxels)"
                    val_fmt = "{:,.0f} vx"
            else:
                if self.unit_mode == "ml":
                    slice_curves = med_image.slice_area_curves_cm2.get(current_slice, {
                        'lv': np.zeros(total_phases),
                        'myo': np.zeros(total_phases),
                        'rv': np.zeros(total_phases)
                    })
                    unit_str = f"Slice {current_slice + 1} 2D Area (cm²)"
                    mode_title = f"Slice {current_slice + 1} Area (cm²)"
                    val_fmt = "{:.2f} cm²"
                else:
                    slice_curves = med_image.slice_area_curves_pixels.get(current_slice, {
                        'lv': np.zeros(total_phases),
                        'myo': np.zeros(total_phases),
                        'rv': np.zeros(total_phases)
                    })
                    unit_str = f"Slice {current_slice + 1} 2D Area (Pixels)"
                    mode_title = f"Slice {current_slice + 1} Area (Pixels)"
                    val_fmt = "{:,.0f} px"

                lv_data = slice_curves['lv']
                myo_data = slice_curves['myo']
                rv_data = slice_curves['rv']

            self.ax_curve.set_ylabel(unit_str, color='#8f9ba8', fontsize=8.5)

            # Plot curves: LV (Red), Myocardium (Green), RV (Blue)
            self.ax_curve.plot(phases_x, lv_data, marker='o', markersize=4.5, color='#ff5c5c',
                               linewidth=2.2, label='Left Ventricle (LV)', zorder=4)
            self.ax_curve.plot(phases_x, myo_data, marker='s', markersize=4.5, color='#51cf66',
                               linewidth=2.0, label='Myocardium (MYO)', zorder=3)
            self.ax_curve.plot(phases_x, rv_data, marker='^', markersize=4.5, color='#339af0',
                               linewidth=2.0, label='Right Ventricle (RV)', zorder=3)

            ed_idx = med_image.ed_phase
            es_idx = med_image.es_phase

            # Mark End-Diastole (ED) Phase
            if 0 <= ed_idx < total_phases:
                self.ax_curve.axvline(x=ed_idx + 1, color='#51cf66', linestyle='--', linewidth=1.8,
                                      alpha=0.85, label=f'[ED] Phase {ed_idx + 1}', zorder=2)
                self.ax_curve.plot(ed_idx + 1, lv_data[ed_idx], marker='o', markersize=7,
                                  color='#51cf66', markeredgecolor='#ffffff', markeredgewidth=1.2, zorder=6)

            # Mark End-Systole (ES) Phase
            if 0 <= es_idx < total_phases:
                self.ax_curve.axvline(x=es_idx + 1, color='#ff6b6b', linestyle='--', linewidth=1.8,
                                      alpha=0.85, label=f'[ES] Phase {es_idx + 1}', zorder=2)
                self.ax_curve.plot(es_idx + 1, lv_data[es_idx], marker='o', markersize=7,
                                  color='#ff6b6b', markeredgecolor='#ffffff', markeredgewidth=1.2, zorder=6)

            # Mark Current Phase
            self.ax_curve.axvline(x=current_phase + 1, color='#00ffff', linestyle='-', linewidth=2.2,
                                  alpha=0.95, label=f'Current Phase ({current_phase + 1})', zorder=5)

            self.ax_curve.set_xlim(0.5, total_phases + 0.5)

            m = med_image.cardiac_metrics
            if m and 'lv_ef' in m:
                ef_val = m['lv_ef']
                ed_val = lv_data[ed_idx] if ed_idx < len(lv_data) else 0
                es_val = lv_data[es_idx] if es_idx < len(lv_data) else 0
                sv_val = max(0.0, ed_val - es_val)
                ed_str = val_fmt.format(ed_val)
                es_str = val_fmt.format(es_val)
                sv_str = val_fmt.format(sv_val)
                title_curve = (
                    f"{mode_title}  |  ED: Phase {ed_idx + 1} ({ed_str})  |  "
                    f"ES: Phase {es_idx + 1} ({es_str})  |  "
                    f"LV EF: {ef_val:.1f}%  |  SV: {sv_str}"
                )
            else:
                title_curve = f"{mode_title}  |  Current Phase: {current_phase + 1}/{total_phases}"

            self.ax_curve.set_title(title_curve, color='#00adb5', fontsize=9.5, fontweight='bold', pad=4)
            self.ax_curve.legend(loc='upper right', facecolor='#21242d', edgecolor='#3b4252',
                                 labelcolor='#eceff4', fontsize=7.5, framealpha=0.85, ncol=3)

        else:
            self.ax_curve.set_ylabel("Volume / Area", color='#8f9ba8', fontsize=8.5)
            if total_phases > 1:
                self.ax_curve.set_xlim(0.5, total_phases + 0.5)
                self.ax_curve.axvline(x=current_phase + 1, color='#00ffff', linestyle='-', linewidth=2)

            self.ax_curve.text(
                0.5, 0.5,
                "Mask Not Loaded: Run 'AI Quantification' or load 'segmentation_all_phases.mat'\n"
                "to display volume curves (LV, RV, MYO) and automatic ED/ES detection.",
                horizontalalignment='center', verticalalignment='center',
                transform=self.ax_curve.transAxes, color='#8f9ba8', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#21242d', edgecolor='#3b4252', alpha=0.8)
            )
            self.ax_curve.set_title("Cardiac Volume Curves (Requires Segmentation)",
                                    color='#6b7280', fontsize=9, fontstyle='italic', pad=4)

        self.draw_idle()

    def on_mouse_click(self, event):
        # Click on 2D image -> select voxel
        if event.inaxes == self.ax_img and event.xdata is not None and event.ydata is not None:
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            self.selected_row = row
            self.selected_col = col
            self.pixel_clicked.emit(row, col)

        # Click on volume curve -> jump to cardiac phase
        elif event.inaxes == self.ax_curve and event.xdata is not None:
            target_phase = int(round(event.xdata)) - 1
            if target_phase >= 0:
                self.phase_clicked.emit(target_phase)

    def on_mouse_move(self, event):
        if event.inaxes == self.ax_img and event.xdata is not None and event.ydata is not None:
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            if self.status_callback and self.img_artist is not None:
                data = self.img_artist.get_array()
                if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                    val = data[row, col]
                    self.status_callback(row, col, val)

    def on_scroll(self, event):
        if event.step > 0:
            delta = 1
        elif event.step < 0:
            delta = -1
        else:
            return

        if event.key == 'control' or event.key == 'shift':
            self.phase_scroll_requested.emit(delta)
        else:
            self.slice_scroll_requested.emit(delta)


# ==============================================================================
# MULTI-VIEW CARDIAC MOSAIC (Cardiac Cycle, Z-Stack, and ED vs ES Comparator)
# ==============================================================================
class MultiSliceGridWidget(QWidget):
    """High-occupancy multi-view grid displaying Cardiac Cycle Phases, Z-Stack Slices, or ED vs ES Comparison."""

    slice_selected = pyqtSignal(int)
    phase_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(3)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #1a1b22; }")
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.fig = Figure(facecolor='#1a1b22')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_area.setWidget(self.canvas)
        self.layout.addWidget(self.scroll_area, 1)

        self.axes = []
        self.tile_metadata = []
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Bottom control toolbar
        self.control_bar = QFrame(self)
        self.control_bar.setStyleSheet(
            "QFrame { background-color: #21242d; border-top: 1px solid #2d313e; border-radius: 6px; padding: 3px 6px; }"
        )
        cb_layout = QHBoxLayout(self.control_bar)
        cb_layout.setContentsMargins(6, 4, 6, 4)
        cb_layout.setSpacing(8)

        # Mode Selector
        cb_layout.addWidget(QLabel("<b>Mode:</b>"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Cardiac Cycle (Phases)", "phases")
        self.combo_mode.addItem("Multi-Slice (Z-Stack)", "slices")
        self.combo_mode.addItem("ED vs ES Comparison", "edes_compare")
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        cb_layout.addWidget(self.combo_mode)

        # Columns Selector
        cb_layout.addWidget(QLabel("<b>Columns:</b>"))
        self.combo_cols = QComboBox()
        self.combo_cols.addItem("Auto (Optimal Fit)", "auto")
        self.combo_cols.addItem("3 Columns (Large)", "3")
        self.combo_cols.addItem("4 Columns", "4")
        self.combo_cols.addItem("5 Columns", "5")
        self.combo_cols.addItem("6 Columns", "6")
        self.combo_cols.addItem("7 Columns", "7")
        self.combo_cols.addItem("8 Columns", "8")
        self.combo_cols.addItem("10 Columns", "10")
        self.combo_cols.addItem("12 Columns", "12")
        self.combo_cols.currentIndexChanged.connect(self.on_cols_changed)
        cb_layout.addWidget(self.combo_cols)

        # Cardiac ROI Focus Checkbox
        self.chk_roi_focus = QCheckBox("Cardiac ROI Focus")
        self.chk_roi_focus.setToolTip("Automatically zoom and center on the ventricular region for maximum cardiac detail")
        self.chk_roi_focus.setChecked(True)
        self.chk_roi_focus.toggled.connect(self.on_toggle_roi)
        cb_layout.addWidget(self.chk_roi_focus)

        # Segmentation Overlay Checkbox
        self.chk_mosaic_mask = QCheckBox("Mask")
        self.chk_mosaic_mask.setChecked(True)
        self.chk_mosaic_mask.toggled.connect(self.on_toggle_mask)
        cb_layout.addWidget(self.chk_mosaic_mask)

        # Opacity Slider
        cb_layout.addWidget(QLabel("Opacity:"))
        self.slider_mosaic_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_mosaic_opacity.setRange(5, 100)
        self.slider_mosaic_opacity.setValue(45)
        self.slider_mosaic_opacity.setFixedWidth(75)
        self.slider_mosaic_opacity.valueChanged.connect(self.on_opacity_changed)
        cb_layout.addWidget(self.slider_mosaic_opacity)

        self.lbl_opacity_val = QLabel("45%")
        self.lbl_opacity_val.setStyleSheet("color: #8f9ba8; font-size: 11px;")
        cb_layout.addWidget(self.lbl_opacity_val)

        cb_layout.addStretch()

        # Jump to ED/ES buttons
        self.btn_go_ed = QPushButton("Jump to ED")
        self.btn_go_ed.setObjectName("edButton")
        self.btn_go_ed.clicked.connect(self.jump_to_ed)
        cb_layout.addWidget(self.btn_go_ed)

        self.btn_go_es = QPushButton("Jump to ES")
        self.btn_go_es.setObjectName("esButton")
        self.btn_go_es.clicked.connect(self.jump_to_es)
        cb_layout.addWidget(self.btn_go_es)

        # Save Mosaic Button
        self.btn_export_mosaic = QPushButton("Save Mosaic")
        self.btn_export_mosaic.setToolTip("Export entire mosaic grid to high-resolution PNG image")
        self.btn_export_mosaic.clicked.connect(self.export_mosaic_image)
        cb_layout.addWidget(self.btn_export_mosaic)

        self.layout.addWidget(self.control_bar, 0)

        # Internal state
        self.med_img = None
        self.current_slice = 0
        self.current_phase = 0
        self.vmin = 0.0
        self.vmax = 1.0
        self.cmap_name = 'bone'
        self.show_mask = True
        self.mask_opacity = 0.45
        self.mosaic_mode = "phases"
        self.cols_mode = "auto"
        self.roi_focus = True

    def on_mode_changed(self, idx):
        self.mosaic_mode = self.combo_mode.currentData()
        self.refresh_grid()

    def on_cols_changed(self, idx):
        self.cols_mode = self.combo_cols.currentData()
        self.refresh_grid()

    def on_toggle_roi(self, checked):
        self.roi_focus = checked
        self.refresh_grid()

    def on_toggle_mask(self, checked):
        self.show_mask = checked
        self.refresh_grid()

    def on_opacity_changed(self, val):
        self.mask_opacity = val / 100.0
        self.lbl_opacity_val.setText(f"{val}%")
        self.refresh_grid()

    def jump_to_ed(self):
        if self.med_img and self.med_img.has_mask:
            self.phase_selected.emit(self.med_img.ed_phase)

    def jump_to_es(self):
        if self.med_img and self.med_img.has_mask:
            self.phase_selected.emit(self.med_img.es_phase)

    def export_mosaic_image(self):
        if self.med_img is None:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Cardiac Mosaic", f"cardiac_mosaic_{self.mosaic_mode}.png",
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if filepath:
            try:
                self.fig.savefig(filepath, dpi=300, facecolor='#1a1b22', edgecolor='none')
                QMessageBox.information(self, "Export Successful", f"Mosaic saved successfully to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save mosaic:\n{str(e)}")

    def update_grid(self, med_img, current_slice, current_phase, vmin, vmax, cmap_name):
        self.med_img = med_img
        self.current_slice = current_slice
        self.current_phase = current_phase
        self.vmin = vmin
        self.vmax = vmax
        self.cmap_name = cmap_name
        self.refresh_grid()

    def get_cardiac_roi_bounds(self):
        """Calculate spatial boundaries (r0, r1, c0, c1) of the ventricular region for centered zooming."""
        if self.med_img is None or self.med_img.data_4d is None:
            return 0, 100, 0, 100

        n_rows = self.med_img.num_rows
        n_cols = self.med_img.num_cols

        if self.med_img.has_mask and self.med_img.mask_4d is not None and np.any(self.med_img.mask_4d > 0):
            nonzeros = np.where(self.med_img.mask_4d > 0)
            min_r, max_r = int(np.min(nonzeros[0])), int(np.max(nonzeros[0]))
            min_c, max_c = int(np.min(nonzeros[1])), int(np.max(nonzeros[1]))

            pad_r = max(18, int((max_r - min_r) * 0.35))
            pad_c = max(18, int((max_c - min_c) * 0.35))

            r0 = max(0, min_r - pad_r)
            r1 = min(n_rows, max_r + pad_r)
            c0 = max(0, min_c - pad_c)
            c1 = min(n_cols, max_c + pad_c)

            h = r1 - r0
            w = c1 - c0
            side = max(h, w)
            center_r = (r0 + r1) / 2.0
            center_c = (c0 + c1) / 2.0

            r0 = max(0, int(round(center_r - side / 2.0)))
            r1 = min(n_rows, int(round(center_r + side / 2.0)))
            c0 = max(0, int(round(center_c - side / 2.0)))
            c1 = min(n_cols, int(round(center_c + side / 2.0)))

            return r0, r1, c0, c1
        else:
            r0 = int(n_rows * 0.20)
            r1 = int(n_rows * 0.80)
            c0 = int(n_cols * 0.20)
            c1 = int(n_cols * 0.80)
            return r0, r1, c0, c1

    def compute_optimal_grid(self, num_items, canvas_w, canvas_h, aspect_ratio):
        """Compute optimal row/column distribution to maximize thumbnail size."""
        if num_items <= 0:
            return 1, 1
        canvas_w = max(300, canvas_w)
        canvas_h = max(200, canvas_h)
        best_c = 1
        best_size = 0.0

        for c in range(1, num_items + 1):
            r = int(np.ceil(num_items / c))
            h_from_w = canvas_w / (c * aspect_ratio)
            h_from_h = canvas_h / r
            tile_h = min(h_from_w, h_from_h)
            if tile_h > best_size:
                best_size = tile_h
                best_c = c

        best_r = int(np.ceil(num_items / best_c))
        return best_c, best_r

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'cols_mode') and self.cols_mode == "auto":
            self.refresh_grid()

    def refresh_grid(self):
        if self.med_img is None or self.med_img.data_4d is None:
            self.fig.clear()
            self.axes = []
            self.tile_metadata = []
            ax = self.fig.add_subplot(1, 1, 1)
            ax.set_facecolor('#13141a')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#2d313e')
            ax.text(
                0.5, 0.5,
                "Multi-View Cardiac Mosaic\n\nOpen a 4D medical imaging file to view the cardiac cycle.",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='#8f9ba8', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#21242d', edgecolor='#3b4252', alpha=0.9)
            )
            self.canvas.draw_idle()
            return

        self.fig.clear()
        self.axes = []
        self.tile_metadata = []

        vp_w = max(400, self.scroll_area.viewport().width())
        vp_h = max(300, self.scroll_area.viewport().height())

        if self.roi_focus:
            roi_r0, roi_r1, roi_c0, roi_c1 = self.get_cardiac_roi_bounds()
            img_w = roi_c1 - roi_c0
            img_h = roi_r1 - roi_r0
        else:
            roi_r0, roi_r1 = 0, self.med_img.num_rows
            roi_c0, roi_c1 = 0, self.med_img.num_cols
            img_w = self.med_img.num_cols
            img_h = self.med_img.num_rows

        aspect = (img_w / img_h) if img_h > 0 else 1.0

        seg_colors = [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.25, 0.25, self.mask_opacity),
            (0.2, 0.85, 0.35, self.mask_opacity),
            (0.2, 0.6, 1.0, self.mask_opacity)
        ]
        seg_cmap = ListedColormap(seg_colors)

        ed_phase = self.med_img.ed_phase if self.med_img.has_mask else -1
        es_phase = self.med_img.es_phase if self.med_img.has_mask else -1

        if self.med_img.has_mask:
            self.btn_go_ed.setText(f"Jump to ED (Phase {ed_phase + 1})")
            self.btn_go_es.setText(f"Jump to ES (Phase {es_phase + 1})")
            self.btn_go_ed.setEnabled(True)
            self.btn_go_es.setEnabled(True)
        else:
            self.btn_go_ed.setText("Jump to ED")
            self.btn_go_es.setText("Jump to ES")
            self.btn_go_ed.setEnabled(False)
            self.btn_go_es.setEnabled(False)

        # ----------------------------------------------------------------------
        # MODE 1: CARDIAC PHASES MOSAIC (CARDIAC CYCLE FOR CURRENT SLICE)
        # ----------------------------------------------------------------------
        if self.mosaic_mode == "phases":
            n_items = self.med_img.num_phases

            if self.cols_mode == "auto":
                cols, rows = self.compute_optimal_grid(n_items, vp_w - 10, vp_h - 35, aspect)
                self.canvas.setMinimumSize(0, 0)
                self.canvas.resize(vp_w, vp_h)
            else:
                cols = max(1, int(self.cols_mode))
                rows = int(np.ceil(n_items / cols))
                col_w = max(150, (vp_w - 20) / cols)
                row_h = col_w / aspect + 22
                needed_h = int(rows * row_h + 35)
                self.canvas.setMinimumSize(vp_w - 10, max(vp_h, needed_h))
                self.canvas.resize(vp_w - 10, max(vp_h, needed_h))

            header = f"Cardiac Cycle Mosaic — Slice {self.current_slice + 1}/{self.med_img.num_slices}"
            if self.med_img.has_mask:
                header += f"  |  ED: Phase {ed_phase + 1}   ES: Phase {es_phase + 1}"
            self.fig.suptitle(header, color='#4da8da', fontsize=10.5, fontweight='bold', y=0.988)

            for p in range(n_items):
                ax = self.fig.add_subplot(rows, cols, p + 1)
                ax.set_facecolor('#13141a')
                slice_data = self.med_img.get_slice_2d(self.current_slice, p)
                ax.imshow(slice_data, cmap=self.cmap_name, vmin=self.vmin, vmax=self.vmax,
                          interpolation='bilinear', origin='upper')

                if self.show_mask and self.med_img.has_mask:
                    mask_data = self.med_img.get_mask_slice_2d(self.current_slice, p)
                    if mask_data is not None:
                        ax.imshow(mask_data, cmap=seg_cmap, vmin=0, vmax=3,
                                  interpolation='nearest', origin='upper')

                if self.roi_focus:
                    ax.set_xlim(roi_c0, roi_c1)
                    ax.set_ylim(roi_r1, roi_r0)

                is_ed = (p == ed_phase)
                is_es = (p == es_phase)
                is_curr = (p == self.current_phase)

                if is_ed:
                    title_str = f"PHASE {p + 1} (ED)"
                    title_color = '#51cf66'
                    border_color = '#51cf66'
                    border_width = 2.2
                elif is_es:
                    title_str = f"PHASE {p + 1} (ES)"
                    title_color = '#ff6b6b'
                    border_color = '#ff6b6b'
                    border_width = 2.2
                elif is_curr:
                    title_str = f"Phase {p + 1} (Curr.)"
                    title_color = '#00ffff'
                    border_color = '#00ffff'
                    border_width = 1.6
                else:
                    title_str = f"Phase {p + 1}"
                    title_color = '#8f9ba8'
                    border_color = '#2d313e'
                    border_width = 0.8

                ax.set_title(title_str, color=title_color, fontsize=8.0,
                             fontweight='bold' if (is_ed or is_es or is_curr) else 'normal', pad=1.5)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(border_width)

                self.axes.append(ax)
                self.tile_metadata.append(('phase', p))

        # ----------------------------------------------------------------------
        # MODE 2: MULTI-SLICE MOSAIC (ALL Z-SLICES FOR CURRENT PHASE)
        # ----------------------------------------------------------------------
        elif self.mosaic_mode == "slices":
            n_items = self.med_img.num_slices

            if self.cols_mode == "auto":
                cols, rows = self.compute_optimal_grid(n_items, vp_w - 10, vp_h - 35, aspect)
                self.canvas.setMinimumSize(0, 0)
                self.canvas.resize(vp_w, vp_h)
            else:
                cols = max(1, int(self.cols_mode))
                rows = int(np.ceil(n_items / cols))
                col_w = max(150, (vp_w - 20) / cols)
                row_h = col_w / aspect + 22
                needed_h = int(rows * row_h + 35)
                self.canvas.setMinimumSize(vp_w - 10, max(vp_h, needed_h))
                self.canvas.resize(vp_w - 10, max(vp_h, needed_h))

            phase_tag = ""
            if self.med_img.has_mask:
                if self.current_phase == ed_phase:
                    phase_tag = " — [END-DIASTOLE - ED]"
                elif self.current_phase == es_phase:
                    phase_tag = " — [END-SYSTOLE - ES]"

            header = f"Multi-Slice Z-Stack Mosaic — Cardiac Phase {self.current_phase + 1}/{self.med_img.num_phases}{phase_tag}"
            self.fig.suptitle(header, color='#4da8da' if not phase_tag else ('#51cf66' if 'DIASTOLE' in phase_tag else '#ff6b6b'),
                              fontsize=10.5, fontweight='bold', y=0.988)

            for s in range(n_items):
                ax = self.fig.add_subplot(rows, cols, s + 1)
                ax.set_facecolor('#13141a')
                slice_data = self.med_img.get_slice_2d(s, self.current_phase)
                ax.imshow(slice_data, cmap=self.cmap_name, vmin=self.vmin, vmax=self.vmax,
                          interpolation='bilinear', origin='upper')

                if self.show_mask and self.med_img.has_mask:
                    mask_data = self.med_img.get_mask_slice_2d(s, self.current_phase)
                    if mask_data is not None:
                        ax.imshow(mask_data, cmap=seg_cmap, vmin=0, vmax=3,
                                  interpolation='nearest', origin='upper')

                if self.roi_focus:
                    ax.set_xlim(roi_c0, roi_c1)
                    ax.set_ylim(roi_r1, roi_r0)

                is_curr = (s == self.current_slice)
                border_color = '#007acc' if is_curr else '#2d313e'
                border_width = 1.8 if is_curr else 0.8

                ax.set_title(f"Slice {s + 1}" + (" (Curr.)" if is_curr else ""),
                             color='#4da8da' if is_curr else '#8f9ba8', fontsize=8.0,
                             fontweight='bold' if is_curr else 'normal', pad=1.5)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(border_width)

                self.axes.append(ax)
                self.tile_metadata.append(('slice', s))

        # ----------------------------------------------------------------------
        # MODE 3: DIRECT COMPARISON ED vs ES (SIDE BY SIDE)
        # ----------------------------------------------------------------------
        elif self.mosaic_mode == "edes_compare":
            n_slices = self.med_img.num_slices
            p_ed = ed_phase if ed_phase >= 0 else 0
            p_es = es_phase if es_phase >= 0 else min(1, self.med_img.num_phases - 1)

            if self.cols_mode == "auto":
                if n_slices <= 14 and vp_w >= 1000:
                    cols = n_slices
                    rows_per_group = 1
                else:
                    cols, _ = self.compute_optimal_grid(n_slices, vp_w - 10, (vp_h - 35) / 2, aspect)
                    rows_per_group = int(np.ceil(n_slices / cols))

                total_rows = rows_per_group * 2
                self.canvas.setMinimumSize(0, 0)
                self.canvas.resize(vp_w, vp_h)
            else:
                cols = max(1, int(self.cols_mode))
                rows_per_group = int(np.ceil(n_slices / cols))
                total_rows = rows_per_group * 2
                col_w = max(150, (vp_w - 20) / cols)
                row_h = col_w / aspect + 22
                needed_h = int(total_rows * row_h + 35)
                self.canvas.setMinimumSize(vp_w - 10, max(vp_h, needed_h))
                self.canvas.resize(vp_w - 10, max(vp_h, needed_h))

            header = f"Cardiac Comparison: End-Diastole (Phase {p_ed + 1}) vs End-Systole (Phase {p_es + 1})"
            self.fig.suptitle(header, color='#4da8da', fontsize=10.5, fontweight='bold', y=0.988)

            # Top Row: End-Diastole (ED)
            for s in range(n_slices):
                r = s // cols
                c = s % cols
                ax = self.fig.add_subplot(total_rows, cols, r * cols + c + 1)
                ax.set_facecolor('#13141a')
                slice_data = self.med_img.get_slice_2d(s, p_ed)
                ax.imshow(slice_data, cmap=self.cmap_name, vmin=self.vmin, vmax=self.vmax,
                          interpolation='bilinear', origin='upper')

                if self.show_mask and self.med_img.has_mask:
                    mask_data = self.med_img.get_mask_slice_2d(s, p_ed)
                    if mask_data is not None:
                        ax.imshow(mask_data, cmap=seg_cmap, vmin=0, vmax=3,
                                  interpolation='nearest', origin='upper')

                if self.roi_focus:
                    ax.set_xlim(roi_c0, roi_c1)
                    ax.set_ylim(roi_r1, roi_r0)

                ax.set_title(f"Slice {s + 1} (ED)", color='#51cf66', fontsize=8.0, fontweight='bold', pad=1.5)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color('#2b8a3e')
                    spine.set_linewidth(1.4)
                self.axes.append(ax)
                self.tile_metadata.append(('compare_ed', s))

            # Bottom Row: End-Systole (ES)
            for s in range(n_slices):
                r = s // cols
                c = s % cols
                ax = self.fig.add_subplot(total_rows, cols, (rows_per_group + r) * cols + c + 1)
                ax.set_facecolor('#13141a')
                slice_data = self.med_img.get_slice_2d(s, p_es)
                ax.imshow(slice_data, cmap=self.cmap_name, vmin=self.vmin, vmax=self.vmax,
                          interpolation='bilinear', origin='upper')

                if self.show_mask and self.med_img.has_mask:
                    mask_data = self.med_img.get_mask_slice_2d(s, p_es)
                    if mask_data is not None:
                        ax.imshow(mask_data, cmap=seg_cmap, vmin=0, vmax=3,
                                  interpolation='nearest', origin='upper')

                if self.roi_focus:
                    ax.set_xlim(roi_c0, roi_c1)
                    ax.set_ylim(roi_r1, roi_r0)

                ax.set_title(f"Slice {s + 1} (ES)", color='#ff6b6b', fontsize=8.0, fontweight='bold', pad=1.5)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color('#c92a2a')
                    spine.set_linewidth(1.4)
                self.axes.append(ax)
                self.tile_metadata.append(('compare_es', s))

        self.fig.subplots_adjust(
            left=0.005, right=0.995,
            top=0.952, bottom=0.008,
            wspace=0.015, hspace=0.065
        )

        self.canvas.draw_idle()

    def on_click(self, event):
        for idx, ax in enumerate(self.axes):
            if event.inaxes == ax and idx < len(self.tile_metadata):
                mtype, val = self.tile_metadata[idx]
                if mtype == 'phase':
                    self.phase_selected.emit(val)
                elif mtype == 'slice':
                    self.slice_selected.emit(val)
                elif mtype in ('compare_ed', 'compare_es'):
                    self.slice_selected.emit(val)
                break


# ==============================================================================
# INTEGRATED NEURAL INFERENCE ENGINE (VENTSEG AI)
# ==============================================================================
def save_mat_dict(filepath, data_dict):
    """Save dictionary to MATLAB .mat format."""
    try:
        scipy.io.savemat(filepath, data_dict)
    except Exception as e:
        print(f"Error saving {filepath}: {e}")


class SegmentationWorker(QThread):
    """Background worker executing integrated direct neural inference (ResNet34-UNet)."""
    progress_updated = pyqtSignal(int, int, str)
    status_message = pyqtSignal(str)
    segmentation_finished = pyqtSignal(dict)
    segmentation_error = pyqtSignal(str)

    def __init__(self, data_4d, file_path="", mode="fast_edes", target_phase=0,
                 device_str="auto", save_mat=True, voxel_size=None, output_dir=None, parent=None):
        super().__init__(parent)
        self.data_4d = data_4d
        self.file_path = file_path
        self.mode = mode
        self.target_phase = target_phase
        self.device_str = device_str
        self.save_mat = save_mat
        self.voxel_size = np.asarray(voxel_size, dtype=np.float64) if voxel_size is not None else np.array([0.5, 0.5, 8.0], dtype=np.float64)
        self.output_dir = output_dir
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        try:
            self._execute_segmentation()
        except Exception as e:
            import traceback
            err_msg = f"{str(e)}\n\nDetails:\n{traceback.format_exc()}"
            self.segmentation_error.emit(err_msg)

    def _execute_segmentation(self):
        """Directly run trained PyTorch ResNet34-UNet in-memory."""
        try:
            import torch
            import albumentations
            from models import model_selector
            from utils.data_augmentation import common_test_augmentation
            from utils.dataload import apply_normalization
            from utils.training import convert_multiclass_mask, reshape_masks
        except Exception as import_err:
            raise RuntimeError(
                f"Failed to import deep learning modules (torch, albumentations, models):\n{import_err}\n\n"
                "Please verify that the virtual environment with PyTorch and Albumentations is active."
            )

        self.status_message.emit("Initializing deep learning engine (ResNet34-UNet)...")

        # Configure compute device (GPU CUDA or CPU)
        if self.device_str == "cuda":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif self.device_str == "cpu":
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.status_message.emit(f"Selected compute device: {device}")

        # Instantiate model
        model = model_selector('resnet34_unet_scratch', num_classes=4, in_channels=1)
        if torch.cuda.is_available() and device.type == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)

        # Locate trained checkpoint
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "models", "model_resnet34_unet_scratch_best_dice.pt"),
            os.path.join(script_dir, "model_resnet34_unet_scratch_best_dice.pt"),
            os.path.join(os.getcwd(), "models", "model_resnet34_unet_scratch_best_dice.pt"),
            os.path.join(os.getcwd(), "model_resnet34_unet_scratch_best_dice.pt")
        ]
        ckpt_path = ""
        for p in candidates:
            if os.path.exists(p):
                ckpt_path = p
                break

        if not ckpt_path:
            raise FileNotFoundError(
                "Model weight file 'model_resnet34_unet_scratch_best_dice.pt' not found in models/ directory."
            )

        self.status_message.emit(f"Loading checkpoint: {os.path.basename(ckpt_path)}...")
        try:
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location=device)

        # DataParallel compatibility
        if not isinstance(model, torch.nn.DataParallel) and all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        elif isinstance(model, torch.nn.DataParallel) and not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()

        common_reshape = common_test_augmentation(224)
        transform = albumentations.Compose(common_reshape)

        def predict_slice(image_2d):
            img_aug = transform(image=image_2d)["image"]
            img_norm = apply_normalization(img_aug, 'standardize')
            tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(tensor)
            mask = convert_multiclass_mask(pred).data.cpu().numpy().squeeze(0)
            mask = reshape_masks(mask, image_2d.shape)
            return mask.astype(np.uint8)

        H, W, Z, T = self.data_4d.shape
        if self.output_dir and str(self.output_dir).strip():
            base_dir = os.path.abspath(str(self.output_dir).strip())
            os.makedirs(base_dir, exist_ok=True)
        elif self.file_path:
            base_dir = os.path.dirname(os.path.abspath(self.file_path))
        else:
            base_dir = os.getcwd()
        saved_files = []

        # -------------------------------------------------------------
        # MODE 1: Fast ED / ES Detection + 3D Diastolic & Systolic Volumes
        # -------------------------------------------------------------
        if self.mode == "fast_edes":
            mid_z = Z // 2
            total_steps = T + 2 * Z
            step = 0

            self.status_message.emit("Step 1/3: Segmenting central slice across all cardiac phases...")
            middle_mask = np.zeros((H, W, T), dtype=np.uint8)
            for t in range(T):
                if self.is_cancelled:
                    return
                middle_mask[:, :, t] = predict_slice(self.data_4d[:, :, mid_z, t])
                step += 1
                self.progress_updated.emit(step, total_steps, f"Central Slice ({mid_z + 1}/{Z}) - Phase {t + 1}/{T}")

            # Identify ED and ES by LV area in central slice
            lv_areas = [np.sum(middle_mask[:, :, t] == 1) for t in range(T)]
            if np.max(lv_areas) > 0:
                ed_p = int(np.argmax(lv_areas))
                es_p = int(np.argmin(lv_areas))
                if ed_p == es_p and T > 1:
                    es_p = (ed_p + T // 2) % T
            else:
                ed_p = 0
                es_p = min(1, T - 1)

            self.status_message.emit(f"Step 2/3: Detected key phases -> ED: Phase {ed_p + 1} | ES: Phase {es_p + 1}")

            self.status_message.emit(f"Step 3/3: Segmenting 3D volume at End-Diastole (Phase {ed_p + 1})...")
            ed_3d = np.zeros((H, W, Z), dtype=np.uint8)
            for z in range(Z):
                if self.is_cancelled:
                    return
                ed_3d[:, :, z] = predict_slice(self.data_4d[:, :, z, ed_p])
                step += 1
                self.progress_updated.emit(step, total_steps, f"ED Volume (Phase {ed_p + 1}) - Slice {z + 1}/{Z}")

            self.status_message.emit(f"Step 3/3: Segmenting 3D volume at End-Systole (Phase {es_p + 1})...")
            es_3d = np.zeros((H, W, Z), dtype=np.uint8)
            for z in range(Z):
                if self.is_cancelled:
                    return
                es_3d[:, :, z] = predict_slice(self.data_4d[:, :, z, es_p])
                step += 1
                self.progress_updated.emit(step, total_steps, f"ES Volume (Phase {es_p + 1}) - Slice {z + 1}/{Z}")

            # Combined 4D Mask
            mask_4d = np.zeros((H, W, Z, T), dtype=np.uint8)
            mask_4d[:, :, mid_z, :] = middle_mask
            mask_4d[:, :, :, ed_p] = ed_3d
            mask_4d[:, :, :, es_p] = es_3d

            if self.save_mat:
                save_mat_dict(os.path.join(base_dir, 'segmentation_all_phases.mat'), {
                    'segmentation_all_phases': mask_4d,
                    'voxel_size': self.voxel_size,
                    'ed_phase': ed_p + 1,
                    'es_phase': es_p + 1
                })
                save_mat_dict(os.path.join(base_dir, 'segmentation_ED.mat'), {
                    'segmentation_ED': ed_3d,
                    'voxel_size': self.voxel_size,
                    'ed_phase': ed_p + 1
                })
                save_mat_dict(os.path.join(base_dir, 'segmentation_ES.mat'), {
                    'segmentation_ES': es_3d,
                    'voxel_size': self.voxel_size,
                    'es_phase': es_p + 1
                })
                save_mat_dict(os.path.join(base_dir, 'image_SA.mat'), {
                    'image_SA': self.data_4d,
                    'voxel_size': self.voxel_size
                })
                saved_files = ['segmentation_all_phases.mat', 'segmentation_ED.mat', 'segmentation_ES.mat', 'image_SA.mat']

            self.status_message.emit("Quantification and segmentation completed successfully.")
            self.segmentation_finished.emit({
                'mask_4d': mask_4d,
                'mode': self.mode,
                'ed_phase': ed_p,
                'es_phase': es_p,
                'mask_filename': f"ED/ES Quantification (Phases {ed_p + 1} and {es_p + 1})",
                'saved_files': saved_files,
                'output_dir': base_dir
            })

        # -------------------------------------------------------------
        # MODE 2: Full 4D Spatio-Temporal Segmentation
        # -------------------------------------------------------------
        elif self.mode == "full_4d":
            total_steps = T * Z
            step = 0
            mask_4d = np.zeros((H, W, Z, T), dtype=np.uint8)

            for t in range(T):
                for z in range(Z):
                    if self.is_cancelled:
                        return
                    mask_4d[:, :, z, t] = predict_slice(self.data_4d[:, :, z, t])
                    step += 1
                    self.progress_updated.emit(step, total_steps, f"Phase {t + 1}/{T}, Slice {z + 1}/{Z}")

            lv_volumes = [np.sum(mask_4d[:, :, :, t] == 1) for t in range(T)]
            if np.max(lv_volumes) > 0:
                ed_p = int(np.argmax(lv_volumes))
                es_p = int(np.argmin(lv_volumes))
            else:
                ed_p = 0
                es_p = min(1, T - 1)

            ed_3d = mask_4d[:, :, :, ed_p]
            es_3d = mask_4d[:, :, :, es_p]

            if self.save_mat:
                save_mat_dict(os.path.join(base_dir, 'segmentation_all_phases.mat'), {
                    'segmentation_all_phases': mask_4d,
                    'voxel_size': self.voxel_size,
                    'ed_phase': ed_p + 1,
                    'es_phase': es_p + 1
                })
                save_mat_dict(os.path.join(base_dir, 'segmentation_ED.mat'), {
                    'segmentation_ED': ed_3d,
                    'voxel_size': self.voxel_size,
                    'ed_phase': ed_p + 1
                })
                save_mat_dict(os.path.join(base_dir, 'segmentation_ES.mat'), {
                    'segmentation_ES': es_3d,
                    'voxel_size': self.voxel_size,
                    'es_phase': es_p + 1
                })
                save_mat_dict(os.path.join(base_dir, 'image_SA.mat'), {
                    'image_SA': self.data_4d,
                    'voxel_size': self.voxel_size
                })
                saved_files = ['segmentation_all_phases.mat', 'segmentation_ED.mat', 'segmentation_ES.mat', 'image_SA.mat']

            self.status_message.emit("4D spatio-temporal segmentation completed successfully.")
            self.segmentation_finished.emit({
                'mask_4d': mask_4d,
                'mode': self.mode,
                'ed_phase': ed_p,
                'es_phase': es_p,
                'mask_filename': "segmentation_all_phases.mat (Full 4D)",
                'saved_files': saved_files,
                'output_dir': base_dir
            })

        # -------------------------------------------------------------
        # MODE 3: Central Slice Only
        # -------------------------------------------------------------
        elif self.mode == "middle":
            mid_z = Z // 2
            total_steps = T
            middle_mask = np.zeros((H, W, T), dtype=np.uint8)

            for t in range(T):
                if self.is_cancelled:
                    return
                middle_mask[:, :, t] = predict_slice(self.data_4d[:, :, mid_z, t])
                self.progress_updated.emit(t + 1, total_steps, f"Central Slice ({mid_z + 1}/{Z}) - Phase {t + 1}/{T}")

            mask_4d = np.zeros((H, W, Z, T), dtype=np.uint8)
            mask_4d[:, :, mid_z, :] = middle_mask

            lv_areas = [np.sum(middle_mask[:, :, t] == 1) for t in range(T)]
            if np.max(lv_areas) > 0:
                ed_p = int(np.argmax(lv_areas))
                es_p = int(np.argmin(lv_areas))
            else:
                ed_p = 0
                es_p = min(1, T - 1)

            if self.save_mat:
                save_mat_dict(os.path.join(base_dir, 'segmentation_all_phases.mat'), {
                    'segmentation_all_phases': mask_4d,
                    'voxel_size': self.voxel_size,
                    'ed_phase': ed_p + 1,
                    'es_phase': es_p + 1
                })
                save_mat_dict(os.path.join(base_dir, 'image_SA.mat'), {
                    'image_SA': self.data_4d,
                    'voxel_size': self.voxel_size
                })
                saved_files = ['segmentation_all_phases.mat', 'image_SA.mat']

            self.status_message.emit("Central slice segmentation completed.")
            self.segmentation_finished.emit({
                'mask_4d': mask_4d,
                'mode': self.mode,
                'ed_phase': ed_p,
                'es_phase': es_p,
                'mask_filename': "segmentation_all_phases.mat (Central Slice)",
                'saved_files': saved_files,
                'output_dir': base_dir
            })

        # -------------------------------------------------------------
        # MODE 4: Single Target Phase
        # -------------------------------------------------------------
        elif self.mode == "single_phase":
            p = max(0, min(self.target_phase, T - 1))
            total_steps = Z
            phase_3d = np.zeros((H, W, Z), dtype=np.uint8)

            for z in range(Z):
                if self.is_cancelled:
                    return
                phase_3d[:, :, z] = predict_slice(self.data_4d[:, :, z, p])
                self.progress_updated.emit(z + 1, total_steps, f"Phase {p + 1}/{T} - Slice {z + 1}/{Z}")

            mask_4d = np.zeros((H, W, Z, T), dtype=np.uint8)
            mask_4d[:, :, :, p] = phase_3d

            if self.save_mat:
                phase_filename = f"segmentation_phase_{p + 1}.mat"
                save_mat_dict(os.path.join(base_dir, phase_filename), {
                    'segmentation_phase': phase_3d,
                    'voxel_size': self.voxel_size,
                    'phase': p + 1
                })
                save_mat_dict(os.path.join(base_dir, 'segmentation_all_phases.mat'), {
                    'segmentation_all_phases': mask_4d,
                    'voxel_size': self.voxel_size
                })
                save_mat_dict(os.path.join(base_dir, 'image_SA.mat'), {
                    'image_SA': self.data_4d,
                    'voxel_size': self.voxel_size
                })
                saved_files = [phase_filename, 'segmentation_all_phases.mat', 'image_SA.mat']

            self.status_message.emit(f"Phase {p + 1} segmentation completed.")
            self.segmentation_finished.emit({
                'mask_4d': mask_4d,
                'mode': self.mode,
                'ed_phase': p,
                'es_phase': p,
                'mask_filename': f"segmentation_phase_{p + 1}.mat (Phase {p + 1})",
                'saved_files': saved_files,
                'output_dir': base_dir
            })


# ==============================================================================
# SPATIAL CALIBRATION (VOXEL SIZE), SEGMENTATION AND CLINICAL REPORT DIALOGS
# ==============================================================================
class VoxelSizeDialog(QDialog):
    """Interactive dialog for inspecting and calibrating physical voxel dimensions (in mm)."""

    def __init__(self, parent=None, med_image=None):
        super().__init__(parent)
        self.med_image = med_image
        self.setWindowTitle("Spatial Calibration & Voxel Size (mm)")
        self.setWindowIcon(get_app_icon())
        self.resize(540, 430)
        self.setStyleSheet(DARK_STYLE_SHEET)

        curr_vx = self.med_image.voxel_size if self.med_image is not None else np.array([1.0, 1.0, 1.0])

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #21242d; border: 1px solid #2d313e; border-radius: 6px; padding: 8px;")
        header_layout = QVBoxLayout(header_frame)
        lbl_h = QLabel(
            "<b>Physical Medical Image Calibration</b><br>"
            "Physical voxel dimensions (in-plane pixel spacing dx, dy and slice thickness dz) allow accurate "
            "conversion of raw voxel counts into <b>milliliters (mL / cm³)</b> and areas into <b>cm²</b> "
            "for physiological clinical metrics."
        )
        lbl_h.setStyleSheet("color: #e0e6ed; font-size: 11.5px;")
        lbl_h.setWordWrap(True)
        header_layout.addWidget(lbl_h)
        layout.addWidget(header_frame)

        # Presets
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("<b>Clinical Presets:</b>"))
        self.combo_presets = QComboBox()
        self.combo_presets.addItem("Custom", None)
        self.combo_presets.addItem("Cardiac Cine MRI (0.50 × 0.50 × 8.00 mm)", [0.5, 0.5, 8.0])
        self.combo_presets.addItem("Standard Cardiac MRI (1.25 × 1.25 × 8.00 mm)", [1.25, 1.25, 8.0])
        self.combo_presets.addItem("Cardiac MRI 1.4 mm (1.40 × 1.40 × 10.00 mm)", [1.4, 1.4, 10.0])
        self.combo_presets.addItem("Isotropic 1 mm (1.00 × 1.00 × 1.00 mm)", [1.0, 1.0, 1.0])
        self.combo_presets.addItem("Isotropic 0.5 mm (0.50 × 0.50 × 0.50 mm)", [0.5, 0.5, 0.5])
        self.combo_presets.currentIndexChanged.connect(self.on_preset_selected)
        preset_row.addWidget(self.combo_presets)
        layout.addLayout(preset_row)

        # Spinboxes for dx, dy, dz
        inputs_group = QGroupBox("Physical Voxel Dimensions (in millimeters - mm)")
        inputs_layout = QGridLayout(inputs_group)
        inputs_layout.setSpacing(10)

        # dx (column spacing / X)
        inputs_layout.addWidget(QLabel("<b>dx (Column Spacing / X):</b>"), 0, 0)
        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(0.01, 100.0)
        self.spin_dx.setDecimals(4)
        self.spin_dx.setSingleStep(0.05)
        self.spin_dx.setValue(float(curr_vx[0]))
        self.spin_dx.setSuffix(" mm")
        self.spin_dx.valueChanged.connect(self.update_computed_info)
        inputs_layout.addWidget(self.spin_dx, 0, 1)

        # dy (row spacing / Y)
        inputs_layout.addWidget(QLabel("<b>dy (Row Spacing / Y):</b>"), 1, 0)
        self.spin_dy = QDoubleSpinBox()
        self.spin_dy.setRange(0.01, 100.0)
        self.spin_dy.setDecimals(4)
        self.spin_dy.setSingleStep(0.05)
        self.spin_dy.setValue(float(curr_vx[1]))
        self.spin_dy.setSuffix(" mm")
        self.spin_dy.valueChanged.connect(self.update_computed_info)
        inputs_layout.addWidget(self.spin_dy, 1, 1)

        # dz (slice thickness / Z)
        inputs_layout.addWidget(QLabel("<b>dz (Slice Thickness / Z):</b>"), 2, 0)
        self.spin_dz = QDoubleSpinBox()
        self.spin_dz.setRange(0.01, 100.0)
        self.spin_dz.setDecimals(4)
        self.spin_dz.setSingleStep(0.5)
        self.spin_dz.setValue(float(curr_vx[2]))
        self.spin_dz.setSuffix(" mm")
        self.spin_dz.valueChanged.connect(self.update_computed_info)
        inputs_layout.addWidget(self.spin_dz, 2, 1)

        layout.addWidget(inputs_group)

        # Computed summary
        calc_frame = QFrame()
        calc_frame.setStyleSheet("background-color: #1a1b22; border: 1px dashed #3b4252; border-radius: 6px; padding: 8px;")
        calc_layout = QVBoxLayout(calc_frame)
        self.lbl_calc_vox_vol = QLabel("Single Voxel Volume: -")
        self.lbl_calc_vox_vol.setStyleSheet("color: #74c0fc; font-size: 11.5px; font-weight: bold;")
        self.lbl_calc_pix_area = QLabel("2D Single Pixel Area: -")
        self.lbl_calc_pix_area.setStyleSheet("color: #51cf66; font-size: 11.5px; font-weight: bold;")
        calc_layout.addWidget(self.lbl_calc_vox_vol)
        calc_layout.addWidget(self.lbl_calc_pix_area)
        layout.addWidget(calc_frame)

        self.chk_save_orig_mat = QCheckBox("Persist voxel_size in image .mat file (and segmentation output files)")
        self.chk_save_orig_mat.setChecked(True)
        layout.addWidget(self.chk_save_orig_mat)

        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_apply = QPushButton("Apply Calibration")
        btn_apply.setObjectName("accentButton")
        btn_apply.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_apply)
        layout.addLayout(btn_layout)

        self.update_computed_info()

    def on_preset_selected(self, idx):
        val = self.combo_presets.currentData()
        if val is not None:
            self.spin_dx.setValue(val[0])
            self.spin_dy.setValue(val[1])
            self.spin_dz.setValue(val[2])

    def update_computed_info(self):
        dx = self.spin_dx.value()
        dy = self.spin_dy.value()
        dz = self.spin_dz.value()

        vol_mm3 = dx * dy * dz
        vol_ml = vol_mm3 / 1000.0

        area_mm2 = dx * dy
        area_cm2 = area_mm2 / 100.0

        self.lbl_calc_vox_vol.setText(
            f"Single Voxel Volume: {vol_mm3:.4f} mm³  =  {vol_ml:.6f} mL (cm³)"
        )
        self.lbl_calc_pix_area.setText(
            f"2D Pixel Area: {area_mm2:.4f} mm²  =  {area_cm2:.6f} cm²"
        )

    def get_voxel_size(self):
        return np.array([self.spin_dx.value(), self.spin_dy.value(), self.spin_dz.value()], dtype=np.float64)

    def should_save_to_mat(self):
        return self.chk_save_orig_mat.isChecked()


class SegmentationDialog(QDialog):
    """Interactive dialog to configure and launch deep learning cardiac segmentation and quantification."""

    def __init__(self, parent=None, med_image=None, current_phase=0, default_output_dir=""):
        super().__init__(parent)
        self.med_image = med_image
        self.current_phase = current_phase
        self.default_output_dir = default_output_dir or (os.path.dirname(os.path.abspath(self.med_image.filename)) if self.med_image and self.med_image.filename else os.getcwd())
        self.setWindowTitle("AI Cardiac Quantification & Segmentation (VENTSEG ResNet34-UNet)")
        self.setWindowIcon(get_app_icon())
        self.resize(640, 580)
        self.setStyleSheet(DARK_STYLE_SHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Image info header
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #21242d; border: 1px solid #2d313e; border-radius: 6px; padding: 8px;")
        info_layout = QVBoxLayout(info_frame)

        fname = os.path.basename(self.med_image.filename) if self.med_image and self.med_image.filename else "In-memory image"
        dims = f"{self.med_image.num_rows} × {self.med_image.num_cols} × {self.med_image.num_slices} slices × {self.med_image.num_phases} phases" if self.med_image else "-"
        vx_str = f"{self.med_image.voxel_size[0]:.2f} × {self.med_image.voxel_size[1]:.2f} × {self.med_image.voxel_size[2]:.2f} mm ({self.med_image.voxel_volume_ml:.4f} mL/vx)" if self.med_image else "-"

        lbl_info = QLabel(
            f"<b>Target Dataset:</b> {fname}<br>"
            f"<b>4D Dimensions:</b> {dims}<br>"
            f"<b>Voxel Dimensions:</b> <span style='color:#00adb5; font-weight:bold;'>{vx_str}</span>"
        )
        lbl_info.setStyleSheet("color: #e0e6ed; font-size: 11.5px;")
        info_layout.addWidget(lbl_info)
        layout.addWidget(info_frame)

        # Mode Selection
        mode_group = QGroupBox("Quantification / Segmentation Mode Selection")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(8)

        self.radio_fast_edes = QRadioButton("Fast ED / ES Detection + 3D Diastolic & Systolic Volumes (Recommended)")
        self.radio_fast_edes.setChecked(True)
        lbl_fast_desc = QLabel("   • Segments central slice to automatically detect ED and ES phases,\n     then segments all slices at those 2 key phases -> segmentation_ED.mat & segmentation_ES.mat.")
        lbl_fast_desc.setStyleSheet("color: #8f9ba8; font-size: 11px;")

        self.radio_full_4d = QRadioButton("Full 4D Spatio-Temporal Segmentation (All Phases & Slices -> segmentation_all_phases.mat)")
        lbl_full_desc = QLabel("   • Segments entire 4D spatio-temporal volume.\n     Generates segmentation_all_phases.mat, segmentation_ED.mat, and segmentation_ES.mat.")
        lbl_full_desc.setStyleSheet("color: #8f9ba8; font-size: 11px;")

        self.radio_middle = QRadioButton("Central Slice Only Across Cardiac Cycle (segmentation_all_phases.mat)")
        lbl_mid_desc = QLabel("   • Segments only the anatomical mid-cavity slice across all cardiac phases.")
        lbl_mid_desc.setStyleSheet("color: #8f9ba8; font-size: 11px;")

        self.radio_single = QRadioButton("Single Target Phase Only")
        phase_row = QHBoxLayout()
        phase_row.addWidget(self.radio_single)
        self.spin_phase = QSpinBox()
        max_p = self.med_image.num_phases if self.med_image else 1
        self.spin_phase.setRange(1, max(1, max_p))
        self.spin_phase.setValue(self.current_phase + 1)
        self.spin_phase.setPrefix("Phase ")
        phase_row.addWidget(self.spin_phase)
        phase_row.addStretch()

        mode_layout.addWidget(self.radio_fast_edes)
        mode_layout.addWidget(lbl_fast_desc)
        mode_layout.addWidget(self.radio_full_4d)
        mode_layout.addWidget(lbl_full_desc)
        mode_layout.addWidget(self.radio_middle)
        mode_layout.addWidget(lbl_mid_desc)
        mode_layout.addLayout(phase_row)

        layout.addWidget(mode_group)

        # Output Folder Group
        folder_group = QGroupBox("Output Directory for Results")
        folder_layout = QVBoxLayout(folder_group)
        folder_layout.setSpacing(6)

        self.chk_save_mat = QCheckBox("Save result files (.mat, report, curves, snapshots) to folder")
        self.chk_save_mat.setChecked(True)
        self.chk_save_mat.toggled.connect(self.on_save_toggled)
        folder_layout.addWidget(self.chk_save_mat)

        dest_row = QHBoxLayout()
        self.line_output_dir = QLineEdit(self.default_output_dir)
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_folder)
        dest_row.addWidget(self.line_output_dir)
        dest_row.addWidget(self.btn_browse)
        folder_layout.addLayout(dest_row)
        layout.addWidget(folder_group)

        # Advanced options (Compute Device)
        adv_row = QHBoxLayout()
        adv_row.addWidget(QLabel("Compute Device:"))
        self.combo_device = QComboBox()
        self.combo_device.addItem("Auto (GPU/CPU)", "auto")
        self.combo_device.addItem("GPU (CUDA)", "cuda")
        self.combo_device.addItem("CPU", "cpu")
        adv_row.addWidget(self.combo_device)
        adv_row.addStretch()
        layout.addLayout(adv_row)

        layout.addStretch()

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_start = QPushButton("Start AI Segmentation")
        btn_start.setObjectName("aiButton")
        btn_start.clicked.connect(self.accept)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_start)
        layout.addLayout(btn_layout)

    def on_save_toggled(self, checked):
        self.line_output_dir.setEnabled(checked)
        self.btn_browse.setEnabled(checked)

    def browse_folder(self):
        current = self.line_output_dir.text().strip() or os.getcwd()
        chosen = QFileDialog.getExistingDirectory(self, "Select Output Directory", current)
        if chosen:
            self.line_output_dir.setText(os.path.abspath(chosen))

    def get_configuration(self):
        if self.radio_fast_edes.isChecked():
            mode = "fast_edes"
        elif self.radio_full_4d.isChecked():
            mode = "full_4d"
        elif self.radio_middle.isChecked():
            mode = "middle"
        else:
            mode = "single_phase"

        target_phase = self.spin_phase.value() - 1
        save_mat = self.chk_save_mat.isChecked()
        device_choice = self.combo_device.currentData()
        output_dir = self.line_output_dir.text().strip() if save_mat else None
        return mode, target_phase, save_mat, device_choice, output_dir


class SegmentationProgressDialog(QDialog):
    """Segmentation progress modal with status details, progress bar, and cancellation support."""

    def __init__(self, parent=None, title="Segmenting & Quantifying..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowIcon(get_app_icon())
        self.resize(520, 220)
        self.setStyleSheet(DARK_STYLE_SHEET)
        self.setModal(True)

        self.worker = None
        self.start_time = time.time()

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.lbl_title = QLabel("<b>Executing Neural Network Model (ResNet34-UNet)...</b>")
        self.lbl_title.setStyleSheet("color: #4da8da; font-size: 13px;")
        layout.addWidget(self.lbl_title)

        self.lbl_detail = QLabel("Initializing inference...")
        self.lbl_detail.setStyleSheet("color: #eceff4; font-size: 12px;")
        self.lbl_detail.setWordWrap(True)
        layout.addWidget(self.lbl_detail)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.lbl_stats = QLabel("Elapsed Time: 0s  |  Progress: 0%")
        self.lbl_stats.setStyleSheet("color: #8f9ba8; font-size: 11px;")
        layout.addWidget(self.lbl_stats)

        layout.addStretch()

        btn_row = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel Process")
        self.btn_cancel.clicked.connect(self.cancel_worker)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)

    def set_worker(self, worker):
        self.worker = worker
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.status_message.connect(self.on_status_message)
        self.worker.segmentation_finished.connect(self.on_finished)
        self.worker.segmentation_error.connect(self.on_error)

    def on_progress_updated(self, current, total, msg):
        pct = int((current / max(1, total)) * 100)
        self.progress_bar.setValue(pct)
        self.lbl_detail.setText(msg)
        elapsed = int(time.time() - self.start_time)
        self.lbl_stats.setText(f"Step {current}/{total} ({pct}%)  |  Elapsed: {elapsed}s")

    def on_status_message(self, msg):
        self.lbl_detail.setText(msg)

    def update_timer(self):
        elapsed = int(time.time() - self.start_time)
        val = self.progress_bar.value()
        self.lbl_stats.setText(f"Progress: {val}%  |  Elapsed Time: {elapsed}s")

    def cancel_worker(self):
        if self.worker:
            self.worker.cancel()
            self.lbl_detail.setText("Cancelling process...")
            self.btn_cancel.setEnabled(False)

    def on_finished(self, results):
        self.timer.stop()
        self.accept()

    def on_error(self, err):
        self.timer.stop()
        self.reject()


class CardiacReportDialog(QDialog):
    """Clinical ventricular function report dialog presenting parameters in mL, g, and voxels."""

    def __init__(self, parent=None, metrics=None, filename="", mode=""):
        super().__init__(parent)
        self.metrics = metrics or {}
        self.filename = filename
        self.mode = mode
        self.setWindowTitle("Ventricular Clinical Report (mL) - VENTSEG")
        self.setWindowIcon(get_app_icon())
        self.resize(680, 580)
        self.setStyleSheet(DARK_STYLE_SHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        lbl_header = QLabel("<h3>Ventricular Quantification Summary (AI - mL)</h3>")
        lbl_header.setStyleSheet("color: #4da8da;")
        layout.addWidget(lbl_header)

        ed_p = self.metrics.get('ed_phase', 0)
        es_p = self.metrics.get('es_phase', 0)

        # Volumes in mL
        edv_ml = self.metrics.get('lv_edv_ml', self.metrics.get('lv_edv', 0.0))
        esv_ml = self.metrics.get('lv_esv_ml', self.metrics.get('lv_esv', 0.0))
        sv_ml = self.metrics.get('lv_sv_ml', self.metrics.get('lv_sv', 0.0))
        ef = self.metrics.get('lv_ef', 0.0)

        edv_vx = self.metrics.get('lv_edv_vx', 0.0)
        esv_vx = self.metrics.get('lv_esv_vx', 0.0)
        sv_vx = self.metrics.get('lv_sv_vx', 0.0)

        myo_ed_ml = self.metrics.get('myo_ed_ml', self.metrics.get('myo_ed', 0.0))
        myo_ed_vx = self.metrics.get('myo_ed_vx', 0.0)
        myo_mass_g = self.metrics.get('myo_mass_g', myo_ed_ml * 1.05)

        rv_edv_ml = self.metrics.get('rv_edv_ml', self.metrics.get('rv_edv', 0.0))
        rv_esv_ml = self.metrics.get('rv_esv_ml', self.metrics.get('rv_esv', 0.0))
        rv_sv_ml = self.metrics.get('rv_sv_ml', self.metrics.get('rv_sv', 0.0))
        rv_ef = self.metrics.get('rv_ef', 0.0)

        vx_size = self.metrics.get('voxel_size', np.array([1.0, 1.0, 1.0]))
        vx_vol_ml = self.metrics.get('voxel_volume_ml', 0.001)

        ef_color = "#51cf66" if ef >= 50.0 else ("#fcc419" if ef >= 40.0 else "#ff6b6b")
        ef_cat = "Normal (>= 50%)" if ef >= 50.0 else ("Mildly Reduced (40-49%)" if ef >= 40.0 else "Significantly Reduced (< 40%)")

        card_frame = QFrame()
        card_frame.setStyleSheet("background-color: #21242d; border: 1px solid #2d313e; border-radius: 8px; padding: 12px;")
        card_layout = QGridLayout(card_frame)
        card_layout.setSpacing(8)

        card_layout.addWidget(QLabel("<b>Dataset:</b>"), 0, 0)
        card_layout.addWidget(QLabel(f"{os.path.basename(self.filename)}"), 0, 1)

        card_layout.addWidget(QLabel("<b>Voxel Dimensions (dx, dy, dz):</b>"), 1, 0)
        card_layout.addWidget(QLabel(f"<b>{vx_size[0]:.2f} × {vx_size[1]:.2f} × {vx_size[2]:.2f} mm</b> ({vx_vol_ml:.4f} mL/voxel)"), 1, 1)

        card_layout.addWidget(QLabel("<b>End-Diastole (ED):</b>"), 2, 0)
        card_layout.addWidget(QLabel(f"<span style='color:#51cf66; font-weight:bold;'>Phase {ed_p + 1}</span>"), 2, 1)

        card_layout.addWidget(QLabel("<b>End-Systole (ES):</b>"), 3, 0)
        card_layout.addWidget(QLabel(f"<span style='color:#ff6b6b; font-weight:bold;'>Phase {es_p + 1}</span>"), 3, 1)

        card_layout.addWidget(QLabel("<b>LV End-Diastolic Volume (EDV):</b>"), 4, 0)
        card_layout.addWidget(QLabel(f"<span style='font-size:13px; font-weight:bold; color:#74c0fc;'>{edv_ml:.2f} mL</span> <span style='color:#8f9ba8;'>({edv_vx:,.0f} voxels)</span>"), 4, 1)

        card_layout.addWidget(QLabel("<b>LV End-Systolic Volume (ESV):</b>"), 5, 0)
        card_layout.addWidget(QLabel(f"<span style='font-size:13px; font-weight:bold; color:#74c0fc;'>{esv_ml:.2f} mL</span> <span style='color:#8f9ba8;'>({esv_vx:,.0f} voxels)</span>"), 5, 1)

        card_layout.addWidget(QLabel("<b>LV Stroke Volume (SV):</b>"), 6, 0)
        card_layout.addWidget(QLabel(f"<span style='color:#00adb5; font-size:13.5px; font-weight:bold;'>{sv_ml:.2f} mL</span> <span style='color:#8f9ba8;'>({sv_vx:,.0f} voxels)</span>"), 6, 1)

        card_layout.addWidget(QLabel("<b>LV Ejection Fraction (EF):</b>"), 7, 0)
        card_layout.addWidget(QLabel(f"<span style='color:{ef_color}; font-size:16px; font-weight:bold;'>{ef:.1f}%</span> ({ef_cat})"), 7, 1)

        card_layout.addWidget(QLabel("<b>LV Myocardium at ED:</b>"), 8, 0)
        card_layout.addWidget(QLabel(f"<b>{myo_ed_ml:.2f} mL</b>  |  Estimated Mass: <span style='color:#51cf66; font-weight:bold;'>{myo_mass_g:.1f} g</span> <span style='color:#8f9ba8;'>({myo_ed_vx:,.0f} vx)</span>"), 8, 1)

        card_layout.addWidget(QLabel("<b>Right Ventricle (RV):</b>"), 9, 0)
        card_layout.addWidget(QLabel(f"EF: <b>{rv_ef:.1f}%</b>  |  EDV: <b>{rv_edv_ml:.2f} mL</b>  |  ESV: <b>{rv_esv_ml:.2f} mL</b>  |  SV: <b>{rv_sv_ml:.2f} mL</b>"), 9, 1)

        layout.addWidget(card_frame)

        # Action Buttons
        btn_row = QHBoxLayout()
        btn_ed = QPushButton("Jump to ED")
        btn_ed.setObjectName("edButton")
        btn_ed.clicked.connect(self.go_to_ed)

        btn_es = QPushButton("Jump to ES")
        btn_es.setObjectName("esButton")
        btn_es.clicked.connect(self.go_to_es)

        btn_mosaic = QPushButton("ED vs ES Mosaic")
        btn_mosaic.clicked.connect(self.open_mosaic_edes)

        btn_save = QPushButton("Export Report (.txt)")
        btn_save.clicked.connect(self.export_report_txt)

        btn_save_all = QPushButton("Save All to Folder...")
        btn_save_all.setToolTip("Export complete results (.mat, clinical report, CSV curves, snapshots, GIF) to a folder")
        btn_save_all.clicked.connect(self.save_all_results)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)

        btn_row.addWidget(btn_ed)
        btn_row.addWidget(btn_es)
        btn_row.addWidget(btn_mosaic)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_save_all)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def save_all_results(self):
        if self.parent() and hasattr(self.parent(), 'export_all_results_to_folder'):
            self.parent().export_all_results_to_folder()

    def go_to_ed(self):
        if self.parent() and hasattr(self.parent(), 'set_phase'):
            self.parent().set_phase(self.metrics.get('ed_phase', 0))
        self.accept()

    def go_to_es(self):
        if self.parent() and hasattr(self.parent(), 'set_phase'):
            self.parent().set_phase(self.metrics.get('es_phase', 0))
        self.accept()

    def open_mosaic_edes(self):
        if self.parent() and hasattr(self.parent(), 'tabs'):
            self.parent().tabs.setCurrentIndex(1)
            if hasattr(self.parent(), 'grid_view_widget'):
                self.parent().grid_view_widget.combo_mode.setCurrentIndex(2)
        self.accept()

    def export_report_txt(self):
        start_dir = getattr(self.parent(), 'last_directory', os.getcwd()) if self.parent() else os.getcwd()
        default_path = os.path.join(start_dir, "ventricular_clinical_report.txt")
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Clinical Report", default_path, "Text Files (*.txt);;CSV Files (*.csv)"
        )
        if filepath:
            try:
                ed_p = self.metrics.get('ed_phase', 0)
                es_p = self.metrics.get('es_phase', 0)
                edv_ml = self.metrics.get('lv_edv_ml', 0.0)
                esv_ml = self.metrics.get('lv_esv_ml', 0.0)
                sv_ml = self.metrics.get('lv_sv_ml', 0.0)
                ef = self.metrics.get('lv_ef', 0.0)
                edv_vx = self.metrics.get('lv_edv_vx', 0.0)
                esv_vx = self.metrics.get('lv_esv_vx', 0.0)
                sv_vx = self.metrics.get('lv_sv_vx', 0.0)

                myo_ed_ml = self.metrics.get('myo_ed_ml', 0.0)
                myo_mass_g = self.metrics.get('myo_mass_g', 0.0)

                rv_edv_ml = self.metrics.get('rv_edv_ml', 0.0)
                rv_esv_ml = self.metrics.get('rv_esv_ml', 0.0)
                rv_sv_ml = self.metrics.get('rv_sv_ml', 0.0)
                rv_ef = self.metrics.get('rv_ef', 0.0)

                vx_size = self.metrics.get('voxel_size', np.array([1.0, 1.0, 1.0]))
                vx_vol_ml = self.metrics.get('voxel_volume_ml', 0.001)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("=" * 75 + "\n")
                    f.write("VENTSEG - VENTRICULAR QUANTIFICATION AND CLINICAL FUNCTION REPORT\n")
                    f.write("=" * 75 + "\n\n")
                    f.write(f"Dataset: {self.filename}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Physical Voxel Spacing: dx={vx_size[0]:.4f} mm, dy={vx_size[1]:.4f} mm, dz={vx_size[2]:.4f} mm\n")
                    f.write(f"Unit Voxel Volume: {vx_vol_ml * 1000:.4f} mm³ ({vx_vol_ml:.6f} mL)\n\n")
                    f.write(f"End-Diastole (ED): Phase {ed_p + 1}\n")
                    f.write(f"End-Systole (ES): Phase {es_p + 1}\n\n")
                    f.write("--- LEFT VENTRICLE (LV) ---\n")
                    f.write(f"End-Diastolic Volume (EDV): {edv_ml:.2f} mL ({edv_vx:,.0f} voxels)\n")
                    f.write(f"End-Systolic Volume (ESV): {esv_ml:.2f} mL ({esv_vx:,.0f} voxels)\n")
                    f.write(f"Stroke Volume (SV): {sv_ml:.2f} mL ({sv_vx:,.0f} voxels)\n")
                    f.write(f"Ejection Fraction (EF): {ef:.2f} %\n\n")
                    f.write("--- MYOCARDIUM ---\n")
                    f.write(f"Myocardial Volume at ED: {myo_ed_ml:.2f} mL\n")
                    f.write(f"Estimated Myocardial Mass (density 1.05 g/mL): {myo_mass_g:.2f} g\n\n")
                    f.write("--- RIGHT VENTRICLE (RV) ---\n")
                    f.write(f"End-Diastolic Volume (RV EDV): {rv_edv_ml:.2f} mL\n")
                    f.write(f"End-Systolic Volume (RV ESV): {rv_esv_ml:.2f} mL\n")
                    f.write(f"Stroke Volume (RV SV): {rv_sv_ml:.2f} mL\n")
                    f.write(f"Ejection Fraction (RV EF): {rv_ef:.2f} %\n")
                    f.write("=" * 75 + "\n")
                QMessageBox.information(self, "Report Saved", f"Clinical report exported successfully to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save clinical report:\n{str(e)}")


# ==============================================================================
# MAIN APPLICATION WINDOW (VentSegViewer4D)
# ==============================================================================
class VentSegViewer4D(QMainWindow):
    """Main Application Window for VENTSEG 4D Cardiac Explorer & Quantification."""

    def __init__(self, initial_filepath=None):
        super().__init__()
        self.setWindowTitle("VENTSEG - 4D Cardiac MRI Medical Image Viewer & Quantification Suite")
        self.setWindowIcon(get_app_icon())
        self.resize(1420, 930)
        self.setMinimumSize(1024, 700)
        self.setStyleSheet(DARK_STYLE_SHEET)

        # Internal state
        self.med_image = None
        self.current_slice = 0
        self.current_phase = 0
        self.current_row = None
        self.current_col = None
        self.unit_mode = "ml"  # "ml" or "voxels"

        # Cine Player Timer
        self.cine_timer = QTimer(self)
        self.cine_timer.timeout.connect(self.advance_cine_phase)
        self.is_playing = False
        self.fps = 15

        # Display Parameters
        self.vmin = 0.0
        self.vmax = 1.0
        self.cmap_name = 'bone'
        self.window_width = 1.0
        self.window_level = 0.5
        self.auto_contrast = True
        self.show_mask = True
        self.mask_opacity = 0.45

        self.last_directory = os.path.dirname(os.path.abspath(initial_filepath)) if (initial_filepath and os.path.exists(initial_filepath)) else os.getcwd()

        # Build UI & Shortcuts
        self.init_ui()
        self.setup_shortcuts()

        # Initialize UI displays
        self.update_cardiac_metrics_ui()
        self.update_metadata_display()

        # Load initial file if specified via CLI
        if initial_filepath and os.path.exists(initial_filepath):
            self.load_file(initial_filepath)

    def init_ui(self):
        self.create_menus()
        self.create_toolbar()

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(self.main_splitter)

        # --- CENTRAL/RIGHT PANEL (Viewer & Visualizations) ---
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()

        # Tab 1: Single View & Volumetric Curves
        self.viewer_canvas = InteractiveViewerCanvas(self)
        self.viewer_canvas.unit_mode = self.unit_mode
        self.viewer_canvas.pixel_clicked.connect(self.on_pixel_clicked)
        self.viewer_canvas.phase_clicked.connect(self.set_phase)
        self.viewer_canvas.slice_scroll_requested.connect(self.on_slice_scroll)
        self.viewer_canvas.phase_scroll_requested.connect(self.on_phase_scroll)
        self.viewer_canvas.status_callback = self.update_hover_status

        self.mpl_toolbar = NavigationToolbar2QT(self.viewer_canvas, self)
        self.mpl_toolbar.setStyleSheet("background-color: #21242d; border: none; color: #eceff4;")

        single_view_widget = QWidget()
        single_view_layout = QVBoxLayout(single_view_widget)
        single_view_layout.setContentsMargins(0, 0, 0, 0)
        single_view_layout.addWidget(self.mpl_toolbar)
        single_view_layout.addWidget(self.viewer_canvas)

        # Tab 2: Cardiac Mosaic Grid
        self.grid_view_widget = MultiSliceGridWidget(self)
        self.grid_view_widget.slice_selected.connect(self.on_grid_slice_selected)
        self.grid_view_widget.phase_selected.connect(self.on_grid_phase_selected)

        self.tabs.addTab(single_view_widget, "Single View & Volumetric Curves (LV, RV, MYO in mL)")
        self.tabs.addTab(self.grid_view_widget, "Cardiac Mosaic (ED/ES Phases & Slices)")
        self.tabs.currentChanged.connect(self.on_tab_changed)

        center_layout.addWidget(self.tabs)

        # --- LEFT SIDEBAR PANEL (Controls & Metrics) ---
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setMinimumWidth(300)
        self.sidebar_scroll.setMaximumWidth(460)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(10)

        sidebar_layout.addWidget(self.create_app_header())
        sidebar_layout.addWidget(self.create_cardiac_metrics_group())
        sidebar_layout.addWidget(self.create_segmentation_group())
        sidebar_layout.addWidget(self.create_slice_control_group())
        sidebar_layout.addWidget(self.create_phase_control_group())
        sidebar_layout.addWidget(self.create_display_control_group())
        sidebar_layout.addWidget(self.create_mask_control_group())
        sidebar_layout.addWidget(self.create_info_group())

        sidebar_layout.addStretch()
        self.sidebar_scroll.setWidget(sidebar_content)

        self.main_splitter.addWidget(self.sidebar_scroll)
        self.main_splitter.addWidget(center_widget)
        self.main_splitter.setCollapsible(0, True)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Open a 4D medical imaging dataset (.mat, .nii, .npy) to start.")

    # ==========================================================================
    # MENUS AND TOOLBARS
    # ==========================================================================
    def create_menus(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("background-color: #21242d; color: #eceff4; font-weight: 500;")

        # File Menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("Open 4D Image (.mat, .nii, .npy)...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        load_mask_action = QAction("Load Mask / Segmentation (.mat, .nii)...", self)
        load_mask_action.setShortcut(QKeySequence("Ctrl+M"))
        load_mask_action.triggered.connect(self.open_mask_dialog)
        file_menu.addAction(load_mask_action)

        file_menu.addSeparator()

        save_all_action = QAction("Save All Results to Directory...", self)
        save_all_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_all_action.triggered.connect(self.export_all_results_to_folder)
        file_menu.addAction(save_all_action)

        file_menu.addSeparator()

        voxel_size_action = QAction("Spatial Calibration / Voxel Size (mm)...", self)
        voxel_size_action.setShortcut(QKeySequence("Ctrl+K"))
        voxel_size_action.triggered.connect(self.open_voxel_size_dialog)
        file_menu.addAction(voxel_size_action)

        file_menu.addSeparator()

        export_img_action = QAction("Export Current View Snapshot (PNG)...", self)
        export_img_action.setShortcut(QKeySequence("Ctrl+S"))
        export_img_action.triggered.connect(self.export_current_snapshot)
        file_menu.addAction(export_img_action)

        export_gif_action = QAction("Export Cardiac Cine as GIF...", self)
        export_gif_action.setShortcut(QKeySequence("Ctrl+G"))
        export_gif_action.triggered.connect(self.export_cine_gif)
        file_menu.addAction(export_gif_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # AI Quantification Menu
        seg_menu = menubar.addMenu("&AI Quantification")

        run_seg_action = QAction("Quantify & Segment with Neural Network...", self)
        run_seg_action.setShortcut(QKeySequence("Ctrl+R"))
        run_seg_action.triggered.connect(self.show_segmentation_dialog)
        seg_menu.addAction(run_seg_action)

        fast_seg_action = QAction("Fast ED / ES Detection + Quantification in mL", self)
        fast_seg_action.triggered.connect(lambda: self.run_direct_segmentation(mode="fast_edes"))
        seg_menu.addAction(fast_seg_action)

        full_seg_action = QAction("Full 4D Spatio-Temporal Segmentation (All Phases)", self)
        full_seg_action.triggered.connect(lambda: self.run_direct_segmentation(mode="full_4d"))
        seg_menu.addAction(full_seg_action)

        mid_seg_action = QAction("Segment Central Slice Only", self)
        mid_seg_action.triggered.connect(lambda: self.run_direct_segmentation(mode="middle"))
        seg_menu.addAction(mid_seg_action)

        seg_menu.addSeparator()
        seg_menu.addAction(save_all_action)
        seg_menu.addAction(voxel_size_action)

        view_report_action = QAction("View Clinical Report & Ventricular Metrics (mL)...", self)
        view_report_action.triggered.connect(self.show_clinical_report_dialog)
        seg_menu.addAction(view_report_action)

        # View Menu
        view_menu = menubar.addMenu("&View")

        toggle_sidebar_action = QAction("Toggle Sidebar Panel (Hide/Show)", self)
        toggle_sidebar_action.setShortcut(QKeySequence("Ctrl+B"))
        toggle_sidebar_action.triggered.connect(self.toggle_sidebar)
        view_menu.addAction(toggle_sidebar_action)

        toggle_units_action = QAction("Toggle Volume Units (mL <-> Voxels)", self)
        toggle_units_action.setShortcut(QKeySequence("U"))
        toggle_units_action.triggered.connect(self.toggle_units_mode)
        view_menu.addAction(toggle_units_action)

        view_menu.addSeparator()

        reset_contrast_action = QAction("Reset Brightness & Contrast", self)
        reset_contrast_action.setShortcut(QKeySequence("R"))
        reset_contrast_action.triggered.connect(self.reset_contrast)
        view_menu.addAction(reset_contrast_action)

        jump_ed_action = QAction("Jump to End-Diastole (ED)", self)
        jump_ed_action.setShortcut(QKeySequence("E"))
        jump_ed_action.triggered.connect(self.jump_to_ed_phase)
        view_menu.addAction(jump_ed_action)

        jump_es_action = QAction("Jump to End-Systole (ES)", self)
        jump_es_action.setShortcut(QKeySequence("S"))
        jump_es_action.triggered.connect(self.jump_to_es_phase)
        view_menu.addAction(jump_es_action)

        # Help Menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("About VENTSEG 4D", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        shortcuts_action = QAction("Keyboard Shortcuts Guide", self)
        shortcuts_action.triggered.connect(self.show_shortcuts_dialog)
        help_menu.addAction(shortcuts_action)

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.btn_toggle_sidebar = QPushButton("Hide Panel")
        self.btn_toggle_sidebar.setToolTip("Hide or show the sidebar control panel to maximize viewing area (Ctrl+B)")
        self.btn_toggle_sidebar.clicked.connect(self.toggle_sidebar)
        toolbar.addWidget(self.btn_toggle_sidebar)

        toolbar.addSeparator()

        btn_open = QPushButton("Open Image")
        btn_open.setObjectName("accentButton")
        btn_open.setToolTip("Open a 4D medical dataset (.mat, .nii, .npy) from any folder (Ctrl+O)")
        btn_open.clicked.connect(self.open_file_dialog)
        toolbar.addWidget(btn_open)

        btn_mask = QPushButton("Load Segmentation")
        btn_mask.setToolTip("Load a companion segmentation mask (.mat, .nii) (Ctrl+M)")
        btn_mask.clicked.connect(self.open_mask_dialog)
        toolbar.addWidget(btn_mask)

        toolbar.addSeparator()

        btn_save_all = QPushButton("Save Results")
        btn_save_all.setToolTip("Export complete results (.mat masks, clinical report, CSV curves, snapshots, and GIF) to any folder (Ctrl+Shift+S)")
        btn_save_all.clicked.connect(self.export_all_results_to_folder)
        toolbar.addWidget(btn_save_all)

        toolbar.addSeparator()

        self.btn_tb_voxel = QPushButton("Voxel Size")
        self.btn_tb_voxel.setToolTip("Inspect and calibrate physical voxel dimensions [dx, dy, dz] in mm (Ctrl+K)")
        self.btn_tb_voxel.clicked.connect(self.open_voxel_size_dialog)
        toolbar.addWidget(self.btn_tb_voxel)

        toolbar.addSeparator()

        self.btn_tb_segment = QPushButton("Quantify / Segment")
        self.btn_tb_segment.setObjectName("aiButton")
        self.btn_tb_segment.setToolTip("Run neural segmentation and automated ventricular quantification (Ctrl+R)")
        self.btn_tb_segment.clicked.connect(self.show_segmentation_dialog)
        toolbar.addWidget(self.btn_tb_segment)

        toolbar.addSeparator()

        self.tb_btn_ed = QPushButton("ED (Diastole)")
        self.tb_btn_ed.setObjectName("edButton")
        self.tb_btn_ed.clicked.connect(self.jump_to_ed_phase)
        toolbar.addWidget(self.tb_btn_ed)

        self.tb_btn_es = QPushButton("ES (Systole)")
        self.tb_btn_es.setObjectName("esButton")
        self.tb_btn_es.clicked.connect(self.jump_to_es_phase)
        toolbar.addWidget(self.tb_btn_es)

        toolbar.addSeparator()

        btn_export_snap = QPushButton("Snapshot")
        btn_export_snap.clicked.connect(self.export_current_snapshot)
        toolbar.addWidget(btn_export_snap)

        btn_export_gif = QPushButton("Export GIF")
        btn_export_gif.clicked.connect(self.export_cine_gif)
        toolbar.addWidget(btn_export_gif)

    # ==========================================================================
    # SIDEBAR CONTROL GROUPS
    # ==========================================================================
    def create_app_header(self):
        """Build branded software header with logo and suite identity."""
        header_frame = QFrame()
        header_frame.setObjectName("appHeaderFrame")
        header_frame.setStyleSheet("""
            QFrame#appHeaderFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a233a, stop:1 #141721);
                border: 1px solid #2d3856;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(8, 6, 8, 6)
        h_layout.setSpacing(12)

        # Logo Pixmap
        base_dir = os.path.dirname(os.path.abspath(__file__))
        png_path = os.path.join(base_dir, "images", "Logo.png")
        if os.path.exists(png_path):
            logo_lbl = QLabel()
            pixmap = QPixmap(png_path)
            if not pixmap.isNull():
                scaled_pix = pixmap.scaled(44, 44, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                logo_lbl.setPixmap(scaled_pix)
                logo_lbl.setStyleSheet("background: transparent;")
                h_layout.addWidget(logo_lbl)

        # Title and description
        text_vbox = QVBoxLayout()
        text_vbox.setSpacing(2)

        title_lbl = QLabel("VENTSEG 4D")
        title_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #4da8da; background: transparent; letter-spacing: 0.5px;")

        sub_lbl = QLabel("Cardiac AI Quantification Suite")
        sub_lbl.setStyleSheet("font-size: 10.5px; color: #94a3b8; font-weight: 500; background: transparent;")

        text_vbox.addWidget(title_lbl)
        text_vbox.addWidget(sub_lbl)
        h_layout.addLayout(text_vbox)
        h_layout.addStretch()

        return header_frame

    def create_cardiac_metrics_group(self):
        group = QGroupBox("Cardiac Metrics & Key Phases (ED / ES)")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        # Volume Unit Selector
        unit_box = QHBoxLayout()
        unit_box.addWidget(QLabel("<b>Units:</b>"))
        self.radio_unit_ml = QRadioButton("mL (cm³)")
        self.radio_unit_ml.setChecked(True)
        self.radio_unit_ml.toggled.connect(self.on_units_toggled)

        self.radio_unit_vx = QRadioButton("Voxels")
        self.radio_unit_vx.toggled.connect(self.on_units_toggled)

        unit_box.addWidget(self.radio_unit_ml)
        unit_box.addWidget(self.radio_unit_vx)
        layout.addLayout(unit_box)

        # Jump buttons
        btn_row = QHBoxLayout()
        self.btn_side_ed = QPushButton("End-Diastole (ED)")
        self.btn_side_ed.setObjectName("edButton")
        self.btn_side_ed.clicked.connect(self.jump_to_ed_phase)

        self.btn_side_es = QPushButton("End-Systole (ES)")
        self.btn_side_es.setObjectName("esButton")
        self.btn_side_es.clicked.connect(self.jump_to_es_phase)

        btn_row.addWidget(self.btn_side_ed)
        btn_row.addWidget(self.btn_side_es)
        layout.addLayout(btn_row)

        # Metrics Grid Cards
        grid_metrics = QGridLayout()
        grid_metrics.setHorizontalSpacing(8)
        grid_metrics.setVerticalSpacing(4)

        self.lbl_metric_ed_phase = QLabel("<b>ED Phase:</b> -")
        self.lbl_metric_es_phase = QLabel("<b>ES Phase:</b> -")
        self.lbl_metric_edv = QLabel("<b>EDV (LV):</b> -")
        self.lbl_metric_esv = QLabel("<b>ESV (LV):</b> -")
        self.lbl_metric_sv = QLabel("<b>Stroke Volume (SV):</b> -")
        self.lbl_metric_ef = QLabel("<b>Ejection Frac. (EF):</b> -")
        self.lbl_metric_myo = QLabel("<b>Myocardium (ED):</b> -")
        self.lbl_metric_rv = QLabel("<b>Right Ventricle (RV):</b> -")

        grid_metrics.addWidget(self.lbl_metric_ed_phase, 0, 0)
        grid_metrics.addWidget(self.lbl_metric_es_phase, 0, 1)
        grid_metrics.addWidget(self.lbl_metric_edv, 1, 0)
        grid_metrics.addWidget(self.lbl_metric_esv, 1, 1)
        grid_metrics.addWidget(self.lbl_metric_sv, 2, 0)
        grid_metrics.addWidget(self.lbl_metric_ef, 2, 1)
        grid_metrics.addWidget(self.lbl_metric_myo, 3, 0, 1, 2)
        grid_metrics.addWidget(self.lbl_metric_rv, 4, 0, 1, 2)

        layout.addLayout(grid_metrics)

        # Curve Mode Selector (3D Volume vs 2D Area)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("<b>Curve Mode:</b>"))
        self.radio_vol3d = QRadioButton("3D Volume")
        self.radio_vol3d.setChecked(True)
        self.radio_vol3d.toggled.connect(self.on_curve_mode_toggled)

        self.radio_area2d = QRadioButton("2D Area")
        self.radio_area2d.toggled.connect(self.on_curve_mode_toggled)

        mode_row.addWidget(self.radio_vol3d)
        mode_row.addWidget(self.radio_area2d)
        layout.addLayout(mode_row)

        return group

    def create_segmentation_group(self):
        group = QGroupBox("AI Quantification (ResNet34-UNet)")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.btn_run_ai = QPushButton("Start AI Quantification...")
        self.btn_run_ai.setObjectName("aiButton")
        self.btn_run_ai.setToolTip("Run neural network segmentation and clinical ventricular quantification (Ctrl+R)")
        self.btn_run_ai.clicked.connect(self.show_segmentation_dialog)
        layout.addWidget(self.btn_run_ai)

        btn_grid = QGridLayout()
        btn_grid.setSpacing(4)

        btn_fast = QPushButton("Fast ED/ES")
        btn_fast.setToolTip("Segments central slice, detects ED/ES phases, and segments 3D volumes at ED and ES (mL)")
        btn_fast.clicked.connect(lambda: self.run_direct_segmentation(mode="fast_edes"))

        btn_full = QPushButton("Full 4D")
        btn_full.setToolTip("Segments all phases and slices (-> segmentation_all_phases.mat, segmentation_ED.mat, segmentation_ES.mat)")
        btn_full.clicked.connect(lambda: self.run_direct_segmentation(mode="full_4d"))

        btn_mid = QPushButton("Central Slice")
        btn_mid.setToolTip("Segments mid-cavity slice across full cardiac cycle (-> segmentation_all_phases.mat)")
        btn_mid.clicked.connect(lambda: self.run_direct_segmentation(mode="middle"))

        btn_phase = QPushButton("Curr. Phase")
        btn_phase.setToolTip("Segments 3D volume for current cardiac phase only (-> segmentation_phase.mat)")
        btn_phase.clicked.connect(lambda: self.run_direct_segmentation(mode="single_phase"))

        btn_grid.addWidget(btn_fast, 0, 0)
        btn_grid.addWidget(btn_full, 0, 1)
        btn_grid.addWidget(btn_mid, 1, 0)
        btn_grid.addWidget(btn_phase, 1, 1)
        layout.addLayout(btn_grid)

        self.btn_view_report = QPushButton("View Clinical Report (mL)")
        self.btn_view_report.setEnabled(False)
        self.btn_view_report.clicked.connect(self.show_clinical_report_dialog)
        layout.addWidget(self.btn_view_report)

        return group

    def create_slice_control_group(self):
        group = QGroupBox("Slice Navigation (Z-Stack / Slices)")
        layout = QVBoxLayout(group)

        top_row = QHBoxLayout()
        lbl_title = QLabel("Current Slice:")
        self.lbl_slice_badge = QLabel("0 / 0")
        self.lbl_slice_badge.setObjectName("badgeLabel")
        top_row.addWidget(lbl_title)
        top_row.addStretch()
        top_row.addWidget(self.lbl_slice_badge)
        layout.addLayout(top_row)

        self.slider_slice = QSlider(Qt.Orientation.Horizontal)
        self.slider_slice.setRange(0, 0)
        self.slider_slice.valueChanged.connect(self.on_slice_slider_changed)
        layout.addWidget(self.slider_slice)

        btn_layout = QHBoxLayout()
        self.btn_slice_first = QPushButton("1")
        self.btn_slice_first.clicked.connect(lambda: self.set_slice(0))

        self.btn_slice_prev = QPushButton("Prev")
        self.btn_slice_prev.clicked.connect(lambda: self.set_slice(self.current_slice - 1))

        self.btn_slice_mid = QPushButton("Center")
        self.btn_slice_mid.clicked.connect(self.go_to_middle_slice)

        self.btn_slice_next = QPushButton("Next")
        self.btn_slice_next.clicked.connect(lambda: self.set_slice(self.current_slice + 1))

        self.btn_slice_last = QPushButton("End")
        self.btn_slice_last.clicked.connect(lambda: self.set_slice(self.med_image.num_slices - 1 if self.med_image else 0))

        btn_layout.addWidget(self.btn_slice_first)
        btn_layout.addWidget(self.btn_slice_prev)
        btn_layout.addWidget(self.btn_slice_mid)
        btn_layout.addWidget(self.btn_slice_next)
        btn_layout.addWidget(self.btn_slice_last)
        layout.addLayout(btn_layout)

        return group

    def create_phase_control_group(self):
        group = QGroupBox("Cine Player & Cardiac Phases (Time)")
        layout = QVBoxLayout(group)

        top_row = QHBoxLayout()
        lbl_title = QLabel("Cardiac Phase:")
        self.lbl_phase_badge = QLabel("0 / 0")
        self.lbl_phase_badge.setObjectName("badgeLabel")
        top_row.addWidget(lbl_title)
        top_row.addStretch()
        top_row.addWidget(self.lbl_phase_badge)
        layout.addLayout(top_row)

        self.slider_phase = QSlider(Qt.Orientation.Horizontal)
        self.slider_phase.setRange(0, 0)
        self.slider_phase.valueChanged.connect(self.on_phase_slider_changed)
        layout.addWidget(self.slider_phase)

        btn_layout = QHBoxLayout()
        self.btn_phase_prev = QPushButton("Prev Phase")
        self.btn_phase_prev.clicked.connect(lambda: self.set_phase(self.current_phase - 1))

        self.btn_play_pause = QPushButton("Play Cine")
        self.btn_play_pause.setObjectName("playButton")
        self.btn_play_pause.clicked.connect(self.toggle_cine_playback)

        self.btn_phase_next = QPushButton("Next Phase")
        self.btn_phase_next.clicked.connect(lambda: self.set_phase(self.current_phase + 1))

        btn_layout.addWidget(self.btn_phase_prev)
        btn_layout.addWidget(self.btn_play_pause)
        btn_layout.addWidget(self.btn_phase_next)
        layout.addLayout(btn_layout)

        fps_layout = QHBoxLayout()
        fps_label = QLabel("Playback Speed:")
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(15)
        self.spin_fps.setSuffix(" FPS")
        self.spin_fps.valueChanged.connect(self.on_fps_changed)
        fps_layout.addWidget(fps_label)
        fps_layout.addStretch()
        fps_layout.addWidget(self.spin_fps)
        layout.addLayout(fps_layout)

        return group

    def create_display_control_group(self):
        group = QGroupBox("Display, Brightness & Contrast")
        layout = QVBoxLayout(group)

        cmap_row = QHBoxLayout()
        cmap_lbl = QLabel("Colormap:")
        self.combo_cmap = QComboBox()
        colormaps = [
            ("Medical / Bone (bone)", "bone"),
            ("Grayscale (gray)", "gray"),
            ("Viridis (Perceptual)", "viridis"),
            ("Plasma", "plasma"),
            ("Inferno", "inferno"),
            ("Magma", "magma"),
            ("Jet (Rainbow)", "jet"),
            ("Turbo", "turbo"),
            ("Hot (Thermal)", "hot"),
            ("Coolwarm (Bipolar)", "coolwarm")
        ]
        for name, key in colormaps:
            self.combo_cmap.addItem(name, key)
        self.combo_cmap.setCurrentIndex(0)
        self.combo_cmap.currentIndexChanged.connect(self.on_cmap_changed)
        cmap_row.addWidget(cmap_lbl)
        cmap_row.addWidget(self.combo_cmap)
        layout.addLayout(cmap_row)

        preset_row = QHBoxLayout()
        btn_auto = QPushButton("Auto (1%-99%)")
        btn_auto.clicked.connect(self.set_auto_contrast)
        btn_minmax = QPushButton("Full Dynamic Range")
        btn_minmax.clicked.connect(self.set_minmax_contrast)
        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self.reset_contrast)
        preset_row.addWidget(btn_auto)
        preset_row.addWidget(btn_minmax)
        preset_row.addWidget(btn_reset)
        layout.addLayout(preset_row)

        grid_wl = QGridLayout()
        grid_wl.addWidget(QLabel("Window:"), 0, 0)
        self.slider_window = QSlider(Qt.Orientation.Horizontal)
        self.slider_window.setRange(1, 1000)
        self.slider_window.setValue(1000)
        self.slider_window.valueChanged.connect(self.on_wl_slider_changed)
        grid_wl.addWidget(self.slider_window, 0, 1)

        grid_wl.addWidget(QLabel("Level:"), 1, 0)
        self.slider_level = QSlider(Qt.Orientation.Horizontal)
        self.slider_level.setRange(0, 1000)
        self.slider_level.setValue(500)
        self.slider_level.valueChanged.connect(self.on_wl_slider_changed)
        grid_wl.addWidget(self.slider_level, 1, 1)

        layout.addLayout(grid_wl)

        return group

    def create_mask_control_group(self):
        group = QGroupBox("Cardiac Segmentation (Overlay)")
        layout = QVBoxLayout(group)

        self.chk_show_mask = QCheckBox("Show Overlay in 2D View")
        self.chk_show_mask.setChecked(True)
        self.chk_show_mask.toggled.connect(self.on_toggle_mask)
        layout.addWidget(self.chk_show_mask)

        opac_row = QHBoxLayout()
        opac_lbl = QLabel("Opacity (Alpha):")
        self.slider_mask_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_mask_opacity.setRange(5, 100)
        self.slider_mask_opacity.setValue(45)
        self.slider_mask_opacity.valueChanged.connect(self.on_mask_opacity_changed)
        opac_row.addWidget(opac_lbl)
        opac_row.addWidget(self.slider_mask_opacity)
        layout.addLayout(opac_row)

        legend_frame = QFrame()
        legend_frame.setStyleSheet("background-color: #1a1b22; border-radius: 4px; padding: 4px;")
        leg_layout = QHBoxLayout(legend_frame)
        leg_layout.setContentsMargins(4, 4, 4, 4)

        lbl_lv = QLabel("LV Endo (1)")
        lbl_lv.setStyleSheet("color: #ff6b6b; font-size: 11px; font-weight: bold;")
        lbl_myo = QLabel("Myo (2)")
        lbl_myo.setStyleSheet("color: #51cf66; font-size: 11px; font-weight: bold;")
        lbl_rv = QLabel("RV (3)")
        lbl_rv.setStyleSheet("color: #339af0; font-size: 11px; font-weight: bold;")

        leg_layout.addWidget(lbl_lv)
        leg_layout.addWidget(lbl_myo)
        leg_layout.addWidget(lbl_rv)
        layout.addWidget(legend_frame)

        return group

    def create_info_group(self):
        group = QGroupBox("Metadata, Calibration & Voxel Size")
        layout = QVBoxLayout(group)

        self.lbl_info_file = QLabel("Dataset: None")
        self.lbl_info_file.setWordWrap(True)
        self.lbl_info_mask = QLabel("Segmentation: None")
        self.lbl_info_mask.setWordWrap(True)
        self.lbl_info_dims = QLabel("Dimensions: -")
        self.lbl_info_voxel_size = QLabel("Voxel Dimensions: -")
        self.lbl_info_voxel_vol = QLabel("Unit Voxel Volume: -")
        self.lbl_info_pixel_area = QLabel("2D Pixel Area: -")
        self.lbl_info_dtype = QLabel("Data Type: -")
        self.lbl_info_range = QLabel("Intensity Dynamic Range: -")

        layout.addWidget(self.lbl_info_file)
        layout.addWidget(self.lbl_info_mask)
        layout.addWidget(self.lbl_info_dims)
        layout.addWidget(self.lbl_info_voxel_size)
        layout.addWidget(self.lbl_info_voxel_vol)
        layout.addWidget(self.lbl_info_pixel_area)
        layout.addWidget(self.lbl_info_dtype)
        layout.addWidget(self.lbl_info_range)

        btn_edit_voxel = QPushButton("Configure Voxel Size (mm)...")
        btn_edit_voxel.setToolTip("Calibrate physical voxel spacing in mm for accurate volume computation in mL (Ctrl+K)")
        btn_edit_voxel.clicked.connect(self.open_voxel_size_dialog)
        layout.addWidget(btn_edit_voxel)

        return group

    def setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.toggle_cine_playback)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.set_phase(self.current_phase - 1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.set_phase(self.current_phase + 1))
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, lambda: self.set_slice(self.current_slice + 1))
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, lambda: self.set_slice(self.current_slice - 1))
        QShortcut(QKeySequence("R"), self, self.reset_contrast)
        QShortcut(QKeySequence("M"), self, lambda: self.chk_show_mask.toggle())
        QShortcut(QKeySequence("E"), self, self.jump_to_ed_phase)
        QShortcut(QKeySequence("S"), self, self.jump_to_es_phase)
        QShortcut(QKeySequence("U"), self, self.toggle_units_mode)
        QShortcut(QKeySequence("Ctrl+B"), self, self.toggle_sidebar)
        QShortcut(QKeySequence("Ctrl+R"), self, self.show_segmentation_dialog)
        QShortcut(QKeySequence("Ctrl+K"), self, self.open_voxel_size_dialog)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, self.export_all_results_to_folder)

    # ==========================================================================
    # VOXEL SIZE CALIBRATION
    # ==========================================================================
    def open_voxel_size_dialog(self):
        """Open modal dialog to inspect or calibrate physical voxel dimensions [dx, dy, dz] in mm."""
        if self.med_image is None:
            QMessageBox.warning(self, "No Image Loaded", "Please open a 4D medical imaging dataset before configuring voxel dimensions.")
            return

        dlg = VoxelSizeDialog(self, med_image=self.med_image)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_vs = dlg.get_voxel_size()
            self.med_image.set_voxel_size(new_vs, mark_as_metadata=True)

            if dlg.should_save_to_mat() and self.med_image.filename:
                base_dir = os.path.dirname(os.path.abspath(self.med_image.filename))
                mat_files_to_update = [
                    os.path.basename(self.med_image.filename),
                    "image_SA.mat", "segmentation_all_phases.mat", "segmentation_ED.mat",
                    "segmentation_ES.mat", "original.mat", "resultado.mat"
                ]
                for mf in set(mat_files_to_update):
                    mf_path = os.path.join(base_dir, mf)
                    if os.path.exists(mf_path) and mf.lower().endswith('.mat'):
                        try:
                            m_dict = scipy.io.loadmat(mf_path)
                            m_dict['voxel_size'] = new_vs
                            save_mat_dict(mf_path, m_dict)
                        except Exception as e:
                            print(f"Failed to update voxel_size in {mf}: {e}")

            self.update_cardiac_metrics_ui()
            self.update_metadata_display()
            self.update_views()
            self.status_bar.showMessage(
                f"Calibration updated: {new_vs[0]:.2f} × {new_vs[1]:.2f} × {new_vs[2]:.2f} mm ({self.med_image.voxel_volume_ml:.4f} mL/vx)",
                5000
            )

    def on_units_toggled(self):
        """Toggle volume units between mL and Voxels."""
        if self.radio_unit_ml.isChecked():
            self.unit_mode = "ml"
        else:
            self.unit_mode = "voxels"

        self.viewer_canvas.unit_mode = self.unit_mode
        self.update_cardiac_metrics_ui()
        self.update_views()

    def toggle_units_mode(self):
        """Toggle unit mode shortcut 'U'."""
        if self.unit_mode == "ml":
            self.radio_unit_vx.setChecked(True)
        else:
            self.radio_unit_ml.setChecked(True)

    # ==========================================================================
    # AI SEGMENTATION AND QUANTIFICATION
    # ==========================================================================
    def show_segmentation_dialog(self):
        """Open modal configuration dialog for deep learning segmentation."""
        if self.med_image is None or self.med_image.data_4d is None:
            QMessageBox.warning(
                self, "No Image Loaded",
                "Please open a 4D medical imaging dataset (.mat, .nii, .npy) via 'Open Image' or Ctrl+O before running AI quantification."
            )
            return

        dlg = SegmentationDialog(self, med_image=self.med_image, current_phase=self.current_phase, default_output_dir=self.last_directory)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            mode, target_phase, save_mat, device_choice, output_dir = dlg.get_configuration()
            if output_dir:
                self.last_directory = output_dir
            self.start_segmentation_worker(mode, target_phase, save_mat, device_choice, output_dir=output_dir)

    def run_direct_segmentation(self, mode="fast_edes"):
        """Run a specific segmentation mode directly."""
        if self.med_image is None or self.med_image.data_4d is None:
            QMessageBox.warning(
                self, "No Image Loaded",
                "Please open a 4D medical imaging dataset before running segmentation."
            )
            return

        target_dir = self.last_directory or (os.path.dirname(os.path.abspath(self.med_image.filename)) if self.med_image.filename else os.getcwd())
        self.start_segmentation_worker(mode=mode, target_phase=self.current_phase, save_mat=True, device_choice="auto", output_dir=target_dir)

    def start_segmentation_worker(self, mode="fast_edes", target_phase=0, save_mat=True, device_choice="auto", output_dir=None):
        """Launch background segmentation worker and display progress dialog."""
        if self.med_image is None or self.med_image.data_4d is None:
            return

        if output_dir is None:
            output_dir = self.last_directory or (os.path.dirname(os.path.abspath(self.med_image.filename)) if self.med_image.filename else os.getcwd())

        mode_titles = {
            "fast_edes": "Fast ED / ES Quantification (VENTSEG AI)",
            "full_4d": "Full 4D Spatio-Temporal Segmentation (segmentation_all_phases.mat)",
            "middle": "Central Slice Segmentation (segmentation_all_phases.mat)",
            "single_phase": f"3D Single-Phase Segmentation - Phase {target_phase + 1}"
        }
        title_str = mode_titles.get(mode, "AI Segmentation")

        progress_dlg = SegmentationProgressDialog(self, title=title_str)

        worker = SegmentationWorker(
            data_4d=self.med_image.data_4d,
            file_path=self.med_image.filename,
            mode=mode,
            target_phase=target_phase,
            device_str=device_choice,
            save_mat=save_mat,
            voxel_size=self.med_image.voxel_size,
            output_dir=output_dir,
            parent=self
        )

        progress_dlg.set_worker(worker)

        result_container = {'data': None, 'error': None}

        def on_success(res):
            result_container['data'] = res

        def on_fail(err):
            result_container['error'] = err

        worker.segmentation_finished.connect(on_success)
        worker.segmentation_error.connect(on_fail)

        worker.start()
        progress_dlg.exec()

        if result_container['error']:
            self.on_segmentation_failed(result_container['error'])
        elif result_container['data']:
            self.on_segmentation_completed(result_container['data'])

    def on_segmentation_completed(self, result_dict):
        """Apply segmentation results to image and update UI."""
        if self.med_image is None:
            return

        mask_4d = result_dict.get('mask_4d')
        mask_name = result_dict.get('mask_filename', 'AI Segmentation')
        mode = result_dict.get('mode', '')
        out_dir = result_dict.get('output_dir', '')

        if mask_4d is not None:
            self.med_image.set_mask(mask_4d, mask_filename=mask_name)
            self.chk_show_mask.setChecked(True)
            self.show_mask = True

            if hasattr(self, 'btn_view_report'):
                self.btn_view_report.setEnabled(True)

            self.update_cardiac_metrics_ui()
            self.update_metadata_display()
            self.update_views()

            saved_files = result_dict.get('saved_files', [])
            saved_str = f" Saved files in '{out_dir}': {', '.join(saved_files)}." if (saved_files and out_dir) else (f" Saved files: {', '.join(saved_files)}." if saved_files else "")
            self.status_bar.showMessage(f"AI segmentation completed successfully.{saved_str}", 6000)

            # Open clinical report
            report_dlg = CardiacReportDialog(
                self, metrics=self.med_image.cardiac_metrics, filename=self.med_image.filename, mode=mode
            )
            report_dlg.exec()

    def on_segmentation_failed(self, error_msg):
        """Display error dialog on segmentation failure."""
        QMessageBox.critical(
            self, "AI Segmentation Error",
            f"Failed to complete the segmentation process:\n\n{error_msg}\n\n"
            "Please check that PyTorch, Albumentations, and model weight checkpoints are properly installed."
        )
        self.status_bar.showMessage("Error during AI segmentation.")

    def show_clinical_report_dialog(self):
        """Display current ventricular clinical report."""
        if self.med_image is None or not self.med_image.has_mask:
            QMessageBox.information(
                self, "No Cardiac Metrics Available",
                "No segmentation mask is loaded to generate a clinical report.\n"
                "Please run 'Quantify / Segment' or load a mask file."
            )
            return

        report_dlg = CardiacReportDialog(
            self, metrics=self.med_image.cardiac_metrics, filename=self.med_image.filename
        )
        report_dlg.exec()

    # ==========================================================================
    # FILE LOADING
    # ==========================================================================
    def open_file_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open 4D Medical Image", self.last_directory,
            "Medical Images (*.mat *.nii *.nii.gz *.npy *.npz);;MATLAB Files (*.mat);;NIfTI Files (*.nii *.nii.gz);;NumPy Files (*.npy *.npz);;All Files (*.*)"
        )
        if filepath:
            self.last_directory = os.path.dirname(os.path.abspath(filepath))
            self.load_file(filepath)

    def load_default_file(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(self.last_directory, "image_SA.mat"),
            os.path.join(os.getcwd(), "image_SA.mat"),
            os.path.join(script_dir, "image_SA.mat"),
            os.path.join(self.last_directory, "original.mat"),
            os.path.join(os.getcwd(), "original.mat"),
            os.path.join(script_dir, "original.mat")
        ]
        target = None
        for p in candidates:
            if os.path.exists(p):
                target = p
                break

        if target:
            self.load_file(target)
        else:
            QMessageBox.warning(self, "File Not Found", f"Could not find 'image_SA.mat' or 'original.mat' in:\n{self.last_directory}")

    def open_mask_dialog(self):
        if self.med_image is None:
            QMessageBox.warning(self, "Attention", "Please open a primary 4D medical image first.")
            return

        start_dir = os.path.dirname(os.path.abspath(self.med_image.filename)) if self.med_image.filename else self.last_directory
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Segmentation Mask", start_dir,
            "Segmentation Files (*.mat *.nii *.nii.gz *.npy);;MATLAB Files (*.mat);;NIfTI Files (*.nii *.nii.gz);;All Files (*.*)"
        )
        if filepath:
            try:
                mask_obj = load_file_4d(filepath)
                self.last_directory = os.path.dirname(os.path.abspath(filepath))
                self.med_image.set_mask(mask_obj.data_4d, mask_filename=os.path.basename(filepath))
                self.chk_show_mask.setChecked(True)
                self.show_mask = True
                self.update_cardiac_metrics_ui()
                self.update_metadata_display()
                self.update_views()
                self.status_bar.showMessage(f"Segmentation mask loaded from: {os.path.basename(filepath)}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Mask", f"Failed to load mask:\n{str(e)}")

    def load_file(self, filepath):
        try:
            self.status_bar.showMessage(f"Loading dataset: {os.path.basename(filepath)}...")
            QApplication.processEvents()

            self.med_image = load_file_4d(filepath)
            self.last_directory = os.path.dirname(os.path.abspath(filepath))
            self.current_slice = 0
            self.current_phase = 0

            # Auto-load companion mask if present in same directory
            base_dir = os.path.dirname(os.path.abspath(filepath))
            candidates_masks = [
                ("segmentation_all_phases.mat", "segmentation_all_phases.mat (Auto)"),
                ("segmentation_ED.mat", "segmentation_ED.mat (Auto)"),
                ("resultado.mat", "resultado.mat (Auto)"),
                ("middle.mat", "middle.mat (Auto)")
            ]

            loaded_mask_name = ""
            for mask_fname, mask_label in candidates_masks:
                mask_full = os.path.join(base_dir, mask_fname)
                if os.path.exists(mask_full) and os.path.abspath(filepath) != os.path.abspath(mask_full):
                    try:
                        mask_obj = load_file_4d(mask_full)
                        self.med_image.set_mask(mask_obj.data_4d, mask_filename=mask_label)
                        loaded_mask_name = mask_fname
                        break
                    except Exception:
                        pass

            self.slider_slice.blockSignals(True)
            self.slider_slice.setRange(0, self.med_image.num_slices - 1)
            self.slider_slice.setValue(0)
            self.slider_slice.blockSignals(False)

            self.slider_phase.blockSignals(True)
            self.slider_phase.setRange(0, self.med_image.num_phases - 1)
            self.slider_phase.setValue(0)
            self.slider_phase.blockSignals(False)

            self.current_row = self.med_image.num_rows // 2
            self.current_col = self.med_image.num_cols // 2
            self.viewer_canvas.selected_row = self.current_row
            self.viewer_canvas.selected_col = self.current_col

            self.set_auto_contrast()
            self.update_cardiac_metrics_ui()
            self.update_metadata_display()
            self.update_views()

            msg = f"Dataset loaded: {os.path.basename(filepath)}"
            if self.med_image.has_voxel_size_metadata:
                msg += f" [Voxel: {self.med_image.voxel_size[0]:.2f}×{self.med_image.voxel_size[1]:.2f}×{self.med_image.voxel_size[2]:.2f} mm]"
            if loaded_mask_name:
                msg += f" (Companion mask '{loaded_mask_name}' automatically associated)"
            self.status_bar.showMessage(msg, 5000)

        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Could not load medical dataset:\n{str(e)}")
            self.status_bar.showMessage("Error loading file.")

    # ==========================================================================
    # NAVIGATION AND CARDIAC PHASES (ED / ES)
    # ==========================================================================
    def set_slice(self, slice_idx):
        if self.med_image is None:
            return
        slice_idx = max(0, min(slice_idx, self.med_image.num_slices - 1))
        if slice_idx != self.current_slice:
            self.current_slice = slice_idx
            self.slider_slice.setValue(slice_idx)
            self.update_views()

    def go_to_middle_slice(self):
        if self.med_image is None:
            return
        mid = self.med_image.num_slices // 2
        self.set_slice(mid)

    def on_slice_slider_changed(self, value):
        self.current_slice = value
        self.update_views()

    def set_phase(self, phase_idx):
        if self.med_image is None:
            return
        phase_idx = max(0, min(phase_idx, self.med_image.num_phases - 1))
        if phase_idx != self.current_phase:
            self.current_phase = phase_idx
            self.slider_phase.setValue(phase_idx)
            self.update_views()

    def on_phase_slider_changed(self, value):
        self.current_phase = value
        self.update_views()

    def jump_to_ed_phase(self):
        if self.med_image and self.med_image.has_mask:
            self.set_phase(self.med_image.ed_phase)
            self.status_bar.showMessage(f"End-Diastole (ED): Phase {self.med_image.ed_phase + 1}", 3000)

    def jump_to_es_phase(self):
        if self.med_image and self.med_image.has_mask:
            self.set_phase(self.med_image.es_phase)
            self.status_bar.showMessage(f"End-Systole (ES): Phase {self.med_image.es_phase + 1}", 3000)

    def on_slice_scroll(self, delta):
        self.set_slice(self.current_slice + delta)

    def on_phase_scroll(self, delta):
        self.set_phase(self.current_phase + delta)

    def on_grid_slice_selected(self, slice_idx):
        self.set_slice(slice_idx)

    def on_grid_phase_selected(self, phase_idx):
        self.set_phase(phase_idx)

    def on_curve_mode_toggled(self):
        if self.radio_vol3d.isChecked():
            self.viewer_canvas.curve_mode = "volume_3d"
        else:
            self.viewer_canvas.curve_mode = "area_2d"
        self.update_views()

    # ==========================================================================
    # CARDIAC CINE PLAYER
    # ==========================================================================
    def toggle_cine_playback(self):
        if self.med_image is None or self.med_image.num_phases <= 1:
            return

        if self.is_playing:
            self.cine_timer.stop()
            self.is_playing = False
            self.btn_play_pause.setText("Play Cine")
            self.btn_play_pause.setObjectName("playButton")
            self.btn_play_pause.setStyleSheet("")
        else:
            interval_ms = int(1000.0 / max(1, self.fps))
            self.cine_timer.start(interval_ms)
            self.is_playing = True
            self.btn_play_pause.setText("Pause Cine")
            self.btn_play_pause.setObjectName("pauseButton")
            self.btn_play_pause.setStyleSheet("")

    def advance_cine_phase(self):
        if self.med_image is None:
            return
        next_phase = (self.current_phase + 1) % self.med_image.num_phases
        self.current_phase = next_phase
        self.slider_phase.blockSignals(True)
        self.slider_phase.setValue(next_phase)
        self.slider_phase.blockSignals(False)
        self.update_views()

    def on_fps_changed(self, val):
        self.fps = val
        if self.is_playing:
            self.cine_timer.setInterval(int(1000.0 / max(1, self.fps)))

    # ==========================================================================
    # CONTRAST AND COLORMAPS
    # ==========================================================================
    def on_cmap_changed(self, idx):
        self.cmap_name = self.combo_cmap.currentData()
        self.update_views()

    def set_auto_contrast(self):
        if self.med_image is None:
            return
        p1, p99 = np.percentile(self.med_image.data_4d, (1.0, 99.0))
        self.vmin = float(p1)
        self.vmax = float(p99) if p99 > p1 else float(p1 + 1.0)

        data_min = self.med_image.min_val
        data_max = self.med_image.max_val
        data_range = max(1e-5, data_max - data_min)

        width_val = int(((self.vmax - self.vmin) / data_range) * 1000)
        level_val = int((((self.vmin + self.vmax) / 2.0 - data_min) / data_range) * 1000)

        self.slider_window.blockSignals(True)
        self.slider_level.blockSignals(True)
        self.slider_window.setValue(max(1, min(1000, width_val)))
        self.slider_level.setValue(max(0, min(1000, level_val)))
        self.slider_window.blockSignals(False)
        self.slider_level.blockSignals(False)

        self.update_views()

    def set_minmax_contrast(self):
        if self.med_image is None:
            return
        self.vmin = self.med_image.min_val
        self.vmax = self.med_image.max_val
        self.slider_window.blockSignals(True)
        self.slider_level.blockSignals(True)
        self.slider_window.setValue(1000)
        self.slider_level.setValue(500)
        self.slider_window.blockSignals(False)
        self.slider_level.blockSignals(False)
        self.update_views()

    def reset_contrast(self):
        self.set_auto_contrast()

    def on_wl_slider_changed(self):
        if self.med_image is None:
            return
        data_min = self.med_image.min_val
        data_max = self.med_image.max_val
        data_range = max(1e-5, data_max - data_min)

        w_ratio = self.slider_window.value() / 1000.0
        l_ratio = self.slider_level.value() / 1000.0

        window = w_ratio * data_range
        level = data_min + l_ratio * data_range

        self.vmin = level - window / 2.0
        self.vmax = level + window / 2.0
        self.update_views()

    def on_toggle_mask(self, checked):
        self.show_mask = checked
        self.update_views()

    def on_mask_opacity_changed(self, value):
        self.mask_opacity = value / 100.0
        self.update_views()

    def on_pixel_clicked(self, row, col):
        self.current_row = row
        self.current_col = col
        self.update_views()

    def update_hover_status(self, row, col, val):
        if self.med_image is not None:
            dx = self.med_image.voxel_size[0]
            dy = self.med_image.voxel_size[1]
            dz = self.med_image.voxel_size[2]
            pos_x_mm = col * dx
            pos_y_mm = row * dy
            pos_z_mm = self.current_slice * dz
            self.status_bar.showMessage(
                f"Voxel: [Row: {row} ({pos_y_mm:.1f} mm), Col: {col} ({pos_x_mm:.1f} mm), Slice: {self.current_slice + 1} ({pos_z_mm:.1f} mm)] | Intensity: {val:.2f}",
                2000
            )
        else:
            self.status_bar.showMessage(f"Position: [Row: {row}, Col: {col}] | Intensity: {val:.2f}", 2000)

    def toggle_sidebar(self):
        """Toggle sidebar visibility to maximize viewport area."""
        if self.sidebar_scroll.isVisible():
            self.sidebar_scroll.hide()
            self.btn_toggle_sidebar.setText("Show Panel")
            self.status_bar.showMessage("Sidebar hidden. View maximized.", 2500)
        else:
            self.sidebar_scroll.show()
            self.btn_toggle_sidebar.setText("Hide Panel")
            self.status_bar.showMessage("Sidebar visible.", 2500)
        QTimer.singleShot(60, self.update_views)

    def on_tab_changed(self, index):
        if index == 1:
            self.grid_view_widget.update_grid(
                self.med_image, self.current_slice, self.current_phase, self.vmin, self.vmax, self.cmap_name
            )
        else:
            self.update_views()

    # ==========================================================================
    # UPDATE VIEWS AND UI METRICS
    # ==========================================================================
    def update_views(self):
        if self.med_image is None:
            return

        total_slices = self.med_image.num_slices
        total_phases = self.med_image.num_phases

        self.lbl_slice_badge.setText(f"{self.current_slice + 1} / {total_slices}")
        self.lbl_phase_badge.setText(f"{self.current_phase + 1} / {total_phases}")

        slice_2d = self.med_image.get_slice_2d(self.current_slice, self.current_phase)
        mask_2d = self.med_image.get_mask_slice_2d(self.current_slice, self.current_phase) if self.show_mask else None

        if self.tabs.currentIndex() == 0:
            self.viewer_canvas.update_display(
                slice_data=slice_2d,
                vmin=self.vmin,
                vmax=self.vmax,
                cmap_name=self.cmap_name,
                mask_slice=mask_2d,
                mask_alpha=self.mask_opacity,
                show_mask=self.show_mask,
                current_slice=self.current_slice,
                total_slices=total_slices,
                current_phase=self.current_phase,
                total_phases=total_phases,
                med_image=self.med_image
            )
        else:
            self.grid_view_widget.update_grid(
                self.med_image, self.current_slice, self.current_phase, self.vmin, self.vmax, self.cmap_name
            )

    def update_cardiac_metrics_ui(self):
        """Update metrics sidebar panel with LV, RV, and MYO values in mL or Voxels."""
        if self.med_image is None or not self.med_image.has_mask:
            self.lbl_metric_ed_phase.setText("<b>ED Phase:</b> -")
            self.lbl_metric_es_phase.setText("<b>ES Phase:</b> -")
            self.lbl_metric_edv.setText("<b>EDV (LV):</b> -")
            self.lbl_metric_esv.setText("<b>ESV (LV):</b> -")
            self.lbl_metric_sv.setText("<b>Stroke Volume (SV):</b> -")
            self.lbl_metric_ef.setText("<b>Ejection Frac. (EF):</b> -")
            self.lbl_metric_myo.setText("<b>Myocardium:</b> -")
            self.lbl_metric_rv.setText("<b>Right Ventricle (RV):</b> -")
            self.btn_side_ed.setEnabled(False)
            self.btn_side_es.setEnabled(False)
            self.tb_btn_ed.setEnabled(False)
            self.tb_btn_es.setEnabled(False)
            return

        m = self.med_image.cardiac_metrics
        ed_p = m.get('ed_phase', 0)
        es_p = m.get('es_phase', 0)

        ef = m.get('lv_ef', 0.0)
        ef_color = "#51cf66" if ef >= 50.0 else ("#fcc419" if ef >= 40.0 else "#ff6b6b")

        if self.unit_mode == "ml":
            edv = m.get('lv_edv_ml', 0.0)
            esv = m.get('lv_esv_ml', 0.0)
            sv = m.get('lv_sv_ml', 0.0)
            myo_ed = m.get('myo_ed_ml', 0.0)
            myo_mass = m.get('myo_mass_g', 0.0)
            rv_ef = m.get('rv_ef', 0.0)
            rv_edv = m.get('rv_edv_ml', 0.0)
            rv_esv = m.get('rv_esv_ml', 0.0)

            self.lbl_metric_edv.setText(f"<b>EDV (LV):</b> <span style='color:#74c0fc; font-weight:bold;'>{edv:.2f} mL</span>")
            self.lbl_metric_esv.setText(f"<b>ESV (LV):</b> <span style='color:#74c0fc; font-weight:bold;'>{esv:.2f} mL</span>")
            self.lbl_metric_sv.setText(f"<b>SV:</b> <span style='color:#00adb5; font-weight:bold;'>{sv:.2f} mL</span>")
            self.lbl_metric_myo.setText(f"<b>Myocardium (ED):</b> {myo_ed:.2f} mL  |  Mass: <span style='color:#51cf66; font-weight:bold;'>{myo_mass:.1f} g</span>")
            self.lbl_metric_rv.setText(f"<b>RV (EF):</b> {rv_ef:.1f}% (EDV: {rv_edv:.1f} mL | ESV: {rv_esv:.1f} mL)")
        else:
            edv = m.get('lv_edv_vx', 0.0)
            esv = m.get('lv_esv_vx', 0.0)
            sv = m.get('lv_sv_vx', 0.0)
            myo_ed = m.get('myo_ed_vx', 0.0)
            rv_ef = m.get('rv_ef', 0.0)
            rv_edv = m.get('rv_edv_vx', 0.0)
            rv_esv = m.get('rv_esv_vx', 0.0)

            self.lbl_metric_edv.setText(f"<b>EDV (LV):</b> {edv:,.0f} vx")
            self.lbl_metric_esv.setText(f"<b>ESV (LV):</b> {esv:,.0f} vx")
            self.lbl_metric_sv.setText(f"<b>SV:</b> {sv:,.0f} vx")
            self.lbl_metric_myo.setText(f"<b>Myocardium (ED):</b> {myo_ed:,.0f} vx")
            self.lbl_metric_rv.setText(f"<b>RV (EF):</b> {rv_ef:.1f}% (EDV: {rv_edv:,.0f} vx | ESV: {rv_esv:,.0f} vx)")

        self.lbl_metric_ed_phase.setText(f"<b>ED Phase:</b> <span style='color:#51cf66; font-weight:bold;'>Phase {ed_p + 1}</span>")
        self.lbl_metric_es_phase.setText(f"<b>ES Phase:</b> <span style='color:#ff6b6b; font-weight:bold;'>Phase {es_p + 1}</span>")
        self.lbl_metric_ef.setText(f"<b>EF:</b> <span style='color:{ef_color}; font-weight:bold;'>{ef:.1f}%</span>")

        self.btn_side_ed.setText(f"Jump to ED (Phase {ed_p + 1})")
        self.btn_side_es.setText(f"Jump to ES (Phase {es_p + 1})")
        self.btn_side_ed.setEnabled(True)
        self.btn_side_es.setEnabled(True)
        self.tb_btn_ed.setEnabled(True)
        self.tb_btn_es.setEnabled(True)

    def update_metadata_display(self):
        if self.med_image is None:
            self.lbl_info_file.setText("<b>Dataset:</b> None")
            self.lbl_info_mask.setText("<b>Segmentation:</b> None")
            self.lbl_info_dims.setText("<b>Dimensions (4D):</b> -")
            self.lbl_info_voxel_size.setText("<b>Voxel Dimensions:</b> -")
            self.lbl_info_voxel_vol.setText("<b>Unit Voxel Volume:</b> -")
            self.lbl_info_pixel_area.setText("<b>2D Pixel Area:</b> -")
            self.lbl_info_dtype.setText("<b>Data Type:</b> -")
            self.lbl_info_range.setText("<b>Dynamic Range:</b> -")
            return

        self.lbl_info_file.setText(f"<b>Dataset:</b> {os.path.basename(self.med_image.filename)} (Var: {self.med_image.var_name})")
        mask_text = self.med_image.mask_filename if self.med_image.has_mask else "None"
        self.lbl_info_mask.setText(f"<b>Segmentation:</b> {mask_text}")
        self.lbl_info_dims.setText(
            f"<b>Dimensions (4D):</b> {self.med_image.num_rows} × {self.med_image.num_cols} × {self.med_image.num_slices} slices × {self.med_image.num_phases} phases"
        )
        vx = self.med_image.voxel_size
        self.lbl_info_voxel_size.setText(
            f"<b>Voxel Dimensions:</b> <span style='color:#00adb5; font-weight:bold;'>{vx[0]:.2f} × {vx[1]:.2f} × {vx[2]:.2f} mm</span>"
        )
        self.lbl_info_voxel_vol.setText(
            f"<b>Unit Voxel Volume:</b> {self.med_image.voxel_volume_mm3:.3f} mm³ ({self.med_image.voxel_volume_ml:.4f} mL)"
        )
        self.lbl_info_pixel_area.setText(
            f"<b>2D Pixel Area:</b> {self.med_image.pixel_area_mm2:.3f} mm² ({self.med_image.pixel_area_cm2:.4f} cm²)"
        )
        self.lbl_info_dtype.setText(f"<b>Data Type:</b> {self.med_image.dtype_str}")
        self.lbl_info_range.setText(f"<b>Dynamic Range:</b> Min: {self.med_image.min_val:.1f}, Max: {self.med_image.max_val:.1f}, Mean: {self.med_image.mean_val:.1f}")

    # ==========================================================================
    # EXPORTING COMPLETE RESULTS
    # ==========================================================================
    def export_all_results_to_folder(self, target_dir=None):
        """Save all generated assets (.mat segmentation masks, clinical text report, CSV volume curves, snapshots, and cine GIF) to target folder."""
        if self.med_image is None or self.med_image.data_4d is None:
            QMessageBox.warning(
                self, "No Image Loaded",
                "Please open a 4D medical imaging dataset before saving results."
            )
            return

        if not target_dir:
            initial_dir = self.last_directory or os.getcwd()
            target_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory for All Results", initial_dir
            )

        if not target_dir:
            return

        target_dir = os.path.abspath(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        self.last_directory = target_dir

        saved_files = []
        try:
            # 1. Original 4D dataset with voxel calibration (.mat)
            save_mat_dict(os.path.join(target_dir, 'image_SA.mat'), {
                'image_SA': self.med_image.data_4d,
                'voxel_size': self.med_image.voxel_size
            })
            saved_files.append("image_SA.mat (4D volume with calibrated voxel dimensions)")

            # 2. Segmentation masks
            if self.med_image.has_mask and self.med_image.mask_4d is not None:
                ed_p = self.med_image.ed_phase
                es_p = self.med_image.es_phase

                # 4D mask
                save_mat_dict(os.path.join(target_dir, 'segmentation_all_phases.mat'), {
                    'segmentation_all_phases': self.med_image.mask_4d,
                    'voxel_size': self.med_image.voxel_size,
                    'ed_phase': ed_p + 1,
                    'es_phase': es_p + 1
                })
                saved_files.append("segmentation_all_phases.mat (4D multi-phase segmentation mask)")

                # ED 3D mask
                ed_3d = self.med_image.mask_4d[:, :, :, ed_p]
                save_mat_dict(os.path.join(target_dir, 'segmentation_ED.mat'), {
                    'segmentation_ED': ed_3d,
                    'voxel_size': self.med_image.voxel_size,
                    'ed_phase': ed_p + 1
                })
                saved_files.append(f"segmentation_ED.mat (3D mask at End-Diastole - Phase {ed_p + 1})")

                # ES 3D mask
                es_3d = self.med_image.mask_4d[:, :, :, es_p]
                save_mat_dict(os.path.join(target_dir, 'segmentation_ES.mat'), {
                    'segmentation_ES': es_3d,
                    'voxel_size': self.med_image.voxel_size,
                    'es_phase': es_p + 1
                })
                saved_files.append(f"segmentation_ES.mat (3D mask at End-Systole - Phase {es_p + 1})")

            # 3. Clinical Text Report (.txt)
            report_txt_path = os.path.join(target_dir, 'ventricular_clinical_report.txt')
            self._write_clinical_report_file(report_txt_path)
            saved_files.append("ventricular_clinical_report.txt (Quantitative parameters in mL and g)")

            # 4. Volumetric Curves (.csv)
            if self.med_image.has_mask and self.med_image.num_phases > 0:
                csv_path = os.path.join(target_dir, 'cardiac_volume_curves.csv')
                self._write_curves_csv_file(csv_path)
                saved_files.append("cardiac_volume_curves.csv (Time-series volumetric curves for LV, RV, MYO)")

            # 5. Snapshot of Current View (PNG)
            png_path = os.path.join(target_dir, 'current_view_snapshot.png')
            self.viewer_canvas.fig.savefig(png_path, dpi=300, facecolor='#1a1b22', edgecolor='none')
            saved_files.append("current_view_snapshot.png (300 DPI high-resolution capture)")

            # 6. Animated Cardiac Cine GIF
            if imageio is not None and self.med_image.num_phases > 1:
                gif_path = os.path.join(target_dir, f'cardiac_cine_slice_{self.current_slice + 1}.gif')
                self._generate_cine_gif_file(gif_path)
                saved_files.append(f"cardiac_cine_slice_{self.current_slice + 1}.gif (Animated cine loop)")

            file_list_str = "\n".join([f"  • {f}" for f in saved_files])
            msg = (
                f"<h3>All results have been exported successfully</h3>"
                f"<p><b>Target Folder:</b><br><code>{target_dir}</code></p>"
                f"<p><b>Generated Files:</b><br><pre>{file_list_str}</pre></p>"
            )
            QMessageBox.information(self, "Results Saved", msg)
            self.status_bar.showMessage(f"All results saved to: {target_dir}", 6000)

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export results to:\n{target_dir}\n\nError details:\n{str(e)}"
            )

    def _write_clinical_report_file(self, filepath):
        m = self.med_image.cardiac_metrics or {}
        ed_p = m.get('ed_phase', 0)
        es_p = m.get('es_phase', 0)
        edv_ml = m.get('lv_edv_ml', 0.0)
        esv_ml = m.get('lv_esv_ml', 0.0)
        sv_ml = m.get('lv_sv_ml', 0.0)
        ef = m.get('lv_ef', 0.0)
        edv_vx = m.get('lv_edv_vx', 0.0)
        esv_vx = m.get('lv_esv_vx', 0.0)
        sv_vx = m.get('lv_sv_vx', 0.0)

        myo_ed_ml = m.get('myo_ed_ml', 0.0)
        myo_mass_g = m.get('myo_mass_g', 0.0)

        rv_edv_ml = m.get('rv_edv_ml', 0.0)
        rv_esv_ml = m.get('rv_esv_ml', 0.0)
        rv_sv_ml = m.get('rv_sv_ml', 0.0)
        rv_ef = m.get('rv_ef', 0.0)

        vx_size = self.med_image.voxel_size
        vx_vol_ml = self.med_image.voxel_volume_ml

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 75 + "\n")
            f.write("VENTSEG - VENTRICULAR QUANTIFICATION AND CLINICAL FUNCTION REPORT\n")
            f.write("=" * 75 + "\n\n")
            f.write(f"Source Dataset: {self.med_image.filename}\n")
            f.write(f"Generation Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"4D Dimensions: {self.med_image.num_rows} × {self.med_image.num_cols} × {self.med_image.num_slices} slices × {self.med_image.num_phases} phases\n")
            f.write(f"Spatial Calibration: dx={vx_size[0]:.4f} mm, dy={vx_size[1]:.4f} mm, dz={vx_size[2]:.4f} mm\n")
            f.write(f"Unit Voxel Volume: {vx_vol_ml * 1000:.4f} mm³ ({vx_vol_ml:.6f} mL)\n\n")
            if self.med_image.has_mask:
                f.write(f"End-Diastole (ED): Phase {ed_p + 1}\n")
                f.write(f"End-Systole (ES): Phase {es_p + 1}\n\n")
                f.write("--- LEFT VENTRICLE (LV) ---\n")
                f.write(f"End-Diastolic Volume (EDV): {edv_ml:.2f} mL ({edv_vx:,.0f} voxels)\n")
                f.write(f"End-Systolic Volume (ESV): {esv_ml:.2f} mL ({esv_vx:,.0f} voxels)\n")
                f.write(f"Stroke Volume (SV): {sv_ml:.2f} mL ({sv_vx:,.0f} voxels)\n")
                f.write(f"Ejection Fraction (EF): {ef:.2f} %\n\n")
                f.write("--- MYOCARDIUM ---\n")
                f.write(f"Myocardial Volume at ED: {myo_ed_ml:.2f} mL\n")
                f.write(f"Estimated Myocardial Mass (density 1.05 g/mL): {myo_mass_g:.2f} g\n\n")
                f.write("--- RIGHT VENTRICLE (RV) ---\n")
                f.write(f"End-Diastolic Volume (RV EDV): {rv_edv_ml:.2f} mL\n")
                f.write(f"End-Systolic Volume (RV ESV): {rv_esv_ml:.2f} mL\n")
                f.write(f"Stroke Volume (RV SV): {rv_sv_ml:.2f} mL\n")
                f.write(f"Ejection Fraction (RV EF): {rv_ef:.2f} %\n")
            else:
                f.write("Note: No segmentation mask associated with this dataset.\n")
            f.write("=" * 75 + "\n")

    def _write_curves_csv_file(self, filepath):
        total_p = self.med_image.num_phases
        lv_ml = self.med_image.volume_curves_ml.get('lv', np.zeros(total_p))
        myo_ml = self.med_image.volume_curves_ml.get('myo', np.zeros(total_p))
        rv_ml = self.med_image.volume_curves_ml.get('rv', np.zeros(total_p))

        lv_vx = self.med_image.volume_curves_voxels.get('lv', np.zeros(total_p))
        myo_vx = self.med_image.volume_curves_voxels.get('myo', np.zeros(total_p))
        rv_vx = self.med_image.volume_curves_voxels.get('rv', np.zeros(total_p))

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Phase,LV_Volume_mL,LV_Volume_voxels,MYO_Volume_mL,MYO_Volume_voxels,RV_Volume_mL,RV_Volume_voxels\n")
            for t in range(total_p):
                f.write(f"{t + 1},{lv_ml[t]:.4f},{lv_vx[t]:.0f},{myo_ml[t]:.4f},{myo_vx[t]:.0f},{rv_ml[t]:.4f},{rv_vx[t]:.0f}\n")

    def _generate_cine_gif_file(self, filepath):
        if imageio is None:
            return
        frames = []
        seg_colors = [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.25, 0.25, self.mask_opacity),
            (0.2, 0.85, 0.35, self.mask_opacity),
            (0.2, 0.6, 1.0, self.mask_opacity)
        ]
        seg_cmap = ListedColormap(seg_colors)

        for p in range(self.med_image.num_phases):
            img = self.med_image.get_slice_2d(self.current_slice, p)
            norm = np.clip((img - self.vmin) / max(1e-5, (self.vmax - self.vmin)), 0, 1)
            cmap = plt.get_cmap(self.cmap_name)
            colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)

            if self.show_mask and self.med_image.has_mask:
                mask_s = self.med_image.get_mask_slice_2d(self.current_slice, p)
                if mask_s is not None:
                    rgba_m = seg_cmap(mask_s)
                    alpha = rgba_m[:, :, 3:4]
                    rgb_m = (rgba_m[:, :, :3] * 255).astype(np.uint8)
                    colored = (colored * (1 - alpha) + rgb_m * alpha).astype(np.uint8)

            frames.append(colored)

        duration = 1.0 / max(1, self.fps)
        imageio.mimsave(filepath, frames, duration=duration, loop=0)

    def export_current_snapshot(self):
        if self.med_image is None:
            QMessageBox.warning(self, "No Image Loaded", "Please open an image dataset before capturing a snapshot.")
            return
        default_path = os.path.join(self.last_directory, "slice_snapshot.png")
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", default_path, "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if filepath:
            self.last_directory = os.path.dirname(os.path.abspath(filepath))
            try:
                self.viewer_canvas.fig.savefig(filepath, dpi=300, facecolor='#1a1b22', edgecolor='none')
                QMessageBox.information(self, "Export Successful", f"Snapshot saved to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export snapshot image:\n{str(e)}")

    def export_cine_gif(self):
        if self.med_image is None:
            QMessageBox.warning(self, "No Image Loaded", "Please open an image dataset before exporting cardiac cine GIF.")
            return
        if imageio is None:
            QMessageBox.warning(self, "Feature Unavailable", "The 'imageio' library is required to export animated GIFs.")
            return

        default_path = os.path.join(self.last_directory, f"cardiac_cine_slice_{self.current_slice + 1}.gif")
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Cardiac Cine GIF", default_path, "Animated GIF (*.gif)"
        )
        if filepath:
            self.last_directory = os.path.dirname(os.path.abspath(filepath))
            try:
                self._generate_cine_gif_file(filepath)
                QMessageBox.information(self, "GIF Created", f"Animated cardiac cine GIF saved to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to generate GIF:\n{str(e)}")

    # ==========================================================================
    # ABOUT AND HELP DIALOGS
    # ==========================================================================
    def show_about_dialog(self):
        dlg = AboutDialog(self)
        dlg.exec()

    def show_shortcuts_dialog(self):
        QMessageBox.information(
            self, "Keyboard Shortcuts Guide",
            "<b>Available Keyboard Shortcuts:</b><br><br>"
            "• <b>Ctrl + O:</b> Open 4D medical imaging dataset (.mat, .nii, .npy)<br>"
            "• <b>Ctrl + M:</b> Load segmentation mask<br>"
            "• <b>Ctrl + Shift + S:</b> Save all results to a directory<br>"
            "• <b>Ctrl + R:</b> AI Cardiac Quantification & Segmentation (ResNet34-UNet)<br>"
            "• <b>Ctrl + K:</b> Spatial Calibration & Voxel Size (in mm)<br>"
            "• <b>U:</b> Toggle Volume Units (mL <-> Voxels)<br>"
            "• <b>Space:</b> Play / Pause Cardiac Cine<br>"
            "• <b>Left / Right Arrow:</b> Previous / Next Cardiac Phase<br>"
            "• <b>Up / Down Arrow:</b> Next / Previous Slice<br>"
            "• <b>E:</b> Jump to End-Diastole (ED)<br>"
            "• <b>S:</b> Jump to End-Systole (ES)<br>"
            "• <b>M:</b> Toggle Segmentation Mask Overlay<br>"
            "• <b>R:</b> Reset Brightness & Contrast (Window/Level)<br>"
            "• <b>Click on Volume Curve:</b> Jump directly to clicked cardiac phase<br>"
            "• <b>Ctrl + S:</b> Save high-resolution snapshot (PNG)<br>"
            "• <b>Ctrl + G:</b> Export cardiac cine as animated GIF"
        )


class AboutDialog(QDialog):
    """About dialog presenting software identity, academic attribution, research grants, and tooling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About VENTSEG 4D")
        self.setWindowIcon(get_app_icon())
        self.resize(660, 560)
        self.setStyleSheet(DARK_STYLE_SHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header Frame with Logo and Title
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a233a, stop:1 #141721);
                border: 1px solid #2d3856;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        h_layout = QHBoxLayout(header_frame)
        h_layout.setSpacing(14)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        png_path = os.path.join(base_dir, "images", "Logo.png")
        if os.path.exists(png_path):
            logo_lbl = QLabel()
            pixmap = QPixmap(png_path)
            if not pixmap.isNull():
                scaled_pix = pixmap.scaled(56, 56, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                logo_lbl.setPixmap(scaled_pix)
                logo_lbl.setStyleSheet("background: transparent;")
                h_layout.addWidget(logo_lbl)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(2)
        lbl_title = QLabel("VENTSEG 4D")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4da8da; background: transparent;")
        lbl_sub = QLabel("Cardiac Medical Image Viewer & AI Quantification Suite")
        lbl_sub.setStyleSheet("font-size: 12px; color: #cbd5e1; font-weight: 500; background: transparent;")
        lbl_ver = QLabel("Version 2.0 • ResNet34-UNet • PyQt6 • 4D Cine Workstation")
        lbl_ver.setStyleSheet("font-size: 11px; color: #94a3b8; background: transparent;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_sub)
        title_vbox.addWidget(lbl_ver)
        h_layout.addLayout(title_vbox)
        h_layout.addStretch()
        layout.addWidget(header_frame)

        # Content Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #21242d; border: 1px solid #2d313e; border-radius: 6px;")
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(14, 14, 14, 14)

        about_html = (
            "<h4 style='color:#4da8da; margin-bottom: 4px;'>Overview & Clinical Capabilities</h4>"
            "<p style='line-height: 1.45; color:#eceff4;'>"
            "<b>VENTSEG 4D</b> is a high-performance desktop medical workstation designed for 4D cine cardiac MRI visualization, "
            "deep learning automated ventricular segmentation (Left Ventricle cavity, Myocardium, Right Ventricle), "
            "automated End-Diastole (ED) and End-Systole (ES) phase detection, and quantitative clinical functional biomarkers computation "
            "(Ejection Fraction %, Stroke Volume mL, EDV/ESV mL, Myocardial Mass g)."
            "</p>"
            "<hr style='border: 1px solid #2d313e;'>"
            "<h4 style='color:#4da8da; margin-bottom: 4px;'>Academic Provenance & Original Framework</h4>"
            "<p style='line-height: 1.45; color:#eceff4;'>"
            "This interface is an advanced adaptation based on the original <b>VENTSEG Framework</b>:<br>"
            "<a href='https://github.com/JulioSoteloParraguez/VENTSEG-ventricular_segmentation_framework/tree/main' style='color:#4da8da; text-decoration: underline;'>"
            "https://github.com/JulioSoteloParraguez/VENTSEG-ventricular_segmentation_framework</a><br><br>"
            "The core code and deep learning algorithms were developed by <b>Alejandro León</b> as part of his master's thesis, "
            "under the academic guidance and supervision of professors <b>Dr. Julio Sotelo</b> and <b>Dr. Rodrigo Salas</b> "
            "from the <b>Universidad de Valparaíso</b> (Chile).<br>"
            "The VENTSEG Framework enables automated segmentation of both the left and right ventricles."
            "</p>"
            "<hr style='border: 1px solid #2d313e;'>"
            "<h4 style='color:#4da8da; margin-bottom: 4px;'>Research Funding & Grants</h4>"
            "<p style='line-height: 1.45; color:#eceff4;'>"
            "This research and development is supported by:<br>"
            "• <b>ANID - Millennium Science Initiative Program</b> — Grant <b>NCN17 129</b><br>"
            "• <b>ANID FONDECYT research initiation</b> — Grant <b>FONDECYT #11200481</b><br>"
            "• <b>ANID - Millennium Science Initiative Program</b> — Grant <b>ICN2021 004</b><br>"
            "• <b>ANID FONDECYT research initiation</b> — Grant <b>FONDECYT #1221938</b>"
            "</p>"
            "<hr style='border: 1px solid #2d313e;'>"
            "<h4 style='color:#4da8da; margin-bottom: 4px;'>AI-Assisted Interface Engineering</h4>"
            "<p style='line-height: 1.45; color:#eceff4;'>"
            "The interactive user interface, multi-view 4D cine rendering architecture, physical voxel calibration system, "
            "and comprehensive clinical reporting suite were developed and modernized with the assistance of <b>Google Antigravity</b>."
            "</p>"
        )

        about_text = QLabel(about_html)
        about_text.setWordWrap(True)
        about_text.setOpenExternalLinks(True)
        about_text.setStyleSheet("font-size: 12px; background: transparent;")
        content_layout.addWidget(about_text)
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

        # Close Button
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_close = QPushButton("Close")
        btn_close.setObjectName("accentButton")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)


# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================
def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ventseg.cardiac.suite.4d")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("VENTSEG 4D")
    app.setStyle("Fusion")
    app.setWindowIcon(get_app_icon())

    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    viewer = VentSegViewer4D(filepath)
    viewer.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
