<div align="center">
  <img src="images/Logo.png" alt="VENTSEG 4D Logo" width="140" />
  <h1>VENTSEG 4D</h1>
  <h3>Cardiac Medical Image Viewer & AI Quantification Suite</h3>
  <p>
    <b>Interactive 4D Cine Cardiac MRI Workstation • Deep Learning Multi-Class Segmentation • Quantitative Functional Biomarkers</b>
  </p>

  [![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
  [![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://riverbankcomputing.com/software/pyqt/)
  [![PyTorch](https://img.shields.io/badge/AI-PyTorch%20%2F%20ResNet34--UNet-EE4C2C.svg)](https://pytorch.org/)
  [![Institution](https://img.shields.io/badge/Universidad%20de%20Valpara%C3%ADso-Chile-003366.svg)](https://www.uv.cl/)
  [![Engineered with Google Antigravity](https://img.shields.io/badge/Engineered%20with-Google%20Antigravity-4285F4.svg)](#ai-assisted-interface-engineering)
</div>

---

<div align="center">
  <img src="images/Main_Windows.png" alt="VENTSEG 4D Main Application Interface" width="100%" />
  <p><em>Figure 1: Main graphical user interface of VENTSEG 4D featuring interactive 4D cine playback, real-time slice scrolling, multi-class segmentation overlays (Left Ventricle, Myocardium, Right Ventricle), synchronized volumetric curves in milliliters (mL), and comprehensive cardiac quantification metrics.</em></p>
</div>

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Installation & Portable Setup](#installation--portable-setup)
- [Download Models (Pre-trained Weights)](#download-models-pre-trained-weights)
- [How to Run](#how-to-run)
- [Project Directory Structure](#project-directory-structure)
- [Cardiac Metrics & AI Models](#cardiac-metrics--ai-models)
- [Supported Formats & Export Capabilities](#supported-formats--export-capabilities)
- [Research Purpose & Disclaimer](#research-purpose--disclaimer)
- [Academic Origin & Acknowledgments](#academic-origin--acknowledgments)
- [Funding & Research Grants](#funding--research-grants)
- [AI-Assisted Interface Engineering](#ai-assisted-interface-engineering)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Overview

**VENTSEG 4D** was developed exclusively for **research and academic purposes** to transfer and share with the scientific and medical imaging community a deep learning model developed for cardiac ventricular segmentation. It offers high-performance 4D (3D spatial + time) cine MRI exploration along with an embedded deep learning architecture (**ResNet34-UNet**) capable of segmenting:
1. **Left Ventricular Cavity (LV)** - Class 1
2. **Myocardium (MYO)** - Class 2
3. **Right Ventricular Cavity (RV)** - Class 3

The suite automates the identification of **End-Diastole (ED)** and **End-Systole (ES)** cardiac phases, computes volumetric time-series curves, calculates functional biomarkers (ejection fraction **EF %**, stroke volume **SV**, and myocardial mass), and exports research figures, summary reports, and animated cine loops.

---

## Key Features

- **High-Performance 4D Cine Playback**: Smooth time-series cine loop playback with adjustable framerate (FPS), scrub bars, play/pause controls, and slice indexing.
- **Multi-View Modes**:
  - **Single Slice View**: High-detail slice-by-slice inspection with overlay masks and interactive zoom/pan.
  - **2x2 Multi-Phase View**: Simultaneous side-by-side comparison of multiple cardiac phases.
  - **Full Slice Mosaic**: Complete spatial anatomical coverage across all Z-slices.
- **Automated AI Segmentation**: In-memory deep learning inference utilizing the pre-trained weights (`models/model_resnet34_unet_scratch_best_dice.pt`). Supports CPU and NVIDIA CUDA GPU execution.
- **Automated Cardiac Phase Detection**: Automatic extraction of peak expansion (ED) and peak contraction (ES) from time-series volume curves.
- **Quantitative Cardiac Biomarkers Calculation**:
  - Left & Right Ventricle Ejection Fraction (LV EF %, RV EF %)
  - Stroke Volume (SV in mL)
  - End-Diastolic Volume (EDV in mL) & End-Systolic Volume (ESV in mL)
  - Myocardial Mass (in grams, using standard myocardial density of 1.05 g/cm³)
- **Export & Reporting Options**:
  - Segmented 4D Dataset (`.mat`)
  - Formatted Quantification Summary Report (`.txt`)
  - Volumetric Time-Series Curves (`.csv`)
  - High-Resolution Viewport Figures (`.png`, `.jpg`)
  - Animated Cine-loops (`.gif`)

---

## System Requirements

- **Operating System**: Windows 10/11 (64-bit), Linux (x86_64), or macOS.
- **Python Version**: Python 3.10, 3.11, or 3.12 (**Python 3.12 64-bit recommended**).
- **Hardware**:
  - Minimum 8 GB RAM (16 GB recommended for large 4D volumes).
  - CPU (Intel/AMD) or NVIDIA GPU with CUDA support for accelerated AI segmentation.

---

## Installation & Portable Setup

If you copy or move this application folder to another machine or directory, you can set it up and run it immediately:

### 1. Create a Virtual Environment

Open a terminal in the application folder:

**Windows (PowerShell / Command Prompt):**
```powershell
py -3.12 -m venv .venv
```

**Linux / macOS:**
```bash
python3.12 -m venv .venv
```

### 2. Activate the Virtual Environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If script execution is blocked on PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**Windows CMD:**
```cmd
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

All required libraries (`PyQt6`, `torch`, `torchvision`, `albumentations`, `opencv-python`, `nibabel`, `numpy`, `scipy`, `matplotlib`, `imageio`, `Pillow`) will be installed automatically.

---

## Download Models (Pre-trained Weights)

Due to GitHub's file size limit (files > 100 MB cannot be hosted directly in the repository), the pre-trained neural network weights (`model_resnet34_unet_scratch_best_dice.pt`, ~143 MB) and model architecture files are hosted on Google Drive:

📥 **[Download Models Folder (Google Drive)](https://drive.google.com/drive/folders/1wZ_0Di2xuCjZOw6atymoHlt45K8AJHeV?usp=sharing)**

### Setup Steps:
1. Open the [Google Drive link](https://drive.google.com/drive/folders/1wZ_0Di2xuCjZOw6atymoHlt45K8AJHeV?usp=sharing).
2. Download the `models` folder or its files (`__init__.py`, `resnet.py`, and `model_resnet34_unet_scratch_best_dice.pt`).
3. Place the `models/` folder directly in the root directory of this project:
   ```text
   ventseg/
   └── models/
       ├── __init__.py
       ├── resnet.py
       └── model_resnet34_unet_scratch_best_dice.pt
   ```

---

## How to Run

### Interactive GUI Mode
With the virtual environment activated, run:
```bash
python viewer_4d.py
```
Then use **File > Open Image...** to load a dataset.

### Direct Dataset Loading
You can pass the path of a `.mat` or `.nii` file directly as a command-line argument:
```bash
python viewer_4d.py path/to/image_SA.mat
```

---

## Project Directory Structure

```text
ventseg/
├── viewer_4d.py                 # Main PyQt6 GUI and application logic
├── requirements.txt             # Python dependencies
├── README.md                    # Detailed documentation and guide
├── readme.txt                   # Plaintext quick-start reference
├── images/
│   ├── Logo.png                 # Application logo (PNG)
│   ├── Logo.ico                 # Multi-resolution application icon (ICO)
│   └── Main_Windows.png         # Main application interface preview
├── models/                      # Downloaded from Google Drive
│   ├── __init__.py              # Model selector and package interface
│   ├── resnet.py                # ResNet34-UNet architecture definitions
│   └── model_resnet34_unet_scratch_best_dice.pt  # Pre-trained AI weights (~143 MB)
└── utils/
    ├── __init__.py              # Utilities package interface
    ├── data_augmentation.py     # Pre-processing and test-time augmentation
    ├── dataload.py              # Medical image loader and normalizer
    └── training.py              # Multiclass mask handling and reshaping
```

---

## Cardiac Metrics & AI Models

The application performs automated segmentation using a 2D ResNet34-UNet model evaluated slice-by-slice across 3D spatial slices ($Z$) and cardiac temporal frames ($T$).

### Metric Formulations:
- **Stroke Volume (SV)**:
  $$\text{SV} = \text{EDV} - \text{ESV}$$
- **Ejection Fraction (EF)**:
  $$\text{EF} (\%) = \left( \frac{\text{SV}}{\text{EDV}} \right) \times 100$$
- **Myocardial Mass**:
  $$\text{Mass (g)} = \text{Volume}_{\text{MYO}} (\text{cm}^3) \times 1.05 \text{ g/cm}^3$$

---

## Supported Formats & Export Capabilities

- **Input Formats**:
  - MATLAB 4D/3D volumes (`.mat` with fields `img`, `image_SA`, `data`, `volume`, or first 3D/4D matrix).
  - NIfTI images (`.nii`, `.nii.gz`).
  - NumPy arrays (`.npy`, `.npz`).
- **Export Formats**:
  - MATLAB `.mat` workspace containing raw volumes, segmentation masks, and metrics.
  - Formatted text report (`.txt`) with metadata and calculated quantitative results.
  - Volumetric time curves (`.csv`).
  - High-resolution figures (`.png`, `.jpg`).
  - Animated Cine-loops (`.gif`).

---

## Research Purpose & Disclaimer

> [!IMPORTANT]
> **Research & Community Knowledge Transfer**
> 
> This software is an academic research suite developed to demonstrate and transfer to the community a deep learning model for automated ventricular segmentation in cine cardiac MRI. It was **not** created by or for certified clinical practice, and it is **not** intended for primary clinical diagnosis or medical treatment decisions.

---

## Academic Origin & Acknowledgments

This software suite and graphical interface is an advanced, modernized adaptation built upon the foundational **VENTSEG Framework**:

> **Original Repository:** [VENTSEG - Ventricular Segmentation Framework](https://github.com/JulioSoteloParraguez/VENTSEG-ventricular_segmentation_framework/tree/main)

The original code, deep learning methodology, and neural network training routines were developed by **Alejandro León** as part of his Master's thesis, under the academic guidance and supervision of professors **Dr. Julio Sotelo** and **Dr. Rodrigo Salas** from the **Universidad de Valparaíso** (Chile).

The **VENTSEG Framework** enables automated multi-class deep learning segmentation of both the left and right cardiac ventricles (Left Ventricular cavity, Myocardium, and Right Ventricular cavity) in cine cardiac magnetic resonance imaging (MRI).

---

## Funding & Research Grants

This research and its software developments were funded and supported by the National Agency for Research and Development of Chile (**ANID**):

- **ANID - Millennium Science Initiative Program** — Grant **NCN17 129**
- **ANID FONDECYT Research Initiation** — Grant **FONDECYT #11200481**
- **ANID - Millennium Science Initiative Program** — Grant **ICN2021 004**
- **ANID FONDECYT Research Initiation** — Grant **FONDECYT #1221938**

---

## AI-Assisted Interface Engineering

The modern interactive user interface, multi-view 4D cine rendering architecture, physical voxel calibration system, quantification reporting suite, and high-performance workflow engineering of this software were developed and enhanced with the assistance of **Google Antigravity**, leveraging advanced agentic AI coding capabilities for medical and scientific application design.

---

## Troubleshooting & FAQ

- **Error: "Failed to import deep learning modules"**: Ensure your virtual environment is active and that the `models/` folder has been downloaded from Google Drive and placed in the project root (`.\.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`).
- **GPU vs CPU**: The application automatically detects available NVIDIA CUDA devices. If no CUDA-capable GPU is found, it automatically falls back to CPU computation.
- **Python Version**: If using Python 3.12, binary wheels are used for all dependencies for fast, compiler-free installation.
