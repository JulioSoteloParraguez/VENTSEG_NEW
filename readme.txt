================================================================================
VENTSEG 4D - Cardiac Medical Image Viewer & AI Quantification Suite
================================================================================

1. RESEARCH PURPOSE & ACADEMIC NATURE:
   This software is developed strictly for research, educational, and scientific
   transfer purposes to share with the community a deep learning model for
   cardiac ventricular segmentation. It was not created by or for certified
   clinical practice and is not a medical device.

2. SYSTEM REQUIREMENTS:
   - Python: 3.10, 3.11, or 3.12 (Python 3.12 64-bit recommended)
   - OS: Windows 10/11, Linux, or macOS
   - Hardware: CPU (x86_64) or NVIDIA GPU with CUDA support

3. QUICK INSTALLATION & SETUP:
   If you don't have python 3.12 you need to install this version. 
   Windows 
     - Download and install from https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe
  
   Open PowerShell of Cursor, Antigravity or Visual Studio and run:
     py -3.12 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install -r requirements.txt

   Windows (Command Prompt / CMD):
     py -3.12 -m venv .venv
     .\.venv\Scripts\activate.bat
     pip install -r requirements.txt

   Linux / macOS:
     python3.12 -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt

   -----------------------------------------------------------------------------
   * NVIDIA GPU ACCELERATION (OPTIONAL BUT STRONGLY RECOMMENDED):
   -----------------------------------------------------------------------------
   By default, `pip install -r requirements.txt` installs the standard/CPU build.
   To accelerate AI segmentation with an NVIDIA GPU:

   A) Standard NVIDIA GPUs (RTX 20xx, 30xx, 40xx, GTX 16xx, Quadro, A100):
      pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124

   B) Latest-Generation NVIDIA GPUs (Blackwell Architecture sm_120, RTX 50xx, RTX PRO Blackwell):
      New Blackwell generation GPUs require CUDA 12.8+ builds to provide sm_120 binary kernels:
      pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
   -----------------------------------------------------------------------------

4. DOWNLOAD PRE-TRAINED MODELS (GOOGLE DRIVE):
   Due to GitHub file size limits, the pre-trained neural network weights (~143 MB)
   are hosted on Google Drive:
   https://drive.google.com/drive/folders/1wZ_0Di2xuCjZOw6atymoHlt45K8AJHeV?usp=sharing

   Download the models folder and place it in the root directory:
   ventseg/models/model_resnet34_unet_scratch_best_dice.pt

5. DOWNLOAD SAMPLE INPUT DATA (GOOGLE DRIVE):
   An example input dataset is available on Google Drive:
   https://drive.google.com/drive/folders/1bKaooRMNtSt7z_AWuwHUdUfG1jtCdl1k?usp=sharing

   * Important: The voxel_size in the example file does NOT correspond to the
     real physical acquisition voxel size. To obtain accurate volumes (mL), check
     the real voxel size in the DICOM metadata and calibrate it either:
     - Directly from the application GUI (Ctrl+K or "Spatial Calibration / Voxel Size")
     - Directly in the MATLAB file (defining 'voxel_size = [dx, dy, dz]' in mm).

6. RUNNING THE APPLICATION:
   - Launch the interactive viewer:
     python viewer_4d.py

   - Or launch directly loading a specific dataset (.mat or .nii):
     python viewer_4d.py path/to/dataset.mat

7. KEY FEATURES:
   - Multi-format Medical I/O: Supports 4D/3D MATLAB volumes (.mat) and NIfTI (.nii, .nii.gz).
   - Real-Time 4D Cine Playback: Cine loop animation with adjustable FPS, slice and phase sliders.
   - Multi-View Visualization: Single slice view, 2x2 multi-phase mosaic, and full spatial grid.
   - Deep Learning AI Segmentation: Embedded ResNet34-UNet architecture for Left Ventricle (LV),
     Right Ventricle (RV), and Myocardium (MYO).
   - Automatic Phase Detection: Automatic calculation of End-Diastole (ED) and End-Systole (ES).
   - Cardiac Quantification Metrics: Ejection Fraction (EF %), Stroke Volume (SV mL),
     End-Diastolic Volume (EDV mL), End-Systolic Volume (ESV mL), and Myocardial Mass (g).
   - Multi-Format Export:
     * Segmented 4D Dataset (.mat)
     * Quantification Summary Report (.txt)
     * Volumetric Time-Series Curves (.csv)
     * Viewport High-Res Snapshots (.png, .jpg)
     * Animated Cine-loops (.gif)

8. PROJECT DIRECTORY STRUCTURE:
   ├── viewer_4d.py          # Main PyQt6 GUI & application entry point
   ├── requirements.txt      # Python dependencies
   ├── README.md             # Complete documentation and user guide
   ├── readme.txt            # Plain text quickstart guide
   ├── images/               # Software logo, multi-resolution icon & preview
   │   ├── Logo.png
   │   ├── Logo.ico
   │   └── Main_Windows.png
   ├── models/               # Downloaded from Google Drive (ResNet34-UNet weights .pt)
   └── utils/                # Data loader, normalization, augmentation & post-processing

9. ACADEMIC PROVENANCE & RESEARCH GRANTS:
   - Original Framework: https://github.com/JulioSoteloParraguez/VENTSEG-ventricular_segmentation_framework/tree/main
   - Developed by Alejandro León as part of his Master's thesis, guided by professors
     Dr. Julio Sotelo and Dr. Rodrigo Salas from Universidad de Valparaíso (Chile).
   - Supported by:
     * ANID - Millennium Science Initiative Program - NCN17 129
     * ANID FONDECYT research initiation FONDECYT #11200481
     * ANID - Millennium Science Initiative Program - ICN2021 004
     * ANID FONDECYT research initiation FONDECYT #1221938

10. AI-ASSISTED INTERFACE ENGINEERING:
   - The interactive 4D workstation, PyQt6 graphical interface, real-time rendering
     engine, physical calibration system, and quantification reporting tools were developed
     and modernized with the assistance of Google Antigravity.
================================================================================
