================================================================================
VENTSEG 4D - Cardiac Medical Image Viewer & AI Quantification Suite
================================================================================

1. SYSTEM REQUIREMENTS:
   - Python: 3.10, 3.11, or 3.12 (Python 3.12 64-bit recommended)
   - OS: Windows 10/11, Linux, or macOS
   - Hardware: CPU (x86_64) or NVIDIA GPU with CUDA support

2. QUICK INSTALLATION & SETUP:
   When moving this folder to a new directory or machine, run:

   Windows (PowerShell):
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

3. RUNNING THE APPLICATION:
   - Launch the interactive viewer:
     python viewer_4d.py

   - Or launch directly loading a specific dataset (.mat or .nii):
     python viewer_4d.py path/to/dataset.mat

4. KEY FEATURES:
   - Multi-format Medical I/O: Supports 4D/3D MATLAB volumes (.mat) and NIfTI (.nii, .nii.gz).
   - Real-Time 4D Cine Playback: Cine loop animation with adjustable FPS, slice and phase sliders.
   - Multi-View Visualization: Single slice view, 2x2 multi-phase mosaic, and full spatial grid.
   - Deep Learning AI Segmentation: Embedded ResNet34-UNet architecture for Left Ventricle (LV),
     Right Ventricle (RV), and Myocardium (MYO).
   - Automatic Phase Detection: Automatic calculation of End-Diastole (ED) and End-Systole (ES).
   - Clinical Quantification Metrics: Ejection Fraction (EF %), Stroke Volume (SV mL),
     End-Diastolic Volume (EDV mL), End-Systolic Volume (ESV mL), and Myocardial Mass (g).
   - Multi-Format Export:
     * Segmented 4D Dataset (.mat)
     * Clinical Diagnostic Summary Report (.txt)
     * Volumetric Time-Series Curves (.csv)
     * Viewport High-Res Snapshots (.png, .jpg)
     * Animated Cine Cine-loops (.gif)

5. PROJECT DIRECTORY STRUCTURE:
   ├── viewer_4d.py          # Main PyQt6 GUI & application entry point
   ├── requirements.txt      # Python dependencies
   ├── README.md             # Complete documentation and user guide
   ├── readme.txt            # Plain text quickstart guide
   ├── images/               # Software logo, multi-resolution icon & preview
   │   ├── Logo.png
   │   ├── Logo.ico
   │   └── Main_Windows.png
   ├── models/               # ResNet34-UNet architecture & pre-trained weights (.pt)
   └── utils/                # Data loader, normalization, augmentation & post-processing

6. ACADEMIC PROVENANCE & RESEARCH GRANTS:
   - Original Framework: https://github.com/JulioSoteloParraguez/VENTSEG-ventricular_segmentation_framework/tree/main
   - Developed by Alejandro León as part of his Master's thesis, guided by professors
     Dr. Julio Sotelo and Dr. Rodrigo Salas from Universidad de Valparaíso (Chile).
   - Supported by:
     * ANID - Millennium Science Initiative Program - NCN17 129
     * ANID FONDECYT research initiation FONDECYT #11200481
     * ANID - Millennium Science Initiative Program - ICN2021 004
     * ANID FONDECYT research initiation FONDECYT #1221938

7. AI-ASSISTED INTERFACE ENGINEERING:
   - The interactive 4D workstation, PyQt6 graphical interface, real-time rendering
     engine, physical calibration system, and clinical reporting tools were developed
     and modernized with the assistance of Google Antigravity.
================================================================================
