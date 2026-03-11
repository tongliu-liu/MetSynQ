# MetSynQ: An Automated and Intelligent Peak Mapping Workflow for Targeted & Untargeted LC-MS

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Coming_Soon-red.svg)](#) 

> **Note:** This repository contains the official implementation of the paper: *"[Insert Your Paper Title Here]"*.

## 📖 Overview

MetSynQ is a robust, high-throughput pipeline designed for both **Multiple Reaction Monitoring (MRM) / Widely-Targeted** and **Untargeted** mass spectrometry data processing. By integrating state-of-the-art deep learning object detection (YOLOv11) with classical signal processing and clustering algorithms, this workflow automates the transition from raw MS data to highly confident, cross-sample consistent peak integration matrices.

**Key Features:**

- **Dual-Mode Capability:** Fully supports both pre-defined MRM transitions (Targeted) and XCMS-based global feature discovery (Untargeted).
- **Automated Data Conversion:** Seamless conversion of vendor-specific `.wiff` files to open `.mzML` formats via Dockerized ProteoWizard.
- **Deep Learning-Assisted Peak Extraction:** Utilizes state-of-the-art YOLO models to recognize complex peak shapes that traditional algorithms miss.
- **Robust Clustering & Anomaly Detection:** Employs DBSCAN and multithreaded group-based inconsistency correction to eliminate false positives and align features.
- **Mathematical Refinding:** Intelligently "re-calls" missing peaks in specific samples to ensure a complete, NA-free quantitative matrix.

---

## ⚙️ Pipeline Architecture

1. **Pre-processing:** Raw `.wiff` files are converted to `.mzML`.
   - *For Untargeted Mode:* Features are automatically extracted via XCMS to generate a target list (`ALL_ions.xlsx`).
2. **Extraction & Detection:** Peak signals are isolated as Regions of Interest (ROIs) and detected using YOLO.
3. **Clustering:** Retention times (RT) and mass transitions are clustered across samples.
4. **Refinding & Correction:** Missing peaks are "re-called", boundaries are dynamically adjusted, and intra-group anomalies are corrected.
5. **Post-processing & Visualization:** Generates final `.csv` peak tables and renders multi-threaded grid visualization plots.

---

## 🛠️ Installation

**1. Clone the repository**
```bash
git clone [https://github.com/](https://github.com/)tongliu-liu/MetSynQ.git
cd MetSynQ
```

**2. Python Environment**
```bash
conda create -n metsynq python=3.10
conda activate metsynq
pip install -r requirements.txt
```

**3. R Environment (Required for Untargeted Mode)**
If you intend to run the untargeted workflow, `XCMS` must be installed in your R environment:
```R
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(c("MSnbase", "xcms"))
```

---

## 📂 Data Preparation (Inputs)

Prepare your project directory with the following structure before running the pipeline:

```text
Your_Project_Folder/
├── mzML/                  # Directory containing .mzML or .wiff files
├── sample_info.csv        # Metadata mapping sample names to their biological groups
└── ALL_ions.xlsx          # Theoretical m/z and RT targets (Only required for Targeted mode)
```

---

## 🚀 Quick Start (Usage)

MetSynQ uses a unified command-line router `main.py`.

### Option A: Widely-Targeted (MRM) Analysis
```bash
python main.py targeted \
    --indir /path/to/Your_Project_Folder \
    --threads 16 \
    --type rp
```

### Option B: Untargeted Analysis
The untargeted mode requires additional parameters for XCMS feature extraction (e.g., ppm, minWidth).
```bash
python main.py untargeted \
    --indir /path/to/Your_Project_Folder \
    --threads 16 \
    --ppm 15 \
    --polarity positive \
    --minWidth 5 \
    --maxWidth 50 \
    --s2n 5
```

---

## 📈 Output & Results

Upon completion, the software generates a `temp/` folder (for intermediate calculations) and the final outputs directly in your project folder:

- **`peak_final.csv`**: The comprehensive feature matrix containing peak areas, heights, retention times, and SNR metrics for all samples.
- **`peak_table_filter.csv`**: The rigorously filtered and aligned peak matrix, perfectly formatted for downstream statistical analysis (e.g., PCA, PLS-DA, Differential analysis).
- **`picture/`**: A directory containing auto-generated, high-resolution grid plots highlighting the integrated peak boundaries (red shaded areas) against theoretical RTs (dashed lines) for rapid visual validation.

---

## 📖 Citation

If you use MetSynQ in your research, please cite:

> **[Your Name], et al. (2026). "[Insert Your Paper Title Here]". *[Journal Name]*, Volume(Issue), Pages.**

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
