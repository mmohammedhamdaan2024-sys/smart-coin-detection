---
title: Smart Coin Detection System
emoji: 🪙
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Smart Coin Detection, Counting, Classification, and Condition Analysis System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Complete%20%26%20Tested-emerald.svg)]()

An intermediate-to-advanced Digital Image Processing system engineered in Python, OpenCV, NumPy, and Streamlit. The system autonomously detects, counts, outlines, measures, classifies, and screens the surface condition of coins from digital photographs without requiring external cloud APIs or GPU hardware.

---

## 🌟 Key Capabilities
- **Digital Image Processing Pipeline:** Step-by-step visualization across 9 distinct transformations (Grayscale, Gaussian Denoising, CLAHE contrast enhancement, Otsu + Adaptive thresholding, Canny edges, elliptical morphology, Watershed segmentation, and annotated overlays).
- **Touching & Overlapping Coin Separation:** Marker-Controlled Watershed segmentation and Euclidean Distance Transforms divide contacting coins along topological contact crests.
- **Radius-Prioritized Non-Maximum Suppression:** Eliminates phantom sub-circles caused by internal die embossings and portraits.
- **Exact Geometric Quantification:** Measures radii, diameters, bounding boxes, centroid coordinates, contour area, arc perimeter, aspect ratio, and dimensionless circularity ($C = \frac{4\pi A}{P^2}$).
- **Physical Millimeter Calibration:** Linear scaling from user-provided reference coins (e.g. US Quarter = 24.26mm) with perspective distortion caveats.
- **Multi-Modal Denomination Matching:** Matches diameter and color material signatures (copper-red, brass-gold, silver-nickel, bimetallic core/ring) against an extensible JSON database. Displays *"Coin detected — denomination uncertain"* when evidence is ambiguous.
- **Surface Condition Screening:** High-pass scratch detection, tarnish Delta-$E$ calculation, and edge irregularity analysis.
- **Scientific Measurement Integrity:** Optical images never fabricate physical weight or authoritative numismatic grades (e.g. MS-65). Official mint masses or manual scale inputs are clearly disclosed.
- **Reports & Auditing:** Instant downloads for CSV spreadsheets, machine-readable JSON schemas, ReportLab PDF inspection reports, and cropped coin images. Local SQLite history audit trails.

---

## 📁 Project Folder Structure
```
coin_detection_system/
│
├── app.py                      # Main Streamlit web application dashboard
├── requirements.txt            # System dependencies
├── README.md                   # Setup guide and instructions
├── config.py                   # Central hyperparameters, thresholds, and paths
│
├── modules/                    # Modular computer vision backend
│   ├── __init__.py
│   ├── image_loader.py         # File ingestion, EXIF correction, dimension validation
│   ├── preprocessing.py        # Grayscale, Gaussian blur, CLAHE, Canny, morphology
│   ├── segmentation.py         # Distance transform & Marker-controlled Watershed
│   ├── measurements.py         # Geometric metrics, circularity, physical calibration
│   ├── reference_database.py   # JSON reference catalog CRUD operations
│   ├── coin_classifier.py      # Color signature, annulus, and diameter classification
│   ├── condition_analyzer.py   # Surface scratch, spot, discoloration, rim screening
│   ├── weight_analysis.py      # Reference mass lookups and scale weight logging
│   ├── age_analysis.py         # Minting year, issue period, and chronological age
│   ├── coin_detector.py        # Central 12-step detection and analysis orchestrator
│   ├── report_generator.py     # CSV, JSON, and PDF summary generators
│   └── history_manager.py      # SQLite local session persistence
│
├── data/
│   ├── coin_reference.json     # Standard demonstration coin database (USD, EUR, GBP, INR)
│   ├── generate_samples.py     # Script to generate benchmark test images
│   ├── README.md               # Data schema documentation
│   └── sample_images/          # Benchmark images (single, multiple, touching, blur, etc.)
│
├── tests/                      # Automated test suite (23 unit & validation tests)
│   ├── __init__.py
│   ├── test_preprocessing.py   # Preprocessing stage transformations
│   ├── test_measurements.py    # Circularity formulas and calibration mechanics
│   ├── test_detection.py       # 12 empirical benchmark scenario tests
│   └── test_reports.py         # CSV, JSON, PDF, and SQLite lifecycle tests
│
├── outputs/                    # Export output directory and SQLite history database
│
└── docs/                       # University project documentation
    ├── project_report.md       # Full 27-section comprehensive academic report
    ├── algorithm_explanation.md# In-depth mathematical formulas & OpenCV details
    └── limitations.md          # Real-world limitations & numismatic disclosures
```

---

## 🚀 Installation and Execution (Windows & VS Code)

### Prerequisites
- Python 3.11 or higher installed on Windows.
- Visual Studio Code installed.

### Step 1: Open Project in VS Code
1. Open Visual Studio Code.
2. Select **File > Open Folder...** and select the `coin_detection_system` directory.
3. Open the integrated terminal (`Ctrl + ~` or **Terminal > New Terminal**).

### Step 2: Create and Activate Virtual Environment
Run the following PowerShell commands in the terminal:
```powershell
python -m venv .venv
.venv\Scripts\activate
```
*(You should see `(.venv)` in your terminal prompt).*

### Step 3: Upgrade Pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Automated Test Suite
Verify that all 23 unit and scenario tests pass:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### Step 5: Launch the Interactive Web Application
```powershell
python -m streamlit run app.py
```
Streamlit will automatically open your default browser at:
`http://localhost:8501`

---

## 📖 User Guide

### 1. Ingesting an Image
- Navigate to **2. Analyze New Image** in the sidebar.
- Drag and drop a photo containing coins (JPG, PNG, WEBP), or select one of the 9 preset laboratory test scenarios from the dropdown.
- Click **🚀 Analyze Coins**.

### 2. Exploring Results
- **1. Dashboard:** Displays top-level summary cards (coins detected, unique denominations, partial face value, anomalies, average diameter, and image quality rating).
- **3. Coin Detection Results:** Use the interactive dropdown to select individual coins. The selected coin is highlighted with a cyan ring on the full image. Review geometric metrics, denomination match, reference mass, and download cropped coin images.
- **4. Coin Condition Analysis:** Review localized anomaly masks (red overlays) highlighting surface scratches and discoloration flecks.
- **5. Image Processing Visualization:** Explore the 9 transformation stages across dedicated tabs with complete technical explanations and parameters.
- **6. Analysis History:** Inspect previously executed analyses stored in local SQLite (`outputs/history.db`).
- **7. Export Reports:** One-click downloads for CSV spreadsheets, JSON data, PDF reports, and annotated PNG images.

---

## 🧪 Scientific Validation Benchmark
The system was validated against 12 documented empirical test scenarios:

| Scenario | Target File | Expected | Detected | Precision | Recall | Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Single Isolated Coin | `sample_1_coin.jpg` | 1 | 1 | 1.00 | 1.00 | 0 |
| 2. Three Separated Coins | `sample_3_coins.jpg` | 3 | 3 | 1.00 | 1.00 | 0 |
| 3. Five Mixed Coins | `sample_5_coins.jpg` | 5 | 5 | 1.00 | 1.00 | 0 |
| 4. Multi-Coin Benchmark | `sample_5_coins.jpg` | 5 | 5 | 1.00 | 1.00 | 0 |
| 5. Touching Coins | `sample_touching_coins.jpg` | 2 | 2 | 1.00 | 1.00 | 0 |
| 6. Partially Overlapping Coins | `sample_overlapping_coins.jpg` | 2 | 1 | 1.00 | 0.50 | 1* |
| 7. Blank Scene (No Coins) | `sample_no_coins.jpg` | 0 | 0 | 1.00 | 1.00 | 0 |
| 8. Blurry Image | `sample_blurry_coin.jpg` | 0 | 0 | 1.00 | 1.00 | 0 |
| 9. Low-Light Image | `sample_low_light.jpg` | 1 | 1 | 1.00 | 1.00 | 0 |
| 10. Damaged / Scratched Coin | `sample_damaged_coin.jpg` | 1 | 1 | 1.00 | 1.00 | 0 |
| 11. Unknown Denomination | `sample_1_coin.jpg` (uncalib.) | 1 | 1 | 1.00 | 1.00 | 0 |
| 12. No Calibration Reference | `sample_1_coin.jpg` (uncalib.) | 1 | 1 | 1.00 | 1.00 | 0 |

---

## 📜 Academic Integrity and Numismatic Disclosures
1. **Measurement Integrity:** Planar millimeter estimations require explicit physical calibration and orthogonal camera alignment.
2. **Weight Disclosure:** 2D optical photographs cannot measure physical mass. Official mint masses or manual scale measurements are clearly distinguished.
3. **Screening vs Grading:** Image-based condition screening identifies visual anomalies; it does not replace certified professional numismatic grading (e.g. PCGS / NGC Sheldon Scale).
