"""
Configuration settings for the Smart Coin Detection System.
Centralizes default parameters, paths, and operational thresholds.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCS_DIR = BASE_DIR / "docs"

# Data Files
REFERENCE_DB_PATH = DATA_DIR / "coin_reference.json"
HISTORY_DB_PATH = OUTPUTS_DIR / "history.db"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"

# Image Processing Defaults
DEFAULT_MAX_IMAGE_DIM = 1600  # Rescaling maximum dimension to prevent memory/performance issues
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]

# Detection Algorithm Hyperparameters
DEFAULT_MIN_DIAMETER_PX = 30
DEFAULT_MAX_DIAMETER_PX = 600
DEFAULT_BLUR_KERNEL_SIZE = 9  # Gaussian blur kernel size (odd integer)
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_GRID_SIZE = (8, 8)

# Canny Edge Detection Thresholds
DEFAULT_CANNY_LOW = 40
DEFAULT_CANNY_HIGH = 120

# Hough Circle Detection Defaults
DEFAULT_HOUGH_DP = 1.2
DEFAULT_HOUGH_MIN_DIST = 45
DEFAULT_HOUGH_PARAM1 = 100   # Upper threshold for internal Canny edge detector
DEFAULT_HOUGH_PARAM2 = 38    # Accumulator threshold for circle centers

# Contour-Based Filtering Defaults
DEFAULT_MIN_CIRCULARITY = 0.65  # 4 * pi * Area / Perimeter^2
DEFAULT_MIN_AREA_PX = 400
DEFAULT_MAX_ASPECT_RATIO_DIFF = 0.35  # abs(w/h - 1.0)

# Condition Analysis Thresholds
ANOMALY_DEFECT_THRESHOLD_PCT = 3.5  # Percentage of anomalous pixels inside inner disk
CONDITION_CATEGORIES = {
    "NO_OBVIOUS": "No obvious visible anomaly detected",
    "MINOR": "Minor visible surface irregularities",
    "MODERATE": "Moderate visible irregularities",
    "SIGNIFICANT": "Significant visible irregularities",
    "UNABLE": "Unable to assess reliably"
}

# UI & Visualization Settings
MAX_DISPLAY_WIDTH = 800
COLOR_PALETTE = {
    "primary": "#1E3A8A",       # Deep Blue
    "secondary": "#0D9488",     # Teal
    "accent": "#F59E0B",        # Amber Gold
    "danger": "#EF4444",        # Crimson
    "success": "#10B981",       # Emerald
    "card_bg": "#1F2937",
    "text": "#F9FAFB"
}
