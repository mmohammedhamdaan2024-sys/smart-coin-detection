"""
Unit tests for Geometric and Physical Measurement module.
Tests circularity, bounding box extraction, scale derivation, and millimeter calibration.
"""

import unittest
import math
from pathlib import Path
import numpy as np
import cv2

import sys
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from modules.measurements import GeometricAnalyzer


class TestGeometricAnalyzer(unittest.TestCase):
    """Verifies geometric accuracy, circularity formulas, and calibration mechanics."""

    def setUp(self):
        self.analyzer_uncalib = GeometricAnalyzer(calibration_scale_mm_per_px=None)
        self.analyzer_calib = GeometricAnalyzer(calibration_scale_mm_per_px=0.15)

    def test_circularity_formula(self):
        """Verifies circularity formula C = 4*pi*Area / (Perimeter^2)."""
        # Theoretical continuous circle of radius 50
        r = 50.0
        area = math.pi * (r ** 2)
        perimeter = 2 * math.pi * r
        c = GeometricAnalyzer.calculate_circularity(area, perimeter)
        self.assertAlmostEqual(c, 1.0, places=3)

        # Synthetic discrete raster circle in OpenCV
        canvas = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(canvas, (100, 100), 50, 255, -1)
        cnts, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_raster = max(cnts, key=cv2.contourArea)
        area_raster = cv2.contourArea(c_raster)
        perim_raster = cv2.arcLength(c_raster, True)
        circ_score = GeometricAnalyzer.calculate_circularity(area_raster, perim_raster)
        self.assertGreater(circ_score, 0.85, "Rasterized circle should exhibit circularity >= 0.85")

    def test_calibration_derivation(self):
        """Verifies scale = known_mm / measured_px."""
        scale = GeometricAnalyzer.derive_scale_from_reference(
            measured_diameter_px=160.0,
            known_diameter_mm=24.26
        )
        self.assertIsNotNone(scale)
        self.assertAlmostEqual(scale, 0.151625, places=5)

    def test_physical_diameter_uncalibrated_vs_calibrated(self):
        """Verifies calibrated analyzer produces millimeter values while uncalibrated remains None."""
        canvas = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(canvas, (100, 100), 40, 255, -1)
        cnts, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(cnts, key=cv2.contourArea)

        uncalib_res = self.analyzer_uncalib.analyze_geometry(contour, circle_fit=(100, 100, 40))
        self.assertIsNone(uncalib_res["diameter_mm"])
        self.assertEqual(uncalib_res["diameter_px"], 80.0)

        calib_res = self.analyzer_calib.analyze_geometry(contour, circle_fit=(100, 100, 40))
        self.assertIsNotNone(calib_res["diameter_mm"])
        self.assertEqual(calib_res["diameter_mm"], 12.0)  # 80px * 0.15 mm/px = 12.0 mm


if __name__ == "__main__":
    unittest.main()
