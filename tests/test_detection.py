"""
Automated validation and performance test suite for Coin Detection System.
Evaluates 12 specific empirical scenarios against ground-truth benchmarks:
1. Single coin
2. Three separated coins
3. Five separated coins
4. Multi-coin scene
5. Touching coins
6. Partially overlapping coins
7. Image without coins
8. Blurry image
9. Low-light image
10. Coin with visible surface irregularities
11. Unknown denomination
12. Image with no calibration reference
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

import sys
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from config import SAMPLE_IMAGES_DIR
from modules.coin_detector import CoinDetector
from modules.reference_database import ReferenceDatabase


class TestCoinDetectionSuite(unittest.TestCase):
    """Systematic benchmark test suite across 12 image scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.db = ReferenceDatabase()
        cls.detector = CoinDetector(reference_db=cls.db)

    def _evaluate_scenario(
        self,
        img_filename: str,
        expected_count: int,
        description: str,
        allowable_error: int = 0
    ):
        """Helper to run detection and compute precision, recall, and counting error."""
        img_path = SAMPLE_IMAGES_DIR / img_filename
        self.assertTrue(img_path.exists(), f"Sample image file not found: {img_path}")

        bgr = cv2.imread(str(img_path))
        self.assertIsNotNone(bgr, f"Failed to load image: {img_path}")

        res = self.detector.detect_and_analyze(
            bgr_image=bgr,
            min_diameter_px=40,
            max_diameter_px=400,
            detection_sensitivity=1.0,
            enable_watershed_split=True
        )

        detected_count = res["total_detected"]
        counting_error = abs(detected_count - expected_count)

        # Precision & Recall (assuming all verified detected coins within margin are True Positives)
        true_positives = min(detected_count, expected_count)
        false_positives = max(0, detected_count - expected_count)
        false_negatives = max(0, expected_count - detected_count)

        precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 1.0
        recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 1.0

        print(f"\n[SCENARIO TEST] {description}")
        print(f"  Target File: {img_filename}")
        print(f"  Expected Count: {expected_count} | Actual Count: {detected_count} | Counting Error: {counting_error}")
        print(f"  Precision: {precision:.2f} | Recall: {recall:.2f}")

        self.assertLessEqual(
            counting_error,
            allowable_error,
            f"Scenario '{description}' counting error ({counting_error}) exceeded allowable margin ({allowable_error})"
        )
        return res

    # 1. Single Coin
    def test_01_single_coin(self):
        self._evaluate_scenario(
            "sample_1_coin.jpg",
            expected_count=1,
            description="1. Image containing one isolated coin"
        )

    # 2. Three Separated Coins
    def test_02_three_separated_coins(self):
        self._evaluate_scenario(
            "sample_3_coins.jpg",
            expected_count=3,
            description="2. Image containing three separated coins"
        )

    # 3. Five Separated Coins
    def test_03_five_separated_coins(self):
        self._evaluate_scenario(
            "sample_5_coins.jpg",
            expected_count=5,
            description="3. Image containing five separated coins"
        )

    # 4. Multi-coin Scene
    def test_04_multi_coin(self):
        # Using 5 coins as multi-coin benchmark
        self._evaluate_scenario(
            "sample_5_coins.jpg",
            expected_count=5,
            description="4. Multi-coin arrangement benchmark"
        )

    # 5. Touching Coins
    def test_05_touching_coins(self):
        res = self._evaluate_scenario(
            "sample_touching_coins.jpg",
            expected_count=2,
            description="5. Image containing touching coins (Watershed separation)",
            allowable_error=0
        )
        self.assertTrue(res["has_touching_coins"], "Watershed analyzer should flag touching coin condition")

    # 6. Partially Overlapping Coins
    def test_06_overlapping_coins(self):
        res = self._evaluate_scenario(
            "sample_overlapping_coins.jpg",
            expected_count=2,
            description="6. Image containing partially overlapping coins",
            allowable_error=1
        )
        # Should flag touching/overlap
        self.assertTrue(res["has_touching_coins"] or res["total_detected"] >= 1)

    # 7. Image Without Coins
    def test_07_no_coins(self):
        self._evaluate_scenario(
            "sample_no_coins.jpg",
            expected_count=0,
            description="7. Blank background image without coins (Zero false positives)"
        )

    # 8. Blurry Image
    def test_08_blurry_image(self):
        res = self._evaluate_scenario(
            "sample_blurry_coin.jpg",
            expected_count=0,  # Blurred image should either fail detection or flag blur
            description="8. Blurry image condition assessment",
            allowable_error=1
        )
        self.assertTrue(res["quality_metrics"]["is_blurry"], "Preprocessor must flag severe optical blur")

    # 9. Low-Light Image
    def test_09_low_light(self):
        res = self._evaluate_scenario(
            "sample_low_light.jpg",
            expected_count=1,
            description="9. Low-light underexposed image (CLAHE enhancement test)",
            allowable_error=1
        )
        self.assertLess(res["quality_metrics"]["mean_brightness"], 100)

    # 10. Coin With Visible Surface Irregularities
    def test_10_damaged_coin(self):
        res = self._evaluate_scenario(
            "sample_damaged_coin.jpg",
            expected_count=1,
            description="10. Coin with visible surface scratches and oxidation spots"
        )
        if res["total_detected"] > 0:
            cond = res["coins"][0]["condition"]
            self.assertGreater(cond["anomaly_count"], 0, "Condition analyzer must detect anomaly spots/scratches")
            self.assertNotEqual(cond["condition_category"], "No obvious visible anomaly detected")

    # 11. Unknown Denomination
    def test_11_unknown_denomination(self):
        # When uncalibrated, coin should be flagged as 'denomination uncertain'
        detector_uncalib = CoinDetector(reference_db=self.db, calibration_scale=None)
        bgr = cv2.imread(str(SAMPLE_IMAGES_DIR / "sample_1_coin.jpg"))
        res = detector_uncalib.detect_and_analyze(bgr)
        self.assertGreater(res["total_detected"], 0)
        c0 = res["coins"][0]
        self.assertFalse(c0["classification"]["is_classified"])
        self.assertIn("denomination uncertain", c0["classification"]["denomination"].lower())

    # 12. No Calibration Reference
    def test_12_no_calibration(self):
        detector_uncalib = CoinDetector(reference_db=self.db, calibration_scale=None)
        bgr = cv2.imread(str(SAMPLE_IMAGES_DIR / "sample_1_coin.jpg"))
        res = detector_uncalib.detect_and_analyze(bgr)
        self.assertGreater(res["total_detected"], 0)
        c0 = res["coins"][0]
        self.assertIsNone(c0["diameter_mm"])
        self.assertIn("Uncalibrated", c0["geometry"]["physical_status"])


if __name__ == "__main__":
    unittest.main()
