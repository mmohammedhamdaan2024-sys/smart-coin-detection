"""
Unit tests for Image Preprocessing Pipeline.
Validates grayscale conversion, Gaussian denoising, CLAHE contrast enhancement,
Canny edge detection, morphology, and image quality assessment.
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

import sys
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from modules.preprocessing import PreprocessingPipeline


class TestPreprocessingPipeline(unittest.TestCase):
    """Verifies each stage of the digital image preprocessing transformations."""

    def setUp(self):
        self.preprocessor = PreprocessingPipeline(
            blur_kernel=9,
            clahe_clip=2.0,
            canny_low=40,
            canny_high=120
        )
        # Create synthetic test BGR image (200x200 with white circle on dark background)
        self.test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(self.test_img, (100, 100), 50, (200, 200, 200), -1)

    def test_pipeline_stages_generation(self):
        """Ensures all intermediate visualization stages are properly generated."""
        res = self.preprocessor.process(self.test_img, apply_enhancement=True)
        stages = res["stages"]

        expected_stages = [
            "1_original",
            "2_grayscale",
            "3_denoised",
            "4_contrast_enhanced",
            "5_thresholded",
            "6_edges",
            "7_morphology"
        ]

        for s in expected_stages:
            self.assertIn(s, stages, f"Missing expected stage '{s}' in preprocessing output")
            stage_data = stages[s]
            self.assertIn("image", stage_data)
            self.assertIn("algorithm", stage_data)
            self.assertIn("purpose", stage_data)
            self.assertIn("explanation", stage_data)
            self.assertIn("parameters", stage_data)

    def test_grayscale_dimensions(self):
        """Verifies grayscale image dimensionality reduction to single channel."""
        res = self.preprocessor.process(self.test_img)
        gray = res["gray"]
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (200, 200))

    def test_edge_detection(self):
        """Verifies Canny edge detection extracts circular boundaries."""
        res = self.preprocessor.process(self.test_img)
        edges = res["edges"]
        edge_pixel_count = np.sum(edges > 0)
        self.assertGreater(edge_pixel_count, 100, "Canny edge detector should extract circle boundary pixels")

    def test_blur_assessment(self):
        """Verifies variance of Laplacian identifies blurred vs sharp images."""
        sharp_gray = cv2.cvtColor(self.test_img, cv2.COLOR_BGR2GRAY)
        sharp_metrics = self.preprocessor.assess_image_quality(sharp_gray)
        
        # Heavy blur
        blurry_gray = cv2.GaussianBlur(sharp_gray, (35, 35), 10)
        blurry_metrics = self.preprocessor.assess_image_quality(blurry_gray)

        self.assertGreater(sharp_metrics["blur_score"], blurry_metrics["blur_score"])
        self.assertTrue(blurry_metrics["is_blurry"])


if __name__ == "__main__":
    unittest.main()
