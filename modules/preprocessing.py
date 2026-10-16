"""
Digital image preprocessing pipeline for Smart Coin Detection System.
Implements grayscale conversion, denoising, CLAHE contrast enhancement,
thresholding, Canny edge detection, and morphological operations.
Captures all intermediate stages for step-by-step visualization.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np

from config import (
    DEFAULT_BLUR_KERNEL_SIZE,
    DEFAULT_CLAHE_CLIP_LIMIT,
    DEFAULT_CLAHE_GRID_SIZE,
    DEFAULT_CANNY_LOW,
    DEFAULT_CANNY_HIGH
)


class PreprocessingPipeline:
    """Applies authentic digital image processing techniques with stage-by-stage auditing."""

    def __init__(
        self,
        blur_kernel: int = DEFAULT_BLUR_KERNEL_SIZE,
        clahe_clip: float = DEFAULT_CLAHE_CLIP_LIMIT,
        clahe_grid: Tuple[int, int] = DEFAULT_CLAHE_GRID_SIZE,
        canny_low: int = DEFAULT_CANNY_LOW,
        canny_high: int = DEFAULT_CANNY_HIGH
    ):
        # Kernel size must be an odd positive integer
        self.blur_kernel = blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.canny_low = canny_low
        self.canny_high = canny_high

    def assess_image_quality(self, gray_img: np.ndarray) -> Dict[str, Any]:
        """
        Calculates objective quality metrics:
        - Sharpness via variance of the Laplacian.
        - Lighting uniformity and overexposure / underexposure percentages.
        """
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        blur_score = float(laplacian.var())

        overexposed_pct = float(np.sum(gray_img >= 250) / gray_img.size * 100)
        underexposed_pct = float(np.sum(gray_img <= 10) / gray_img.size * 100)
        mean_brightness = float(np.mean(gray_img))

        is_blurry = blur_score < 70.0
        has_harsh_glare = overexposed_pct > 8.0

        return {
            "blur_score": round(blur_score, 2),
            "is_blurry": is_blurry,
            "mean_brightness": round(mean_brightness, 2),
            "overexposed_pct": round(overexposed_pct, 2),
            "underexposed_pct": round(underexposed_pct, 2),
            "has_harsh_glare": has_harsh_glare,
            "quality_rating": "Poor (Blurry/Glare)" if (is_blurry or has_harsh_glare) else "Good"
        }

    def process(
        self,
        bgr_image: np.ndarray,
        apply_enhancement: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the step-by-step preprocessing pipeline.
        Returns a dictionary containing all transformed image arrays and documented metadata.
        """
        stages = {}

        # Stage 1: Original Image (RGB for display)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        stages["1_original"] = {
            "title": "Stage 1: Original Input Image",
            "image": rgb_image,
            "is_gray": False,
            "algorithm": "RGB Color Space Ingestion",
            "purpose": "Provides the baseline photographic representation of coins and background.",
            "explanation": "Standard 3-channel optical image preserving full spectral information (Red, Green, Blue).",
            "parameters": f"Dimensions: {bgr_image.shape[1]}x{bgr_image.shape[0]} px, Channels: 3"
        }

        # Stage 2: Grayscale Conversion
        # CIE standard: Y = 0.299*R + 0.587*G + 0.114*B
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        stages["2_grayscale"] = {
            "title": "Stage 2: Grayscale Conversion",
            "image": gray_image,
            "is_gray": True,
            "algorithm": "CIE Luminance Weighting (cv2.COLOR_BGR2GRAY)",
            "purpose": "Reduces dimensionality from 3 channels to 1 intensity channel while preserving spatial gradients.",
            "explanation": "Human perception is most sensitive to green luminance. Grayscale simplifies edge detection and circle fitting by eliminating hue distractions.",
            "parameters": "Formula: Y = 0.299R + 0.587G + 0.114B"
        }

        # Quality check
        quality_metrics = self.assess_image_quality(gray_image)

        # Stage 3: Denoised Image
        # Edge-preserving Gaussian smoothing or Median filtering
        denoised_image = cv2.GaussianBlur(
            gray_image,
            (self.blur_kernel, self.blur_kernel),
            sigmaX=1.5,
            sigmaY=1.5
        )
        stages["3_denoised"] = {
            "title": "Stage 3: Denoised Image",
            "image": denoised_image,
            "is_gray": True,
            "algorithm": f"2D Gaussian Blur (Kernel: {self.blur_kernel}x{self.blur_kernel}, σ=1.5)",
            "purpose": "Suppresses high-frequency camera sensor noise and surface micro-speckle without shifting coin boundary coordinates.",
            "explanation": "Convolves the image with a 2D Gaussian kernel bell curve. Eliminates false edge gradients while retaining circular boundaries.",
            "parameters": f"Kernel Size: ({self.blur_kernel}, {self.blur_kernel}), SigmaX: 1.5, SigmaY: 1.5"
        }

        # Stage 4: Contrast-Enhanced Image (CLAHE)
        if apply_enhancement:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=self.clahe_grid
            )
            contrast_enhanced = clahe.apply(denoised_image)
            enhancement_status = "Enabled"
        else:
            contrast_enhanced = denoised_image.copy()
            enhancement_status = "Bypassed"

        stages["4_contrast_enhanced"] = {
            "title": "Stage 4: Contrast-Enhanced Image",
            "image": contrast_enhanced,
            "is_gray": True,
            "algorithm": f"CLAHE (Contrast Limited Adaptive Histogram Equalization) [{enhancement_status}]",
            "purpose": "Equalizes local luminance contrast to expose subtle coin rims against low-contrast or shadowed backgrounds.",
            "explanation": "Divides the image into contextual tiles (8x8) and computes local histograms. Clips excessive peak amplification to prevent noise over-saturation.",
            "parameters": f"Clip Limit: {self.clahe_clip}, Tile Grid: {self.clahe_grid}"
        }

        # Stage 5: Thresholded Image
        # Adaptive Gaussian Thresholding combined with Otsu thresholding
        otsu_val, otsu_thresh = cv2.threshold(
            contrast_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        adaptive_thresh = cv2.adaptiveThreshold(
            contrast_enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=4
        )
        # Combine Otsu and Adaptive to ensure robust foreground extraction
        combined_threshold = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
        stages["5_thresholded"] = {
            "title": "Stage 5: Binarized / Thresholded Image",
            "image": combined_threshold,
            "is_gray": True,
            "algorithm": f"Hybrid Otsu's Global (T={int(otsu_val)}) + Adaptive Gaussian (Block=21, C=4)",
            "purpose": "Segments potential coin candidate regions (foreground white pixels) from the image backdrop (background black pixels).",
            "explanation": "Otsu minimizes intra-class intensity variance, while Adaptive Gaussian calculates dynamic thresholds locally across uneven illumination gradients.",
            "parameters": f"Otsu Optimal T: {int(otsu_val)}, Adaptive Neighborhood: 21px, Constant C: 4"
        }

        # Stage 6: Edge Detection Result (Canny)
        edges = cv2.Canny(
            contrast_enhanced,
            threshold1=self.canny_low,
            threshold2=self.canny_high,
            apertureSize=3,
            L2gradient=True
        )
        stages["6_edges"] = {
            "title": "Stage 6: Canny Edge Detection",
            "image": edges,
            "is_gray": True,
            "algorithm": f"Canny Edge Detector with L2 Gradient Norm ({self.canny_low}, {self.canny_high})",
            "purpose": "Extracts sharp structural 1-pixel boundary contours corresponding to coin perimeters.",
            "explanation": "Computes directional intensity gradients, applies non-maximum suppression (thinning), and performs dual-threshold hysteresis tracking.",
            "parameters": f"Low Threshold: {self.canny_low}, High Threshold: {self.canny_high}, L2 Norm: True"
        }

        # Stage 7: Morphological Processing Result
        # Elliptical structuring element matches circular coin geometry
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph_closed = cv2.morphologyEx(combined_threshold, cv2.MORPH_CLOSE, kernel_morph, iterations=2)
        morph_opened = cv2.morphologyEx(morph_closed, cv2.MORPH_OPEN, kernel_morph, iterations=1)
        stages["7_morphology"] = {
            "title": "Stage 7: Morphological Filtering Result",
            "image": morph_opened,
            "is_gray": True,
            "algorithm": "Morphological Closing followed by Opening with Elliptical Structuring Element",
            "purpose": "Seals small boundary gaps inside coin silhouettes and eliminates isolated pixel noise flecks.",
            "explanation": "Closing (dilation then erosion) connects disconnected boundary segments; Opening (erosion then dilation) removes background debris.",
            "parameters": "Kernel: 5x5 Ellipse, Closing Iterations: 2, Opening Iterations: 1"
        }

        return {
            "stages": stages,
            "gray": gray_image,
            "denoised": denoised_image,
            "enhanced": contrast_enhanced,
            "edges": edges,
            "thresholded": combined_threshold,
            "morph_cleaned": morph_opened,
            "quality_metrics": quality_metrics
        }
