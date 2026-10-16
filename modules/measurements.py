"""
Geometric analysis and physical measurement module for Smart Coin Detection System.
Calculates area, perimeter, circularity, bounding box, radius, diameter in pixels,
and calibrated physical diameter in millimeters with perspective distortion caveats.
"""

from typing import Dict, Any, Optional, Tuple
import math
import cv2
import numpy as np


class GeometricAnalyzer:
    """Computes exact pixel geometry and converts to physical units when calibration is established."""

    def __init__(self, calibration_scale_mm_per_px: Optional[float] = None):
        """
        calibration_scale_mm_per_px: Millimeters represented by one pixel.
        If None, physical measurements remain strictly unavailable/uncalibrated.
        """
        self.scale = calibration_scale_mm_per_px

    @staticmethod
    def calculate_circularity(area: float, perimeter: float) -> float:
        """
        Calculates dimensionless circularity metric:
        C = 4 * π * Area / (Perimeter^2)
        A perfect continuous mathematical circle yields exactly 1.0.
        On discrete digital grids, values typically range from 0.80 to 1.02.
        """
        if perimeter <= 0:
            return 0.0
        c = (4.0 * math.pi * area) / (perimeter * perimeter)
        # Cap slight discrete pixel quantization overshoot to 1.0 for clarity
        return min(round(float(c), 4), 1.0)

    def analyze_geometry(
        self,
        contour: np.ndarray,
        circle_fit: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, Any]:
        """
        Computes comprehensive geometric parameters for a coin contour.
        If circle_fit is provided as (center_x, center_y, radius), blends both representations.
        """
        # 1. Contour Area & Perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        circularity = self.calculate_circularity(area, perimeter)

        # 2. Axis-Aligned Bounding Box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = round(float(w) / float(h), 3) if h > 0 else 1.0

        # 3. Minimum Enclosing Circle
        (enc_cx, enc_cy), enc_radius = cv2.minEnclosingCircle(contour)

        # 4. Fitted Ellipse / Orientation (if contour has sufficient points)
        orientation_deg = 0.0
        major_axis = float(enc_radius * 2)
        minor_axis = float(enc_radius * 2)
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (ell_cx, ell_cy), (ell_d1, ell_d2), angle = ellipse
                orientation_deg = round(float(angle), 1)
                major_axis = round(float(max(ell_d1, ell_d2)), 2)
                minor_axis = round(float(min(ell_d1, ell_d2)), 2)
            except Exception:
                pass

        # 5. Determine Center and Radius
        # Prefer circle_fit (from Circular Hough Transform) if supplied, else minEnclosingCircle
        if circle_fit is not None:
            cx, cy, radius = circle_fit
        else:
            cx, cy, radius = enc_cx, enc_cy, enc_radius

        diameter_px = round(float(radius * 2), 2)

        # 6. Physical Diameter Calculation (Millimeters)
        diameter_mm = None
        physical_status = "Uncalibrated — Requires known scale reference"
        calibration_notes = "Perspective angle perpendicularity assumed. Real-world distortion may affect accuracy."

        if self.scale is not None and self.scale > 0:
            diameter_mm = round(float(diameter_px * self.scale), 2)
            physical_status = "Calibrated"
        else:
            calibration_notes = "No valid calibration scale was supplied. Physical millimeter diameter cannot be fabricated from pixel count alone."

        return {
            "center_px": (round(float(cx), 1), round(float(cy), 1)),
            "radius_px": round(float(radius), 2),
            "diameter_px": diameter_px,
            "bbox_xywh": (int(x), int(y), int(w), int(h)),
            "contour_area_px2": round(float(area), 1),
            "perimeter_px": round(float(perimeter), 2),
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "orientation_deg": orientation_deg,
            "major_axis_px": major_axis,
            "minor_axis_px": minor_axis,
            "diameter_mm": diameter_mm,
            "physical_status": physical_status,
            "calibration_notes": calibration_notes
        }

    @staticmethod
    def derive_scale_from_reference(
        measured_diameter_px: float,
        known_diameter_mm: float
    ) -> Optional[float]:
        """
        Derives millimeter-per-pixel scale:
        Scale = Known Reference Diameter (mm) / Measured Reference Diameter (px)
        """
        if measured_diameter_px <= 0 or known_diameter_mm <= 0:
            return None
        return float(known_diameter_mm / measured_diameter_px)
