"""
Visual condition and surface anomaly screening module for Smart Coin Detection System.
Analyzes surface discoloration, scratches, oxidation spots, edge irregularities,
and generates annotated defect masks with formal numismatic screening disclaimers.
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


class ConditionAnalyzer:
    """Automated visual screening tool for surface anomalies and wear indicators."""

    DISCLAIMER = (
        "Notice: Image-based condition analysis is an automated visual screening tool, "
        "not a professional coin-grading or authentication service. Real numismatic grading "
        "requires microscopic physical inspection, luster evaluation, and metallurgical assay."
    )

    def analyze_condition(
        self,
        coin_crop_bgr: np.ndarray,
        image_quality_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Screens coin surface for visual scratches, spots, discoloration, and rim irregularities.
        Produces quantified defect metrics and an annotated overlay image.
        """
        if coin_crop_bgr is None or coin_crop_bgr.size == 0:
            return {
                "condition_category": "Unable to assess reliably",
                "anomaly_count": 0,
                "anomaly_area_pct": 0.0,
                "annotated_crop": None,
                "confidence_score": 0.0,
                "screening_summary": "Empty or corrupted coin crop.",
                "disclaimer": self.DISCLAIMER
            }

        h, w = coin_crop_bgr.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 2

        # 1. Quality Check - Verify if blur or extreme glare prevents reliable analysis
        is_blurry = image_quality_metrics.get("is_blurry", False)
        has_glare = image_quality_metrics.get("has_harsh_glare", False)

        if is_blurry or radius < 25:
            annotated = coin_crop_bgr.copy()
            cv2.putText(annotated, "QUALITY LIMITED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return {
                "condition_category": "Unable to assess reliably",
                "anomaly_count": 0,
                "anomaly_area_pct": 0.0,
                "annotated_crop": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                "confidence_score": 30.0,
                "screening_summary": "Image resolution is too low or optical blur prevents surface feature assessment.",
                "has_edge_irregularity": False,
                "disclaimer": self.DISCLAIMER
            }

        # 2. Masking the Coin Interior (Excluding outer background and peripheral rim)
        coin_interior_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(coin_interior_mask, (center_x, center_y), int(radius * 0.82), 255, -1)
        interior_pixel_count = np.sum(coin_interior_mask == 255)

        if interior_pixel_count == 0:
            interior_pixel_count = 1

        # 3. Surface Discoloration and Dark Spots (Corrosion-like patches)
        lab = cv2.cvtColor(coin_crop_bgr, cv2.COLOR_BGR2LAB)
        interior_lab = lab[coin_interior_mask == 255]
        
        median_l = np.median(interior_lab[:, 0])
        median_a = np.median(interior_lab[:, 1])
        median_b = np.median(interior_lab[:, 2])

        # Color deviation delta E from median coin metal tone
        diff_l = lab[:, :, 0].astype(np.float32) - median_l
        diff_a = lab[:, :, 1].astype(np.float32) - median_a
        diff_b = lab[:, :, 2].astype(np.float32) - median_b
        delta_e = np.sqrt(diff_l**2 + diff_a**2 + diff_b**2)

        # Detect severe discoloration / oxidation spots
        discoloration_mask = (delta_e > 26.0) & (coin_interior_mask == 255)
        # Also detect harsh dark spots (possible pits / heavy tarnish)
        dark_spots_mask = (lab[:, :, 0] < (median_l * 0.45)) & (coin_interior_mask == 255)

        # 4. Scratches and High-Frequency Linear Abrasions
        gray = cv2.cvtColor(coin_crop_bgr, cv2.COLOR_BGR2GRAY)
        # Morphological black-hat highlights narrow dark scratches
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_line)
        _, scratch_mask = cv2.threshold(blackhat, 22, 255, cv2.THRESH_BINARY)
        scratch_mask = cv2.bitwise_and(scratch_mask, coin_interior_mask)

        # Combine candidate anomaly masks
        combined_anomaly_mask = np.zeros((h, w), dtype=np.uint8)
        combined_anomaly_mask[discoloration_mask] = 255
        combined_anomaly_mask[dark_spots_mask] = 255
        combined_anomaly_mask[scratch_mask == 255] = 255

        # Noise filter: clean isolated single pixels
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_anomaly_mask = cv2.morphologyEx(combined_anomaly_mask, cv2.MORPH_OPEN, kernel_clean)

        # 5. Connected Component Counting of Anomaly Regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_anomaly_mask)
        
        # Count significant anomaly clusters (ignore clusters smaller than 6 pixels)
        valid_anomalies = 0
        significant_anomaly_mask = np.zeros_like(combined_anomaly_mask)
        for i in range(1, num_labels):
            cluster_area = stats[i, cv2.CC_STAT_AREA]
            if cluster_area >= 6:
                valid_anomalies += 1
                significant_anomaly_mask[labels == i] = 255

        anomaly_pixels = np.sum(significant_anomaly_mask == 255)
        anomaly_area_pct = round(float((anomaly_pixels / interior_pixel_count) * 100), 2)

        # 6. Edge / Rim Irregularity Check
        # Detect if outer perimeter noticeably deviates from circularity
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray_blur, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_edge_irregularity = False
        if cnts:
            max_c = max(cnts, key=cv2.contourArea)
            if len(max_c) > 20:
                dists = [np.hypot(pt[0][0] - center_x, pt[0][1] - center_y) for pt in max_c]
                max_dev = max(abs(d - radius) for d in dists)
                if max_dev > (radius * 0.18):
                    has_edge_irregularity = True

        # 7. Category Assignment
        if anomaly_area_pct < 1.8 and valid_anomalies <= 2 and not has_edge_irregularity:
            category = "No obvious visible anomaly detected"
            confidence = 88.0
            summary = "Coin surface shows uniform coloration and clean visible field with no severe blemishes."
        elif anomaly_area_pct < 5.0 and not has_edge_irregularity:
            category = "Minor visible surface irregularities"
            confidence = 82.0
            summary = f"Detected {valid_anomalies} minor surface irregularities covering {anomaly_area_pct}% of the visible field (e.g., hairline handling marks or mild toning)."
        elif anomaly_area_pct < 12.0 or (has_edge_irregularity and anomaly_area_pct < 6.0):
            category = "Moderate visible irregularities"
            confidence = 78.0
            summary = f"Observed {valid_anomalies} noticeable blemishes covering {anomaly_area_pct}% of field{' with possible rim nick' if has_edge_irregularity else ''}."
        else:
            category = "Significant visible irregularities"
            confidence = 85.0
            summary = f"Surface exhibits significant visible wear, staining, or corrosion-like deposits covering {anomaly_area_pct}% of the inspected area."

        # 8. Create Visual Overlay (Red/Amber highlighting on original crop)
        overlay = coin_crop_bgr.copy()
        # Red tint on anomalies
        overlay[significant_anomaly_mask == 255] = [0, 60, 255]
        # Alpha blend: 65% original + 35% highlight
        annotated_bgr = cv2.addWeighted(overlay, 0.40, coin_crop_bgr, 0.60, 0)
        
        # Draw rim perimeter guideline in cyan
        cv2.circle(annotated_bgr, (center_x, center_y), int(radius * 0.82), (255, 255, 0), 1)

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        return {
            "condition_category": category,
            "anomaly_count": valid_anomalies,
            "anomaly_area_pct": anomaly_area_pct,
            "has_edge_irregularity": has_edge_irregularity,
            "annotated_crop": annotated_rgb,
            "anomaly_mask": significant_anomaly_mask,
            "confidence_score": confidence,
            "screening_summary": summary,
            "disclaimer": self.DISCLAIMER
        }
