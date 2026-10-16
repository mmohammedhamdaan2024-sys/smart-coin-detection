"""
Central coin detection and analysis orchestrator for Smart Coin Detection System.
Combines Circular Hough Transform and Watershed/Contour segmentation with
Non-Maximum Suppression, geometric measurements, classification, and condition screening.
"""

from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from config import (
    DEFAULT_MIN_DIAMETER_PX,
    DEFAULT_MAX_DIAMETER_PX,
    DEFAULT_HOUGH_DP,
    DEFAULT_HOUGH_PARAM1,
    DEFAULT_HOUGH_PARAM2,
    DEFAULT_MIN_CIRCULARITY
)
from .preprocessing import PreprocessingPipeline
from .segmentation import CoinSegmenter
from .measurements import GeometricAnalyzer
from .reference_database import ReferenceDatabase
from .coin_classifier import CoinClassifier
from .condition_analyzer import ConditionAnalyzer
from .weight_analysis import WeightEstimator
from .age_analysis import AgeEstimator


class CoinDetector:
    """End-to-end coin detection, counting, segmentation, and evaluation engine."""

    def __init__(
        self,
        reference_db: Optional[ReferenceDatabase] = None,
        calibration_scale: Optional[float] = None
    ):
        self.db = reference_db or ReferenceDatabase()
        self.calibration_scale = calibration_scale
        self.preprocessor = PreprocessingPipeline()
        self.segmenter = CoinSegmenter()
        self.geometric_analyzer = GeometricAnalyzer(calibration_scale)
        self.classifier = CoinClassifier(self.db)
        self.condition_analyzer = ConditionAnalyzer()
        self.weight_estimator = WeightEstimator()
        self.age_estimator = AgeEstimator()

    def set_calibration_scale(self, scale_mm_per_px: Optional[float]):
        """Updates calibration scale dynamically."""
        self.calibration_scale = scale_mm_per_px
        self.geometric_analyzer.scale = scale_mm_per_px

    def detect_and_analyze(
        self,
        bgr_image: np.ndarray,
        min_diameter_px: int = DEFAULT_MIN_DIAMETER_PX,
        max_diameter_px: int = DEFAULT_MAX_DIAMETER_PX,
        detection_sensitivity: float = 1.0,
        enable_watershed_split: bool = True,
        preferred_currency: Optional[str] = None,
        manual_coin_data: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes the full 12-step detection and analysis pipeline.
        Returns complete detected coins catalog, intermediate stages, and aggregate statistics.
        """
        h, w = bgr_image.shape[:2]
        manual_data = manual_coin_data or {}

        # Step 1-5: Digital Preprocessing Pipeline
        prep_results = self.preprocessor.process(bgr_image, apply_enhancement=True)
        gray = prep_results["gray"]
        denoised = prep_results["denoised"]
        enhanced = prep_results["enhanced"]
        edges = prep_results["edges"]
        morph_mask = prep_results["morph_cleaned"]
        quality_metrics = prep_results["quality_metrics"]

        # Step 6: Multi-Method Candidate Boundary Detection
        min_r = max(int(min_diameter_px // 2), 10)
        max_r = max(int(max_diameter_px // 2), min_r + 5)
        candidates = []

        # Method A: Circular Hough Transform
        # Adjust Hough param2 based on user sensitivity: higher sensitivity -> lower threshold
        hough_param2 = max(24, int(DEFAULT_HOUGH_PARAM2 / max(detection_sensitivity, 0.4)))
        hough_min_dist = max(int(min_r * 1.8), 40)

        circles = cv2.HoughCircles(
            denoised,
            cv2.HOUGH_GRADIENT,
            dp=DEFAULT_HOUGH_DP,
            minDist=hough_min_dist,
            param1=DEFAULT_HOUGH_PARAM1,
            param2=hough_param2,
            minRadius=min_r,
            maxRadius=max_r
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for pt in circles[0, :]:
                cx, cy, r = int(pt[0]), int(pt[1]), int(pt[2])
                # Boundary safety check
                if 0 <= cx < w and 0 <= cy < h:
                    candidates.append({
                        "source": "hough",
                        "center": (float(cx), float(cy)),
                        "radius": float(r),
                        "score": 0.85
                    })

        # Method B: Watershed Segmentation & Distance Transform
        seg_results = self.segmenter.segment_coins(bgr_image, morph_mask)
        touching_pairs = seg_results["touching_pairs"]

        if enable_watershed_split:
            for seg in seg_results["segments"]:
                scx, scy = seg["center"]
                sr = seg["radius"]
                cnt = seg["contour"]
                area = cv2.contourArea(cnt)
                perim = cv2.arcLength(cnt, True)
                circ = (4.0 * np.pi * area) / (perim * perim) if perim > 0 else 0.0
                if circ >= DEFAULT_MIN_CIRCULARITY and min_r <= sr <= max_r and 0 <= scx < w and 0 <= scy < h:
                    candidates.append({
                        "source": "watershed",
                        "center": (float(scx), float(scy)),
                        "radius": float(sr),
                        "contour": cnt,
                        "score": 0.80
                    })

        # Step 7 & 8: Candidate Region Filtering and Non-Maximum Suppression (Duplicate Handling)
        merged_candidates = self._merge_and_suppress_duplicates(candidates, iou_thresh=0.35)

        # Sort candidate coins top-to-bottom, left-to-right for consistent numbering
        merged_candidates.sort(key=lambda c: (round(c["center"][1] / 60) * 60, c["center"][0]))

        # Step 9-11: Individual Coin Processing, Measurement, Classification, and Screening
        detected_coins = []
        coin_id = 1

        for cand in merged_candidates:
            cx, cy = cand["center"]
            r = cand["radius"]
            d_px = r * 2.0

            if not (min_diameter_px <= d_px <= max_diameter_px):
                continue

            # Extract tight bounding box with a 15% contextual safety margin
            margin = int(r * 0.15)
            x1 = max(0, int(cx - r - margin))
            y1 = max(0, int(cy - r - margin))
            x2 = min(w, int(cx + r + margin))
            y2 = min(h, int(cy + r + margin))

            coin_crop_bgr = bgr_image[y1:y2, x1:x2].copy()
            if coin_crop_bgr.size == 0 or coin_crop_bgr.shape[0] < 10 or coin_crop_bgr.shape[1] < 10:
                continue

            # Build circular mask contour if source was Hough without explicit contour
            contour = cand.get("contour")
            if contour is None:
                # Synthesize contour from fitted circle points for uniform geometric calculation
                pts = [
                    [int(cx + r * np.cos(theta)), int(cy + r * np.sin(theta))]
                    for theta in np.linspace(0, 2 * np.pi, 60, endpoint=False)
                ]
                contour = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

            # Geometry analysis
            geo_info = self.geometric_analyzer.analyze_geometry(
                contour=contour,
                circle_fit=(cx, cy, r)
            )

            # Check circularity threshold
            if geo_info["circularity"] < DEFAULT_MIN_CIRCULARITY:
                continue

            # Calculate detection confidence score based on edge gradient intensity along circle
            detection_confidence = self._calculate_detection_confidence(
                edges=edges,
                cx=cx,
                cy=cy,
                radius=r,
                circularity=geo_info["circularity"]
            )

            # Check touching / overlapping status
            is_touching, overlap_info = self._check_overlap_status(cx, cy, r, coin_id, merged_candidates)

            # Denomination classification
            class_info = self.classifier.classify_coin(
                coin_crop_bgr=coin_crop_bgr,
                diameter_mm=geo_info["diameter_mm"],
                preferred_currency=preferred_currency
            )

            ref_data = None
            if class_info["is_classified"] and class_info["reference_id"]:
                ref_data = self.db.get_coin_by_id(class_info["reference_id"])

            # Surface condition screening
            cond_info = self.condition_analyzer.analyze_condition(
                coin_crop_bgr=coin_crop_bgr,
                image_quality_metrics=quality_metrics
            )

            # User manual overrides (scale weight or confirmed mint year)
            user_override = manual_data.get(coin_id, {})
            manual_weight = user_override.get("manual_weight_g")
            manual_year = user_override.get("manual_year")

            # Weight estimation & disclaimer
            weight_info = self.weight_estimator.analyze_weight(
                is_classified=class_info["is_classified"],
                reference_data=ref_data,
                manual_weight_g=manual_weight
            )

            # Age and Mint Year analysis
            age_info = self.age_estimator.analyze_age_and_history(
                is_classified=class_info["is_classified"],
                reference_data=ref_data,
                manual_or_ocr_year=manual_year
            )

            coin_record = {
                "id": coin_id,
                "center_x": round(cx, 1),
                "center_y": round(cy, 1),
                "radius_px": round(r, 1),
                "diameter_px": geo_info["diameter_px"],
                "diameter_mm": geo_info["diameter_mm"],
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "contour": contour,
                "detection_confidence": detection_confidence,
                "detection_status": "Overlapping / Touching" if is_touching else "Cleanly Detected",
                "is_touching": is_touching,
                "overlap_notes": overlap_info,
                "crop_rgb": cv2.cvtColor(coin_crop_bgr, cv2.COLOR_BGR2RGB),
                "crop_bgr": coin_crop_bgr,
                "geometry": geo_info,
                "classification": class_info,
                "condition": cond_info,
                "weight": weight_info,
                "age": age_info,
                "reference_data": ref_data
            }

            detected_coins.append(coin_record)
            coin_id += 1

        # Step 12: Generate Visual Overlays & Pipeline Stages
        annotated_bgr = self.generate_annotated_image(bgr_image, detected_coins)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Stage 8: Coin Segmentation Result
        stages = prep_results["stages"]
        stages["8_segmentation"] = {
            "title": "Stage 8: Marker-Controlled Watershed & Distance Transform",
            "image": cv2.cvtColor(seg_results["segmentation_vis"], cv2.COLOR_BGR2RGB),
            "is_gray": False,
            "algorithm": "Euclidean Distance Transform + Marker-Controlled Watershed",
            "purpose": "Separates contacting or clustered coins along their contact ridges.",
            "explanation": "Calculates Euclidean distance peaks to identify coin core markers, then floods topological gradients to divide touching boundaries.",
            "parameters": f"Distance Threshold Ratio: 0.42, Touching Pairs Identified: {len(touching_pairs)}"
        }

        # Stage 9: Final Detection Result
        stages["9_final_detection"] = {
            "title": "Stage 9: Final Detection & Metric Overlay",
            "image": annotated_rgb,
            "is_gray": False,
            "algorithm": "Fused Hough Transform + Contour Analysis with Non-Maximum Suppression",
            "purpose": "Final visual deliverable displaying localized coin IDs, bounding rings, and status badges.",
            "explanation": "Draws verified coin boundaries, identification numbers, and diameter estimations over the original color frame.",
            "parameters": f"Total Detected: {len(detected_coins)}, Min Diameter: {min_diameter_px}px, Max: {max_diameter_px}px"
        }

        # Summary Statistics
        summary_stats = self._calculate_summary_statistics(detected_coins)

        return {
            "coins": detected_coins,
            "total_detected": len(detected_coins),
            "annotated_rgb": annotated_rgb,
            "annotated_bgr": annotated_bgr,
            "pipeline_stages": stages,
            "quality_metrics": quality_metrics,
            "summary_stats": summary_stats,
            "has_touching_coins": len(touching_pairs) > 0,
            "calibration_scale": self.calibration_scale
        }

    def _merge_and_suppress_duplicates(
        self,
        candidates: List[Dict[str, Any]],
        iou_thresh: float = 0.35
    ) -> List[Dict[str, Any]]:
        """Suppresses duplicate and concentric/nested circle detections."""
        if not candidates:
            return []

        # Sort candidates primarily by radius descending (so outer coin boundaries take precedence
        # over internal concentric rings or embossings), breaking ties by score
        candidates.sort(key=lambda c: (c["radius"], c.get("score", 0.5)), reverse=True)
        kept = []

        for cand in candidates:
            c1_x, c1_y = cand["center"]
            r1 = cand["radius"]
            is_duplicate = False

            for k in kept:
                c2_x, c2_y = k["center"]
                r2 = k["radius"]
                dist = np.hypot(c1_x - c2_x, c1_y - c2_y)

                # 1. Concentric / nested circle check:
                # If smaller circle's center is well inside the larger circle (dist < 0.82 * max(r1, r2)),
                # it is an internal relief/ring or duplicate.
                if dist < (max(r1, r2) * 0.82):
                    is_duplicate = True
                    break

                # 2. Significant overlap check (for near-equal circles):
                avg_r = (r1 + r2) / 2.0
                if dist < avg_r * 0.65:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(cand)

        return kept

    def _calculate_detection_confidence(
        self,
        edges: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        circularity: float
    ) -> float:
        """Evaluates edge coherence along the circle perimeter."""
        h, w = edges.shape[:2]
        num_samples = 36
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        edge_hits = 0

        for ang in angles:
            px = int(cx + radius * np.cos(ang))
            py = int(cy + radius * np.sin(ang))
            
            # Check a 3x3 neighborhood around expected perimeter
            if 1 <= px < w - 1 and 1 <= py < h - 1:
                patch = edges[py - 1:py + 2, px - 1:px + 2]
                if np.any(patch > 0):
                    edge_hits += 1

        edge_ratio = edge_hits / float(num_samples)
        # Weighted combination of edge support and contour circularity
        score = (edge_ratio * 0.50 + circularity * 0.50) * 100.0
        return round(float(min(max(score, 50.0), 99.0)), 1)

    def _check_overlap_status(
        self,
        cx: float,
        cy: float,
        r: float,
        current_id: int,
        all_candidates: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """Detects if coin touches or overlaps with any neighboring candidate."""
        for other in all_candidates:
            ocx, ocy = other["center"]
            orr = other["radius"]
            if ocx == cx and ocy == cy:
                continue

            dist = np.hypot(cx - ocx, cy - ocy)
            if dist < (r + orr):
                overlap_ratio = (r + orr - dist) / min(r, orr)
                if overlap_ratio > 0.10:
                    return True, f"Touching/overlapping neighbor detected (overlap ratio: {overlap_ratio:.2f})."

        return False, "Isolated, non-touching coin."

    def generate_annotated_image(
        self,
        bgr_image: np.ndarray,
        detected_coins: List[Dict[str, Any]],
        selected_coin_id: Optional[int] = None
    ) -> np.ndarray:
        """Draws visual detection boundaries, unique ID badges, and highlights selected coin."""
        annotated = bgr_image.copy()

        for coin in detected_coins:
            cid = coin["id"]
            cx = int(coin["center_x"])
            cy = int(coin["center_y"])
            r = int(coin["radius_px"])
            is_selected = (selected_coin_id is not None and cid == selected_coin_id)
            is_touching = coin.get("is_touching", False)
            is_classified = coin["classification"]["is_classified"]

            # Color scheme:
            # Selected: Vibrant Cyan (255, 255, 0) in BGR -> (0, 255, 255)
            # Touching: Amber/Red (0, 140, 255)
            # Classified: Emerald Green (0, 200, 50)
            # Uncertain: Sky Blue (235, 170, 50)
            if is_selected:
                color = (0, 255, 255)  # Yellow/Cyan highlight
                thickness = 4
            elif is_touching:
                color = (0, 140, 255)  # Orange
                thickness = 2
            elif is_classified:
                color = (50, 205, 50)  # Lime green
                thickness = 2
            else:
                color = (255, 190, 0)  # Deep cyan
                thickness = 2

            # 1. Draw outer circle
            cv2.circle(annotated, (cx, cy), r, color, thickness, lineType=cv2.LINE_AA)
            # Draw center point
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

            # 2. Draw ID Label Badge
            label_text = f"#{cid}"
            if coin["diameter_mm"] is not None:
                label_text += f" ({coin['diameter_mm']}mm)"
            else:
                label_text += f" ({int(coin['diameter_px'])}px)"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55 if not is_selected else 0.70
            font_thickness = 1 if not is_selected else 2
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            badge_x1 = cx - text_w // 2 - 4
            badge_y1 = cy - r - text_h - 8
            badge_x2 = badge_x1 + text_w + 8
            badge_y2 = badge_y1 + text_h + 6

            # Keep inside image bounds
            if badge_y1 < 0:
                badge_y1 = cy + r + 4
                badge_y2 = badge_y1 + text_h + 8

            # Background rectangle for text readability
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 24, 30), -1)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)

            # Draw text
            text_pos_y = badge_y2 - 4
            cv2.putText(
                annotated,
                label_text,
                (badge_x1 + 4, text_pos_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA
            )

        return annotated

    def _calculate_summary_statistics(
        self,
        detected_coins: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculates aggregated metrics, monetary sums, and condition distributions."""
        total = len(detected_coins)
        if total == 0:
            return {
                "total_coins": 0,
                "classified_coins": 0,
                "unclassified_coins": 0,
                "denomination_counts": {},
                "condition_counts": {},
                "avg_diameter_px": 0.0,
                "median_diameter_px": 0.0,
                "min_diameter_px": 0.0,
                "max_diameter_px": 0.0,
                "avg_circularity": 0.0,
                "total_monetary_value": 0.0,
                "monetary_currency": "N/A",
                "is_monetary_partial": False,
                "touching_coins_count": 0,
                "anomaly_coins_count": 0
            }

        diameters_px = [c["diameter_px"] for c in detected_coins]
        circularities = [c["geometry"]["circularity"] for c in detected_coins]

        classified_count = 0
        unclassified_count = 0
        denom_counts: Dict[str, int] = {}
        condition_counts: Dict[str, int] = {}
        total_value = 0.0
        currencies = set()
        anomaly_count = 0
        touching_count = 0

        for c in detected_coins:
            # Classification breakdown
            cl = c["classification"]
            if cl["is_classified"]:
                classified_count += 1
                denom = cl["denomination"]
                denom_counts[denom] = denom_counts.get(denom, 0) + 1
                if cl["face_value"] is not None:
                    total_value += cl["face_value"]
                    if cl["currency"]:
                        currencies.add(cl["currency"])
            else:
                unclassified_count += 1
                denom_counts["Uncertain Denomination"] = denom_counts.get("Uncertain Denomination", 0) + 1

            # Condition breakdown
            cond_cat = c["condition"]["condition_category"]
            condition_counts[cond_cat] = condition_counts.get(cond_cat, 0) + 1
            if cond_cat != "No obvious visible anomaly detected":
                anomaly_count += 1

            # Touching check
            if c.get("is_touching", False):
                touching_count += 1

        primary_curr = list(currencies)[0] if len(currencies) == 1 else ("Multiple" if len(currencies) > 1 else "N/A")
        is_partial = (unclassified_count > 0) or (len(currencies) > 1)

        return {
            "total_coins": total,
            "classified_coins": classified_count,
            "unclassified_coins": unclassified_count,
            "denomination_counts": denom_counts,
            "condition_counts": condition_counts,
            "avg_diameter_px": round(float(np.mean(diameters_px)), 1),
            "median_diameter_px": round(float(np.median(diameters_px)), 1),
            "min_diameter_px": round(float(np.min(diameters_px)), 1),
            "max_diameter_px": round(float(np.max(diameters_px)), 1),
            "avg_circularity": round(float(np.mean(circularities)), 3),
            "total_monetary_value": round(total_value, 2),
            "monetary_currency": primary_curr,
            "is_monetary_partial": is_partial,
            "touching_coins_count": touching_count,
            "anomaly_coins_count": anomaly_count
        }
