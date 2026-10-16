"""
Coin classification module for Smart Coin Detection System.
Analyzes color signatures, radial bimetallic composition, texture profiles,
and calibrated physical diameter to classify coins against the reference database.
"""

from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from .reference_database import ReferenceDatabase


class CoinClassifier:
    """Classifies detected coins using multi-modal visual features and reference matching."""

    def __init__(self, db: Optional[ReferenceDatabase] = None):
        self.db = db or ReferenceDatabase()

    def analyze_color_and_composition(self, coin_crop_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes the coin's chromatic signature and checks for bimetallic ring/core separation.
        """
        if coin_crop_bgr is None or coin_crop_bgr.size == 0:
            return {
                "detected_color_class": "unknown",
                "is_bimetallic": False,
                "mean_saturation": 0.0,
                "mean_hue": 0.0,
                "mean_brightness": 0.0
            }

        h, w = coin_crop_bgr.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 2

        # Convert to HSV and LAB
        hsv = cv2.cvtColor(coin_crop_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(coin_crop_bgr, cv2.COLOR_BGR2LAB)

        # Create circular masks for inner core (0 to 0.55 R) and outer ring (0.70 to 0.92 R)
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        full_coin_mask = np.zeros((h, w), dtype=np.uint8)

        cv2.circle(inner_mask, center, int(radius * 0.50), 255, -1)
        cv2.circle(outer_mask, center, int(radius * 0.90), 255, -1)
        inner_annulus = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(inner_annulus, center, int(radius * 0.65), 255, -1)
        outer_mask = cv2.subtract(outer_mask, inner_annulus)

        cv2.circle(full_coin_mask, center, int(radius * 0.88), 255, -1)

        # Full coin chromatic statistics
        coin_hsv_pixels = hsv[full_coin_mask == 255]
        if len(coin_hsv_pixels) == 0:
            return {"detected_color_class": "unknown", "is_bimetallic": False}

        mean_h = float(np.mean(coin_hsv_pixels[:, 0]))
        mean_s = float(np.mean(coin_hsv_pixels[:, 1]))
        mean_v = float(np.mean(coin_hsv_pixels[:, 2]))

        # Compare inner vs outer ring in LAB color space to detect bimetallic coins (e.g. 1€, 2€, ₹10, £1)
        inner_lab_pixels = lab[inner_mask == 255]
        outer_lab_pixels = lab[outer_mask == 255]

        is_bimetallic = False
        if len(inner_lab_pixels) > 50 and len(outer_lab_pixels) > 50:
            inner_b = np.mean(inner_lab_pixels[:, 2])  # b* channel: yellow/blue
            outer_b = np.mean(outer_lab_pixels[:, 2])
            delta_b = abs(float(inner_b - outer_b))
            
            inner_l = np.mean(inner_lab_pixels[:, 0])
            outer_l = np.mean(outer_lab_pixels[:, 0])
            delta_l = abs(float(inner_l - outer_l))

            # Bimetallic coins exhibit a noticeable color difference between core and ring
            if delta_b > 12.0 or (delta_b > 8.0 and delta_l > 15.0):
                is_bimetallic = True

        # Classify color family
        if is_bimetallic:
            color_class = "bimetallic"
        elif mean_s < 38.0:
            # Low saturation indicates silver/cupronickel/stainless steel
            color_class = "silver_nickel"
        elif (mean_h <= 18 or mean_h >= 165) and mean_s >= 38.0:
            # Reddish-brown / copper tone
            color_class = "copper_red"
        elif 18 < mean_h <= 45 and mean_s >= 38.0:
            # Golden brass / Nordic gold tone
            color_class = "brass_gold"
        else:
            color_class = "silver_nickel"  # Default fallback for neutral oxidized metals

        return {
            "detected_color_class": color_class,
            "is_bimetallic": is_bimetallic,
            "mean_hue": round(mean_h, 1),
            "mean_saturation": round(mean_s, 1),
            "mean_brightness": round(mean_v, 1)
        }

    def classify_coin(
        self,
        coin_crop_bgr: np.ndarray,
        diameter_mm: Optional[float] = None,
        preferred_currency: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classifies coin against reference catalog.
        Reports exact denomination when evidence is sufficient; otherwise reports 'Coin detected — denomination uncertain'.
        """
        color_info = self.analyze_color_and_composition(coin_crop_bgr)
        color_class = color_info["detected_color_class"]

        result = {
            "is_classified": False,
            "reference_id": None,
            "currency": None,
            "currency_symbol": "",
            "country": None,
            "denomination": "Coin detected — denomination uncertain",
            "face_value": None,
            "classification_confidence": 0.0,
            "classification_method": "Visual Color & Dimension Analysis",
            "detected_color_class": color_class,
            "is_bimetallic": color_info["is_bimetallic"],
            "top_candidates": [],
            "notes": ""
        }

        # Case 1: Uncalibrated (No physical diameter in mm available)
        if diameter_mm is None or diameter_mm <= 0:
            # Search DB for coins matching color class
            all_coins = self.db.get_all_coins()
            color_candidates = []
            for c in all_coins:
                if preferred_currency and preferred_currency != "ALL" and c.get("currency") != preferred_currency:
                    continue
                if c.get("color_class") == color_class:
                    color_candidates.append({
                        "reference_id": c["reference_id"],
                        "denomination": c["denomination"],
                        "currency": c["currency"],
                        "country": c["country"],
                        "face_value": c["face_value"],
                        "reference_diameter_mm": c["reference_diameter_mm"],
                        "confidence_pct": 45.0,
                        "match_reason": f"Matches observed metal class ({color_class.replace('_', ' ')})"
                    })

            result["denomination"] = "Coin detected — denomination uncertain"
            result["classification_confidence"] = 0.0
            result["top_candidates"] = color_candidates[:4]
            result["notes"] = "Scale uncalibrated. Optical diameter alone without calibration cannot reliably distinguish denominations. Provide a reference object to establish physical scale."
            return result

        # Case 2: Calibrated (Physical diameter is known)
        # Search for candidates within diameter tolerance (+- 1.2 mm)
        candidates = self.db.find_candidates_by_diameter(
            measured_mm=diameter_mm,
            tolerance_mm=1.5,
            currency_filter=preferred_currency
        )

        if not candidates:
            result["denomination"] = "Coin detected — denomination uncertain"
            result["classification_confidence"] = 0.0
            result["notes"] = f"Measured diameter ({diameter_mm:.1f} mm) does not match any coin in reference database within 1.5mm tolerance."
            return result

        # Score candidates combining diameter proximity + color match
        scored_candidates = []
        for cand in candidates:
            score = cand["diameter_similarity_pct"] * 0.70  # 70% weight on diameter
            
            # 30% weight on color class alignment
            cand_color = cand.get("color_class", "")
            if cand_color == color_class:
                score += 30.0
            elif (cand_color == "silver_nickel" and color_class == "brass_gold") or \
                 (cand_color == "copper_red" and color_class == "brass_gold"):
                score += 10.0  # Tarnish or lighting artifact partial credit
            else:
                score += 5.0

            scored_candidates.append({
                "candidate": cand,
                "total_score": round(min(score, 98.0), 1)
            })

        scored_candidates.sort(key=lambda x: x["total_score"], reverse=True)
        top_match = scored_candidates[0]
        top_coin = top_match["candidate"]
        top_score = top_match["total_score"]

        # Check if top candidate is clearly distinguishable from runner up
        runner_up_score = scored_candidates[1]["total_score"] if len(scored_candidates) > 1 else 0.0
        score_gap = top_score - runner_up_score

        formatted_candidates = [
            {
                "reference_id": sc["candidate"]["reference_id"],
                "denomination": sc["candidate"]["denomination"],
                "currency": sc["candidate"]["currency"],
                "country": sc["candidate"]["country"],
                "face_value": sc["candidate"]["face_value"],
                "reference_diameter_mm": sc["candidate"]["reference_diameter_mm"],
                "confidence_pct": sc["total_score"],
                "match_reason": f"Diameter Δ={sc['candidate']['diameter_delta_mm']}mm, Metal: {sc['candidate']['color_class']}"
            }
            for sc in scored_candidates[:4]
        ]
        result["top_candidates"] = formatted_candidates

        # If top match is strong and distinct
        if top_score >= 70.0 and (score_gap >= 10.0 or len(scored_candidates) == 1 or preferred_currency not in [None, "ALL"]):
            result["is_classified"] = True
            result["reference_id"] = top_coin["reference_id"]
            result["currency"] = top_coin["currency"]
            result["currency_symbol"] = top_coin.get("currency_symbol", "")
            result["country"] = top_coin["country"]
            result["denomination"] = top_coin["denomination"]
            result["face_value"] = top_coin["face_value"]
            result["classification_confidence"] = top_score
            result["notes"] = f"Classified with {top_score}% confidence based on {diameter_mm:.1f} mm diameter and {color_class.replace('_', ' ')} visual profile."
        else:
            # Ambiguous (e.g. US Quarter 24.26mm vs EUR 50c 24.25mm without preferred currency)
            result["denomination"] = "Coin detected — denomination uncertain"
            result["classification_confidence"] = top_score
            result["notes"] = f"Ambiguous match between {top_coin['denomination']} and {scored_candidates[1]['candidate']['denomination'] if len(scored_candidates) > 1 else 'other types'}. Select a specific currency filter in settings for definitive classification."

        return result
