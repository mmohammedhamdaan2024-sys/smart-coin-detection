"""
Coin segmentation module for Smart Coin Detection System.
Implements Euclidean Distance Transform, Marker-Controlled Watershed Segmentation,
and touching/overlapping coin separation and topological analysis.
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


class CoinSegmenter:
    """Separates isolated, touching, and partially overlapping coins using Watershed and Distance Transforms."""

    def __init__(self, distance_thresh_ratio: float = 0.42):
        self.distance_thresh_ratio = distance_thresh_ratio

    def segment_coins(
        self,
        bgr_image: np.ndarray,
        binary_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Executes distance-transform and marker-controlled watershed segmentation.
        Identifies separated regions, detects touching/overlapping clusters, and builds visual segmentation maps.
        """
        h, w = binary_mask.shape[:2]

        # 1. Morphological noise cleanup on binary mask
        kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_3, iterations=2)
        sure_bg = cv2.dilate(opening, kernel_7, iterations=3)

        # 2. Euclidean Distance Transform (L2 metric)
        # Calculates Euclidean distance of every foreground pixel to the nearest zero pixel
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()

        # Normalized distance map for stage visualization (0-255 uint8)
        if max_dist > 0:
            dist_vis = np.uint8((dist_transform / max_dist) * 255)
        else:
            dist_vis = np.zeros_like(opening, dtype=np.uint8)

        # 3. Marker Extraction via Thresholding the Distance Transform
        # Peaks represent the core centers of coins even when their outer rims touch
        thresh_level = self.distance_thresh_ratio * max_dist if max_dist > 0 else 0
        _, sure_fg = cv2.threshold(dist_transform, thresh_level, 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Unknown boundary zone
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 4. Connected Components on Sure Foreground
        num_labels, markers = cv2.connectedComponents(sure_fg)

        # Add 1 to all labels so that background is 1 instead of 0
        markers = markers + 1
        # Mark the unknown transition boundary as 0 for Watershed to flood
        markers[unknown == 255] = 0

        # 5. Marker-Controlled Watershed Algorithm
        # Operates on a 3-channel BGR image
        watershed_markers = markers.copy()
        cv2.watershed(bgr_image, watershed_markers)

        # 6. Generate Color-coded Segmentation Visualization
        # Background is 1, boundaries are -1, individual coins are 2, 3, ...
        seg_vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Deterministic pseudorandom distinct colors for segments
        np.random.seed(42)
        colors = np.random.randint(40, 240, size=(max(num_labels + 2, 100), 3), dtype=np.uint8)
        colors[1] = [20, 20, 20]  # Background dark slate

        unique_labels = np.unique(watershed_markers)
        for label in unique_labels:
            if label == -1:
                continue  # Ridge line
            seg_vis[watershed_markers == label] = colors[label % len(colors)]

        # Highlight watershed division boundaries in bright yellow
        seg_vis[watershed_markers == -1] = [0, 255, 255]

        # 7. Extract Individual Mask Contours from Watershed Segments
        extracted_segments = []
        for label in range(2, num_labels + 1):
            mask_label = np.uint8(watershed_markers == label)
            cnts, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            
            # Largest contour for this segment
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area < 100:  # Ignore tiny noise artifacts
                continue

            # Minimum enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            
            extracted_segments.append({
                "label": int(label),
                "contour": c,
                "area": float(area),
                "center": (float(cx), float(cy)),
                "radius": float(radius)
            })

        # 8. Overlap and Touching Analysis
        touching_pairs = []
        overlapping_clusters = set()

        for i in range(len(extracted_segments)):
            seg_a = extracted_segments[i]
            ca_x, ca_y = seg_a["center"]
            ra = seg_a["radius"]

            for j in range(i + 1, len(extracted_segments)):
                seg_b = extracted_segments[j]
                cb_x, cb_y = seg_b["center"]
                rb = seg_b["radius"]

                dist = np.hypot(ca_x - cb_x, ca_y - cb_y)
                sum_radii = ra + rb

                # If distance between centers is less than the sum of radii, they touch or overlap
                if dist < sum_radii:
                    overlap_amount = sum_radii - dist
                    ratio = overlap_amount / min(ra, rb)
                    touching_pairs.append({
                        "coin_a": seg_a["label"],
                        "coin_b": seg_b["label"],
                        "distance": round(float(dist), 2),
                        "overlap_ratio": round(float(ratio), 2),
                        "is_severe_overlap": ratio > 0.35
                    })
                    overlapping_clusters.add(seg_a["label"])
                    overlapping_clusters.add(seg_b["label"])

        return {
            "num_regions_found": len(extracted_segments),
            "watershed_markers": watershed_markers,
            "dist_transform": dist_transform,
            "dist_vis": dist_vis,
            "segmentation_vis": seg_vis,
            "segments": extracted_segments,
            "touching_pairs": touching_pairs,
            "touching_labels": list(overlapping_clusters),
            "has_touching_coins": len(touching_pairs) > 0
        }
