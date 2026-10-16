"""
Unit tests for Report Generation and History Database.
Validates CSV export formatting, JSON structure, PDF compilation, and SQLite transactions.
"""

import unittest
from pathlib import Path
import json
import numpy as np

import sys
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from modules.report_generator import ReportGenerator
from modules.history_manager import HistoryManager


class TestReportsAndHistory(unittest.TestCase):
    """Verifies report generation integrity and history database persistence."""

    def setUp(self):
        self.mock_coins = [
            {
                "id": 1,
                "center_x": 400.0,
                "center_y": 300.0,
                "radius_px": 80.0,
                "diameter_px": 160.0,
                "diameter_mm": 24.26,
                "detection_confidence": 92.5,
                "detection_status": "Cleanly Detected",
                "is_touching": False,
                "geometry": {
                    "diameter_px": 160.0,
                    "diameter_mm": 24.26,
                    "contour_area_px2": 20106.2,
                    "perimeter_px": 502.6,
                    "circularity": 0.995,
                    "aspect_ratio": 1.0,
                    "physical_status": "Calibrated"
                },
                "classification": {
                    "is_classified": True,
                    "denomination": "25 Cents (Quarter)",
                    "currency": "USD",
                    "currency_symbol": "$",
                    "country": "United States",
                    "face_value": 0.25,
                    "classification_confidence": 95.0,
                    "detected_color_class": "silver_nickel",
                    "is_bimetallic": False,
                    "notes": "Positive match"
                },
                "condition": {
                    "condition_category": "No obvious visible anomaly detected",
                    "anomaly_count": 0,
                    "anomaly_area_pct": 0.0,
                    "has_edge_irregularity": False,
                    "confidence_score": 88.0,
                    "screening_summary": "Clean surface",
                    "disclaimer": "Screening tool only."
                },
                "weight": {
                    "weight_status": "Reference Standard",
                    "actual_weight_g": None,
                    "reference_mass_g": 5.67,
                    "display_text": "Reference Mass: 5.67 g"
                },
                "age": {
                    "minting_year": None,
                    "estimated_age_years": None,
                    "issue_period": "1999-Present",
                    "age_status": "Issue Range: 1999-Present"
                }
            }
        ]

        self.mock_analysis = {
            "coins": self.mock_coins,
            "total_detected": 1,
            "summary_stats": {
                "total_coins": 1,
                "classified_coins": 1,
                "total_monetary_value": 0.25,
                "monetary_currency": "USD",
                "avg_diameter_px": 160.0,
                "anomaly_coins_count": 0,
                "touching_coins_count": 0,
                "is_monetary_partial": False
            },
            "quality_metrics": {
                "blur_score": 150.0,
                "quality_rating": "Good",
                "is_blurry": False,
                "overexposed_pct": 0.0
            }
        }

        self.mock_meta = {
            "file_name": "test_coin.jpg",
            "width": 800,
            "height": 600,
            "channels": 3,
            "scale_factor": 1.0
        }

    def test_csv_generation(self):
        """Verifies CSV generation produces correct headers and populated rows."""
        csv_str = ReportGenerator.generate_csv(self.mock_coins)
        self.assertIn("Coin ID", csv_str)
        self.assertIn("Denomination", csv_str)
        self.assertIn("25 Cents (Quarter)", csv_str)
        self.assertIn("24.26", csv_str)

    def test_json_generation(self):
        """Verifies JSON generation conforms to expected schema."""
        json_str = ReportGenerator.generate_json(
            self.mock_analysis,
            self.mock_meta,
            {"min_diam_px": 30, "max_diam_px": 500}
        )
        parsed = json.loads(json_str)
        self.assertEqual(parsed["detection_summary"]["total_coins"], 1)
        self.assertEqual(len(parsed["coins"]), 1)
        self.assertIn("scientific_limitations", parsed)

    def test_pdf_generation(self):
        """Verifies PDF binary generation creates a valid non-empty PDF document."""
        dummy_rgb = np.zeros((300, 400, 3), dtype=np.uint8)
        pdf_bytes = ReportGenerator.generate_pdf(
            self.mock_analysis,
            self.mock_meta,
            annotated_rgb=dummy_rgb
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 500)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "PDF stream must begin with '%PDF' magic header")

    def test_history_database_lifecycle(self):
        """Verifies SQLite history insertion, retrieval, and deletion."""
        test_db_path = Path(__file__).resolve().parent.parent / "outputs" / "test_history.db"
        if test_db_path.exists():
            test_db_path.unlink()

        hm = HistoryManager(test_db_path)
        row_id = hm.save_analysis(
            filename="unit_test.jpg",
            width=800,
            height=600,
            summary_stats=self.mock_analysis["summary_stats"],
            config_parameters={"test_mode": True},
            coins_list=self.mock_coins
        )
        self.assertGreater(row_id, 0)

        df = hm.get_all_records()
        self.assertEqual(len(df), 1)

        record = hm.get_record_by_id(row_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["filename"], "unit_test.jpg")
        self.assertEqual(record["total_coins"], 1)

        deleted = hm.delete_record(row_id)
        self.assertTrue(deleted)
        self.assertEqual(len(hm.get_all_records()), 0)

        # Cleanup
        if test_db_path.exists():
            test_db_path.unlink()


if __name__ == "__main__":
    unittest.main()
