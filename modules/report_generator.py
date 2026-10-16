"""
Report generation module for Smart Coin Detection System.
Generates comprehensive CSV, JSON, and PDF summary inspection reports
with complete numismatic disclosures, image metrics, and limitation notes.
"""

from typing import Dict, Any, List, Optional
import io
import json
import csv
from datetime import datetime
import numpy as np
from PIL import Image

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether


class ReportGenerator:
    """Generates standard CSV, structured JSON, and professional academic PDF reports."""

    LIMITATIONS_NOTE = (
        "1. Perspective Distortion: Planar measurements assume the optical camera axis is perpendicular to the coin surface.\n"
        "2. Optical Scale: Millimeter dimensions require physical calibration with a known reference.\n"
        "3. Weight Physics: Mass cannot be derived from 2D optical images; only official mint references or scale measurements are shown.\n"
        "4. Visual Condition: Visual screening detects visible surface discoloration/scratches, not certified numismatic grades (e.g. MS-65)."
    )

    @classmethod
    def generate_csv(cls, detected_coins: List[Dict[str, Any]]) -> str:
        """Produces tabular CSV export containing one row per coin."""
        output = io.StringIO()
        fieldnames = [
            "Coin ID",
            "Detection Status",
            "Currency",
            "Country",
            "Denomination",
            "Classification Confidence",
            "Diameter Pixels",
            "Diameter Millimeters",
            "Contour Area",
            "Perimeter",
            "Circularity",
            "Condition Category",
            "Possible Anomaly Count",
            "Minting Year",
            "Estimated Age",
            "Reference Mass Grams",
            "Actual Weight Grams",
            "Measurement Source",
            "Notes"
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for c in detected_coins:
            geo = c.get("geometry", {})
            cl = c.get("classification", {})
            cond = c.get("condition", {})
            wt = c.get("weight", {})
            age = c.get("age", {})

            writer.writerow({
                "Coin ID": c.get("id"),
                "Detection Status": c.get("detection_status", "Detected"),
                "Currency": cl.get("currency") or "Unavailable",
                "Country": cl.get("country") or "Unavailable",
                "Denomination": cl.get("denomination", "Uncertain"),
                "Classification Confidence": f"{cl.get('classification_confidence', 0.0)}%" if cl.get("classification_confidence") else "N/A",
                "Diameter Pixels": geo.get("diameter_px", ""),
                "Diameter Millimeters": geo.get("diameter_mm") if geo.get("diameter_mm") is not None else "Uncalibrated",
                "Contour Area": geo.get("contour_area_px2", ""),
                "Perimeter": geo.get("perimeter_px", ""),
                "Circularity": geo.get("circularity", ""),
                "Condition Category": cond.get("condition_category", "Unassessed"),
                "Possible Anomaly Count": cond.get("anomaly_count", 0),
                "Minting Year": age.get("minting_year") if age.get("minting_year") else "Not Readable",
                "Estimated Age": age.get("estimated_age_years") if age.get("estimated_age_years") else "Unknown",
                "Reference Mass Grams": wt.get("reference_mass_g") if wt.get("reference_mass_g") else "N/A",
                "Actual Weight Grams": wt.get("actual_weight_g") if wt.get("actual_weight_g") else "Not Measured",
                "Measurement Source": geo.get("physical_status", "Pixel Count"),
                "Notes": cl.get("notes", "") or cond.get("screening_summary", "")
            })

        return output.getvalue()

    @classmethod
    def generate_json(
        cls,
        analysis_result: Dict[str, Any],
        image_metadata: Dict[str, Any],
        config_parameters: Dict[str, Any]
    ) -> str:
        """Builds structured machine-readable JSON document."""
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "system": "Smart Coin Detection, Counting, and Condition Analysis System",
            "version": "1.0.0",
            "image_metadata": {
                "file_name": image_metadata.get("file_name", "unknown"),
                "width_px": image_metadata.get("width", 0),
                "height_px": image_metadata.get("height", 0),
                "channels": image_metadata.get("channels", 3),
                "scale_factor_applied": image_metadata.get("scale_factor", 1.0)
            },
            "processing_configuration": config_parameters,
            "detection_summary": analysis_result.get("summary_stats", {}),
            "image_quality_assessment": analysis_result.get("quality_metrics", {}),
            "coins": []
        }

        for c in analysis_result.get("coins", []):
            report["coins"].append({
                "id": c.get("id"),
                "center_pixel": [c.get("center_x"), c.get("center_y")],
                "radius_pixel": c.get("radius_px"),
                "diameter_pixel": c.get("diameter_px"),
                "diameter_millimeter": c.get("diameter_mm"),
                "detection_confidence_pct": c.get("detection_confidence"),
                "is_touching_or_overlapping": c.get("is_touching"),
                "geometry": {
                    "area_px2": c["geometry"].get("contour_area_px2"),
                    "perimeter_px": c["geometry"].get("perimeter_px"),
                    "circularity_index": c["geometry"].get("circularity"),
                    "aspect_ratio": c["geometry"].get("aspect_ratio"),
                    "calibration_status": c["geometry"].get("physical_status")
                },
                "classification": {
                    "is_classified": c["classification"].get("is_classified"),
                    "denomination": c["classification"].get("denomination"),
                    "currency": c["classification"].get("currency"),
                    "country": c["classification"].get("country"),
                    "face_value": c["classification"].get("face_value"),
                    "confidence_pct": c["classification"].get("classification_confidence"),
                    "color_class": c["classification"].get("detected_color_class"),
                    "is_bimetallic": c["classification"].get("is_bimetallic")
                },
                "condition_analysis": {
                    "category": c["condition"].get("condition_category"),
                    "anomaly_regions_count": c["condition"].get("anomaly_count"),
                    "anomaly_field_pct": c["condition"].get("anomaly_area_pct"),
                    "has_edge_irregularity": c["condition"].get("has_edge_irregularity"),
                    "screening_confidence_pct": c["condition"].get("confidence_score"),
                    "disclaimer": c["condition"].get("disclaimer")
                },
                "weight_analysis": c.get("weight", {}),
                "age_analysis": c.get("age", {})
            })

        report["scientific_limitations"] = [
            "Optical 2D photography cannot measure physical coin mass.",
            "Millimeter diameters require explicit physical reference calibration.",
            "Perspective distortion alters circularity when camera is non-normal to surface plane.",
            "Condition assessment is an optical screening indicator, not numismatic grading."
        ]

        return json.dumps(report, indent=2, ensure_ascii=False)

    @classmethod
    def generate_pdf(
        cls,
        analysis_result: Dict[str, Any],
        image_metadata: Dict[str, Any],
        annotated_rgb: np.ndarray
    ) -> bytes:
        """Renders an academic project inspection summary in PDF format."""
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1E3A8A"),
            spaceAfter=6
        )
        subtitle_style = ParagraphStyle(
            "SubTitleStyle",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=14
        )
        h2_style = ParagraphStyle(
            "H2Style",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            "BodyStyle",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#1F2937")
        )
        disclaimer_style = ParagraphStyle(
            "DisclaimerStyle",
            parent=styles["Italic"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#6B7280")
        )

        story = []

        # 1. Header & Title
        story.append(Paragraph("Smart Coin Detection, Counting & Condition Analysis System", title_style))
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Target: {image_metadata.get('file_name', 'Input Image')}",
            subtitle_style
        ))
        story.append(Spacer(1, 8))

        # 2. Executive Summary Metrics Table
        stats = analysis_result.get("summary_stats", {})
        total_coins = stats.get("total_coins", 0)
        classified_coins = stats.get("classified_coins", 0)
        total_val = stats.get("total_monetary_value", 0.0)
        curr = stats.get("monetary_currency", "")
        avg_diam = stats.get("avg_diameter_px", 0.0)
        anomalies = stats.get("anomaly_coins_count", 0)

        val_display = f"{curr} {total_val:.2f}" if total_val > 0 else "N/A (Unclassified)"
        if stats.get("is_monetary_partial"):
            val_display += " (Partial)"

        summary_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Total Coins Detected", str(total_coins), "Average Diameter (px)", f"{avg_diam} px"],
            ["Classified Denominations", str(classified_coins), "Coins with Anomalies", str(anomalies)],
            ["Total Face Value", val_display, "Overlapping Regions", str(stats.get("touching_coins_count", 0))]
        ]

        t_summary = Table(summary_data, colWidths=[130, 135, 130, 135])
        t_summary.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F9FAFB"), colors.white]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t_summary)
        story.append(Spacer(1, 12))

        # 3. Detection Image
        if annotated_rgb is not None:
            story.append(Paragraph("Annotated Visual Detections", h2_style))
            img_pil = Image.fromarray(annotated_rgb)
            img_pil.thumbnail((480, 240))
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='JPEG', quality=85)
            img_byte_arr.seek(0)
            rl_img = RLImage(img_byte_arr, width=420, height=210)
            story.append(rl_img)
            story.append(Spacer(1, 10))

        # 4. Detailed Coin Records Table
        story.append(Paragraph("Individual Coin Measurements & Findings", h2_style))
        coin_table_data = [
            ["ID", "Denomination", "Diam (px)", "Diam (mm)", "Circularity", "Condition Category", "Ref Mass"]
        ]

        for c in analysis_result.get("coins", [])[:25]:  # Limit table rows to prevent multi-page overflow
            cl = c.get("classification", {})
            geo = c.get("geometry", {})
            cond = c.get("condition", {})
            wt = c.get("weight", {})

            d_mm = f"{geo.get('diameter_mm')} mm" if geo.get("diameter_mm") is not None else "Uncalib."
            mass_str = f"{wt.get('reference_mass_g')} g" if wt.get("reference_mass_g") is not None else "N/A"

            coin_table_data.append([
                f"#{c.get('id')}",
                cl.get("denomination", "Uncertain")[:22],
                f"{geo.get('diameter_px')}",
                d_mm,
                f"{geo.get('circularity'):.3f}",
                cond.get("condition_category", "N/A")[:20],
                mass_str
            ])

        t_coins = Table(coin_table_data, colWidths=[35, 120, 55, 65, 55, 130, 60])
        t_coins.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D9488")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_coins)
        story.append(Spacer(1, 14))

        # 5. Scientific Limitations & Disclaimers
        story.append(Paragraph("Image Processing Limitations & Numismatic Disclaimers", h2_style))
        for line in cls.LIMITATIONS_NOTE.split("\n"):
            story.append(Paragraph(line, disclaimer_style))
            story.append(Spacer(1, 3))

        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
