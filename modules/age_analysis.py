"""
Coin history, minting year, and age estimation module for Smart Coin Detection System.
Provides historical issue periods, validates 4-digit calendar years,
computes approximate chronological age, and upholds numismatic integrity.
"""

from typing import Dict, Any, Optional
import datetime
import re


class AgeEstimator:
    """Manages coin minting years, historical ranges, and age calculations."""

    CURRENT_YEAR = datetime.date.today().year

    @classmethod
    def analyze_age_and_history(
        cls,
        is_classified: bool,
        reference_data: Optional[Dict[str, Any]],
        manual_or_ocr_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculates chronological age when a confirmed minting year is provided,
        or reports the documented issue range from official mint records.
        """
        issue_period = "Unknown"
        hist_source = "Unavailable"
        if reference_data:
            issue_period = reference_data.get("issue_period", "Unknown")
            hist_source = reference_data.get("source", "Reference Database")

        result = {
            "minting_year": None,
            "issue_period": issue_period,
            "estimated_age_years": None,
            "age_status": "Year Unknown",
            "historical_source": hist_source,
            "display_text": "",
            "notes": ""
        }

        # Validate year if supplied
        confirmed_year = None
        if manual_or_ocr_year is not None:
            try:
                yr = int(manual_or_ocr_year)
                # Reasonable numismatic year boundary
                if 1500 <= yr <= cls.CURRENT_YEAR:
                    confirmed_year = yr
            except (ValueError, TypeError):
                pass

        if confirmed_year is not None:
            age = cls.CURRENT_YEAR - confirmed_year
            result["minting_year"] = confirmed_year
            result["estimated_age_years"] = age
            result["age_status"] = f"Confirmed Mint Year ({confirmed_year})"
            result["display_text"] = f"Minted in {confirmed_year} (~{age} years old relative to {cls.CURRENT_YEAR})"
            result["notes"] = (
                f"Calculated as Current Year ({cls.CURRENT_YEAR}) − Confirmed Year ({confirmed_year}). "
                "Represents chronological age since striking, distinct from duration of active circulation."
            )
            return result

        # Year not readable from image
        if is_classified and issue_period != "Unknown":
            result["age_status"] = f"Issue Range: {issue_period}"
            result["display_text"] = f"Minting year could not be determined from the image. Issue series: {issue_period}"
            result["notes"] = (
                f"The identified coin design was struck during {issue_period}. "
                "Because individual numeral dies cannot be resolved from optical resolution alone, "
                "exact age is not calculated to prevent fabrication."
            )
        else:
            result["age_status"] = "Year Unknown"
            result["display_text"] = "Minting year could not be determined from the image."
            result["notes"] = "Coin type or minting date could not be established from visual inspection."

        return result
