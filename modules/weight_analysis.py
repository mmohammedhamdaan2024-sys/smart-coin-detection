"""
Coin weight analysis module for Smart Coin Detection System.
Enforces strict physical measurement integrity: photographic pixels cannot measure mass.
Provides reference mint masses, allows user-entered scale weights, and generates honest disclosures.
"""

from typing import Dict, Any, Optional


class WeightEstimator:
    """Manages coin weight reporting, reference mass lookups, and manual scale measurements."""

    PHOTO_DISCLAIMER = (
        "A photograph cannot measure a coin's physical weight or mass. "
        "Two coins of identical diameter can have completely different thicknesses, internal alloys, "
        "or hollow structures. Pixel area must never be used to fabricate a weight measurement."
    )

    def analyze_weight(
        self,
        is_classified: bool,
        reference_data: Optional[Dict[str, Any]],
        manual_weight_g: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates weight status:
        - Manual scale measurement (highest ground-truth priority)
        - Published reference mass (if coin denomination is positively identified)
        - Unavailable (if denomination is unknown and no scale measurement provided)
        """
        ref_mass = None
        if reference_data:
            ref_mass = reference_data.get("reference_mass_g")

        result = {
            "weight_status": "Unavailable",
            "actual_weight_g": None,
            "reference_mass_g": ref_mass,
            "mass_delta_g": None,
            "display_text": "",
            "method_notes": "",
            "disclaimer": self.PHOTO_DISCLAIMER
        }

        # Case 1: User provided a physical measurement from a precision scale
        if manual_weight_g is not None and manual_weight_g > 0:
            result["actual_weight_g"] = round(float(manual_weight_g), 3)
            result["weight_status"] = "Measured (Physical Scale)"
            
            if ref_mass:
                delta = round(float(manual_weight_g - ref_mass), 3)
                result["mass_delta_g"] = delta
                result["display_text"] = f"Scale Weight: {manual_weight_g:.2f} g (Ref: {ref_mass:.2f} g, Δ: {delta:+.2f} g)"
                result["method_notes"] = f"Physical scale measurement recorded. Deviation of {delta:+.2f} g reflects normal circulation wear or minting tolerance."
            else:
                result["display_text"] = f"Scale Weight: {manual_weight_g:.2f} g"
                result["method_notes"] = "Physical measurement manually recorded by user using external laboratory scale."
            return result

        # Case 2: Coin is positively classified, reference standard is known
        if is_classified and ref_mass is not None and ref_mass > 0:
            result["weight_status"] = "Reference Standard"
            result["display_text"] = f"Reference Mass: {ref_mass:.2f} grams"
            result["method_notes"] = (
                "This is the official published reference mass for the identified coin type, "
                "not a physical measurement of the uploaded coin."
            )
            return result

        # Case 3: Denomination uncertain or unclassified
        result["weight_status"] = "Unavailable"
        result["display_text"] = "Weight unavailable — identify the coin or provide a physical measurement."
        result["method_notes"] = "Weight cannot be determined from an optical image without known coin type or scale measurement."
        return result
