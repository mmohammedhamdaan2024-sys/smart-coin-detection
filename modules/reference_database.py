"""
Coin reference database interface for Smart Coin Detection System.
Provides CRUD operations, tolerance-based diameter matching, and currency filtering.
"""

from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path
import pandas as pd

from config import REFERENCE_DB_PATH


class ReferenceDatabase:
    """Encapsulates the local coin reference catalog."""

    def __init__(self, db_path: Path = REFERENCE_DB_PATH):
        self.db_path = db_path
        self.data: Dict[str, Any] = {}
        self.coins: List[Dict[str, Any]] = []
        self.load_database()

    def load_database(self) -> bool:
        """Loads reference catalog from JSON file."""
        if not self.db_path.exists():
            self.coins = []
            return False

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
                self.coins = self.data.get("coins", [])
            return True
        except Exception as e:
            print(f"Error loading coin reference database: {e}")
            self.coins = []
            return False

    def save_database(self) -> bool:
        """Persists the current coin catalog back to JSON."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.data["coins"] = self.coins
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving reference database: {e}")
            return False

    def get_all_coins(self) -> List[Dict[str, Any]]:
        """Returns all reference coin records."""
        return self.coins

    def get_currencies(self) -> List[str]:
        """Returns list of distinct currency codes present in database."""
        return sorted(list({c.get("currency", "UNKNOWN") for c in self.coins}))

    def get_coin_by_id(self, reference_id: str) -> Optional[Dict[str, Any]]:
        """Finds a coin by its unique reference ID."""
        for c in self.coins:
            if c.get("reference_id") == reference_id:
                return c
        return None

    def find_candidates_by_diameter(
        self,
        measured_mm: float,
        tolerance_mm: float = 1.0,
        currency_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Finds candidate reference coins whose standard diameter falls within
        [measured_mm - tolerance_mm, measured_mm + tolerance_mm].
        """
        candidates = []
        for c in self.coins:
            if currency_filter and currency_filter != "ALL" and c.get("currency") != currency_filter:
                continue

            ref_d = c.get("reference_diameter_mm", 0.0)
            diff = abs(ref_d - measured_mm)
            if diff <= tolerance_mm:
                # Similarity percentage based on diameter proximity
                sim = max(0.0, 100.0 - (diff / tolerance_mm) * 50.0)
                entry = dict(c)
                entry["diameter_delta_mm"] = round(diff, 2)
                entry["diameter_similarity_pct"] = round(sim, 1)
                candidates.append(entry)

        # Sort closest diameter match first
        candidates.sort(key=lambda x: x["diameter_delta_mm"])
        return candidates

    def add_reference_coin(self, coin_record: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates and appends a new coin record."""
        ref_id = coin_record.get("reference_id", "").strip().upper()
        if not ref_id:
            return False, "Reference ID cannot be empty."

        if self.get_coin_by_id(ref_id):
            return False, f"A coin with Reference ID '{ref_id}' already exists."

        # Ensure required numeric fields
        try:
            coin_record["reference_diameter_mm"] = float(coin_record["reference_diameter_mm"])
            coin_record["face_value"] = float(coin_record["face_value"])
            if "reference_mass_g" in coin_record and coin_record["reference_mass_g"] is not None:
                coin_record["reference_mass_g"] = float(coin_record["reference_mass_g"])
        except ValueError:
            return False, "Diameter, face value, and mass must be valid numeric values."

        self.coins.append(coin_record)
        saved = self.save_database()
        if saved:
            return True, f"Coin '{ref_id}' added successfully."
        return False, "Failed to persist to disk."

    def to_dataframe(self) -> pd.DataFrame:
        """Exports coin database as a pandas DataFrame for UI tables."""
        if not self.coins:
            return pd.DataFrame()
        df = pd.DataFrame(self.coins)
        columns_order = [
            "reference_id", "currency", "country", "denomination", "face_value",
            "reference_diameter_mm", "reference_mass_g", "material", "color_class",
            "issue_period", "source"
        ]
        present_cols = [c for c in columns_order if c in df.columns]
        return df[present_cols]
