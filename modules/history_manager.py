"""
Local analysis history manager using SQLite for Smart Coin Detection System.
Stores session summaries, allows historical reviews, record deletions, and exports.
"""

from typing import Dict, Any, List, Optional
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from config import HISTORY_DB_PATH


class HistoryManager:
    """Manages SQLite database for historical coin detection runs."""

    def __init__(self, db_path: Path = HISTORY_DB_PATH):
        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.db_path))

    def _init_database(self):
        """Initializes SQLite schema if not existing."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    dimensions TEXT NOT NULL,
                    total_coins INTEGER NOT NULL,
                    classified_coins INTEGER NOT NULL,
                    total_value REAL NOT NULL,
                    currency TEXT,
                    summary_json TEXT,
                    config_json TEXT,
                    coins_json TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_analysis(
        self,
        filename: str,
        width: int,
        height: int,
        summary_stats: Dict[str, Any],
        config_parameters: Dict[str, Any],
        coins_list: List[Dict[str, Any]]
    ) -> int:
        """Saves an analysis record into SQLite and returns the generated analysis ID."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dims = f"{width}x{height}"
        total_coins = summary_stats.get("total_coins", 0)
        classified_coins = summary_stats.get("classified_coins", 0)
        total_val = summary_stats.get("total_monetary_value", 0.0)
        currency = summary_stats.get("monetary_currency", "N/A")

        # Strip non-serializable elements (like numpy arrays) from coins list before saving
        clean_coins = []
        for c in coins_list:
            clean_coins.append({
                "id": c.get("id"),
                "center_x": c.get("center_x"),
                "center_y": c.get("center_y"),
                "radius_px": c.get("radius_px"),
                "diameter_px": c.get("diameter_px"),
                "diameter_mm": c.get("diameter_mm"),
                "detection_confidence": c.get("detection_confidence"),
                "is_touching": c.get("is_touching"),
                "classification": c.get("classification"),
                "condition": {
                    "category": c["condition"].get("condition_category"),
                    "anomalies": c["condition"].get("anomaly_count"),
                    "area_pct": c["condition"].get("anomaly_area_pct")
                },
                "weight": c.get("weight"),
                "age": c.get("age")
            })

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_history 
                (timestamp, filename, dimensions, total_coins, classified_coins, total_value, currency, summary_json, config_json, coins_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts,
                filename,
                dims,
                total_coins,
                classified_coins,
                total_val,
                currency,
                json.dumps(summary_stats),
                json.dumps(config_parameters),
                json.dumps(clean_coins)
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_all_records(self) -> pd.DataFrame:
        """Retrieves history as a pandas DataFrame."""
        conn = self._get_connection()
        try:
            df = pd.read_sql_query("""
                SELECT id, timestamp, filename, dimensions, total_coins, classified_coins, total_value, currency 
                FROM analysis_history 
                ORDER BY id DESC
            """, conn)
            return df
        finally:
            conn.close()

    def get_record_by_id(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """Fetches full saved session details by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,))
            row = cursor.fetchone()
            if not row:
                return None

            return {
                "id": row[0],
                "timestamp": row[1],
                "filename": row[2],
                "dimensions": row[3],
                "total_coins": row[4],
                "classified_coins": row[5],
                "total_value": row[6],
                "currency": row[7],
                "summary": json.loads(row[8]) if row[8] else {},
                "config": json.loads(row[9]) if row[9] else {},
                "coins": json.loads(row[10]) if row[10] else []
            }
        finally:
            conn.close()

    def delete_record(self, analysis_id: int) -> bool:
        """Deletes a specific history record."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analysis_history WHERE id = ?", (analysis_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def clear_history(self) -> bool:
        """Wipes the entire history table."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analysis_history")
            conn.commit()
            return True
        finally:
            conn.close()
