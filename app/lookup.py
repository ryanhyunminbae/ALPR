"""
Vehicle lookup utilities backed by a local CSV file.

The lookup intentionally uses a small, non-sensitive dataset for demo purposes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .ocr import normalize_plate_text


class VehicleLookup:
    """
    Simple CSV-backed lookup for vehicle metadata keyed by license plate.
    """

    def __init__(self, csv_path: Path | str | None = None) -> None:
        self._csv_path = Path(csv_path) if csv_path else self._default_csv_path()
        if not self._csv_path.exists():
            raise FileNotFoundError(
                f"Lookup CSV not found at {self._csv_path}. "
                "Create the dataset as described in README.md."
            )
        self._dataframe = self._load_dataframe()

    @staticmethod
    def _default_csv_path() -> Path:
        return Path(__file__).resolve().parent / "data" / "vehicles.csv"

    def _load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self._csv_path, dtype=str).fillna("")
        if "plate" not in df.columns:
            raise ValueError("vehicles.csv must include a 'plate' column.")
        df["plate_normalized"] = df["plate"].map(normalize_plate_text)
        return df

    def refresh(self) -> None:
        """Reload the dataset from disk."""
        self._dataframe = self._load_dataframe()

    def lookup(self, plate_text: str) -> Optional[Dict[str, str]]:
        """
        Lookup vehicle metadata by license plate text.

        Returns the first matching record as a dictionary excluding the
        normalization helper columns.
        """
        normalized = normalize_plate_text(plate_text)
        if not normalized:
            return None

        matches = self._dataframe[self._dataframe["plate_normalized"] == normalized]
        if matches.empty:
            return None

        row = matches.iloc[0].drop(labels=["plate_normalized"])
        return row.to_dict()

