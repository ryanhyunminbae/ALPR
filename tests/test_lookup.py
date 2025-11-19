from pathlib import Path

import pandas as pd
import pytest

from app.lookup import VehicleLookup


def write_csv(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_lookup_returns_matching_record(tmp_path: Path):
    csv_path = tmp_path / "vehicles.csv"
    write_csv(
        csv_path,
        [
            {"plate": "ABC123", "make": "Test", "model": "Car", "color": "Blue", "year": "2020", "registration_status": "Active"}
        ],
    )

    lookup = VehicleLookup(csv_path)
    result = lookup.lookup("abc-123")
    assert result is not None
    assert result["make"] == "Test"
    assert result["plate"] == "ABC123"


def test_lookup_returns_none_when_missing(tmp_path: Path):
    csv_path = tmp_path / "vehicles.csv"
    write_csv(
        csv_path,
        [
            {"plate": "ZZZ999", "make": "Test", "model": "Car", "color": "Red", "year": "2019", "registration_status": "Expired"}
        ],
    )

    lookup = VehicleLookup(csv_path)
    assert lookup.lookup("NOTFOUND") is None


def test_lookup_requires_plate_column(tmp_path: Path):
    csv_path = tmp_path / "vehicles.csv"
    write_csv(csv_path, [{"foo": "bar"}])

    with pytest.raises(ValueError):
        VehicleLookup(csv_path)

