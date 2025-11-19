from app.ocr import normalize_plate_text


def test_normalize_plate_text_strips_noise():
    assert normalize_plate_text("  aBc -123 ") == "ABC123"


def test_normalize_plate_text_replaces_common_confusions():
    assert normalize_plate_text("O1L") == "011"


def test_normalize_plate_text_handles_empty():
    assert normalize_plate_text("") == ""

