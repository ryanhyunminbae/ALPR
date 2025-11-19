# ALPR Demo App

An educational **Automatic License Plate Recognition (ALPR)** system built with Python. This project demonstrates a complete ML pipeline: from object detection to OCR to data lookup, all wrapped in a production-ready API.

> ⚠️ **Important:** This project is for educational and testing purposes only. It does **not** access any government or DMV databases and ships with fictional sample data only. Real-world deployments of ALPR technology are subject to strict legal and privacy regulations—consult qualified legal counsel before any practical use.

## What This Project Does

Given an image of a vehicle, this app:
1. **Detects** license plates in the image using a trained YOLO model
2. **Reads** the plate text using OCR (Optical Character Recognition)
3. **Looks up** vehicle information from a local demo database
4. **Optionally searches** the web for public references to the plate
5. **Returns** all results as JSON with annotated images

## How It Works (Architecture Overview)

```
User uploads image
    ↓
FastAPI receives image → decodes to NumPy array
    ↓
PlateDetector (YOLO) → finds license plate bounding boxes
    ↓
For each detected plate:
    ├─ Crop the plate region from image
    ├─ PlateOCR (EasyOCR) → extracts text from crop
    ├─ VehicleLookup → searches CSV for matching plate
    └─ WebSearch (optional) → scrapes public web results
    ↓
Combine all results → return JSON response
```

### Component Breakdown

**1. Detection (`app/detector.py`)**
- Uses **YOLOv8** (You Only Look Once), a state-of-the-art object detection model
- Trained on a Kaggle dataset of car images with license plate annotations
- Outputs bounding boxes `[x, y, width, height]` for each detected plate
- Can detect multiple plates in a single image

**2. OCR (`app/ocr.py`)**
- Uses **EasyOCR**, a pre-trained text recognition model
- Takes the cropped plate image and extracts alphanumeric text
- Normalizes the text (removes spaces, handles common OCR mistakes like `O` vs `0`)
- Returns the recognized text with a confidence score

**3. Lookup (`app/lookup.py`)**
- Reads a local CSV file (`app/data/vehicles.csv`) containing demo vehicle data
- Matches the recognized plate text against the database
- Returns vehicle metadata (make, model, color, year, registration status)
- Uses the same normalization function as OCR to handle formatting differences

**4. Web Search (`app/websearch.py`)**
- Optional feature that searches DuckDuckGo for public references to the plate
- Only enabled when `PLATE_WEB_SEARCH_ENABLED=1` environment variable is set
- Parses search results and returns titles, URLs, and snippets

**5. API (`app/main.py`)**
- Built with **FastAPI**, a modern Python web framework
- Exposes two endpoints:
  - `GET /` - HTML upload form for manual testing
  - `POST /analyze` - JSON API for programmatic access
- Returns structured JSON with all detection, OCR, and lookup results
- Includes base64-encoded images for easy display in web clients

## Features

- **YOLOv8-based detection** - Trained on real license plate data from Kaggle
- **EasyOCR integration** - Pre-trained OCR model with text normalization
- **Local CSV database** - Fast, privacy-safe lookup without external APIs
- **Optional web search** - DuckDuckGo integration for public plate references
- **RESTful API** - FastAPI with automatic OpenAPI documentation
- **Modern web UI** - HTML dashboard with JSON visualization
- **Base64 image encoding** - Annotated images and plate crops included in responses
- **Comprehensive tests** - Unit tests for all core components

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Internet connection (for downloading ML model weights on first run)

### Installation

```bash
# Clone or navigate to the project directory
cd /Users/ryanbae/lprs

# Create a virtual environment
python -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate
# Or on Windows:
# .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Important:** You'll need to train or download a YOLO model for detection. See the "Training Your Own Detector" section below.

### Running the API

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate

# Start the server
uvicorn app.main:app --reload
```

The service will be available at `http://127.0.0.1:8000`

- Open `http://127.0.0.1:8000` in your browser to use the web interface
- Or use the API directly:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

### Response Format

The API returns JSON with this structure:

```json
{
  "plates": [
    {
      "bbox": [100, 200, 150, 50],
      "text": "ABC1234",
      "confidence": 0.95,
      "lookup": {
        "plate": "ABC1234",
        "make": "Toyota",
        "model": "Corolla",
        "color": "Blue",
        "year": "2018",
        "registration_status": "Active"
      },
      "plate_crop_base64": "data:image/jpeg;base64,...",
      "web_results": [...]
    }
  ],
  "image_size": {"width": 1920, "height": 1080},
  "annotated_image_base64": "data:image/jpeg;base64,..."
}
```

## Training Your Own Detector

The app expects a YOLOv8 model trained on license plates. Here's how to create one:

### 1. Download a Dataset

We used the [Car Plate Detection dataset from Kaggle](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection):

```bash
# Install KaggleHub
pip install kagglehub

# Download the dataset
python scripts/download_kaggle_dataset.py
```

### 2. Convert Annotations

The dataset comes in Pascal VOC XML format. Convert it to YOLO format:

```bash
python convert_voc_to_yolo.py \
  --voc-root /path/to/downloaded/dataset \
  --yolo-root dataset/all
```

### 3. Split into Train/Val

```bash
# The conversion script handles this automatically
# You'll have dataset/train/ and dataset/val/ directories
```

### 4. Train YOLO

```bash
# Install Ultralytics
pip install ultralytics

# Train the model
yolo detect train \
  data=dataset/plates.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  project=runs \
  name=alpr-plates
```

### 5. Copy Weights to App

```bash
cp runs/detect/alpr-plates/weights/best.pt \
   app/models/yolov8n-plates.pt
```

## Project Structure

```
lprs/
├── app/
│   ├── main.py           # FastAPI application and endpoints
│   ├── detector.py        # YOLO-based plate detection
│   ├── ocr.py            # EasyOCR text recognition
│   ├── lookup.py          # CSV database lookup
│   ├── websearch.py      # DuckDuckGo web search integration
│   ├── data/
│   │   └── vehicles.csv  # Demo vehicle database
│   └── models/
│       └── yolov8n-plates.pt  # Trained YOLO weights (you provide)
├── tests/
│   ├── test_detector.py
│   ├── test_ocr.py
│   ├── test_lookup.py
│   └── test_websearch.py
├── scripts/
│   └── download_kaggle_dataset.py
├── requirements.txt
└── README.md
```

## Configuration

### Environment Variables

- `PLATE_WEB_SEARCH_ENABLED=1` - Enable web search feature (disabled by default)
- `PORT=8000` - Change the server port (default: 8000)

### Customization

- **Detection confidence**: Adjust `confidence` parameter in `PlateDetector.__init__()`
- **OCR normalization**: Modify `normalize_plate_text()` in `app/ocr.py`
- **Database**: Edit `app/data/vehicles.csv` to add your own demo data

## Testing

Run the test suite:

```bash
pytest
```

Tests cover:
- Text normalization logic
- CSV lookup functionality
- Detector initialization
- Web search parsing

## Troubleshooting

**"YOLO weights not found"**
- Make sure you've trained or downloaded a YOLO model and placed it at `app/models/yolov8n-plates.pt`
- See "Training Your Own Detector" section above

**"No plates detected"**
- Try images with clear, front-facing license plates
- Lower the confidence threshold in `PlateDetector` (default: 0.25)
- Retrain the model on images similar to your use case

**"OCR returns wrong text"**
- OCR accuracy depends on image quality, lighting, and plate angle
- Try preprocessing the images (brightness, contrast adjustment) before upload
- Consider training a custom OCR model on plate-specific fonts

**"Web search not working"**
- Set `PLATE_WEB_SEARCH_ENABLED=1` environment variable
- Check your internet connection
- DuckDuckGo may rate-limit requests

## Technical Stack

- **Python 3.10+** - Core language
- **FastAPI** - Web framework and API
- **YOLOv8 (Ultralytics)** - Object detection
- **EasyOCR** - Text recognition
- **OpenCV** - Image processing
- **Pandas** - Data manipulation
- **Pydantic** - Data validation
- **Pytest** - Testing framework

## Legal & Privacy Disclaimer

This repository is a **demo project** for educational purposes. It is not production-ready and must not be used for:
- Surveillance or monitoring of individuals
- Law enforcement without proper authorization
- Any activity involving real personal data without consent

ALPR technology is regulated in many jurisdictions. Ensure full compliance with:
- Local privacy laws (GDPR, CCPA, etc.)
- Data protection regulations
- Ethical guidelines for ML/AI systems

The authors assume no liability for misuse of this software.
# ALPR
