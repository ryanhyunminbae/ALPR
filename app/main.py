"""
FastAPI application wiring together detection, OCR, and lookup components.

The service exposes a simple HTML form for manual testing and a JSON API for
programmatic use. Upload an image containing a vehicle license plate to receive
detected bounding boxes, recognized plate text, and demo vehicle metadata.
"""

from __future__ import annotations

import base64
import os
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .detector import Detection, PlateDetector
from .lookup import VehicleLookup
from .ocr import PlateOCR
from .websearch import search_public_info


def _image_from_upload(data: bytes) -> np.ndarray:
    """Decode raw bytes from an uploaded image."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image. Ensure the file is a valid JPEG or PNG.")
    return image


def _encode_image(image: np.ndarray, format: str = ".jpg") -> str:
    """Encode an image as base64 for JSON transport."""
    success, buffer = cv2.imencode(format, image)
    if not success:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(buffer).decode("utf-8")


class PlateMatch(BaseModel):
    bbox: List[int] = Field(..., description="[x, y, width, height] of detection")
    text: Optional[str] = Field(None, description="Normalized plate text")
    confidence: Optional[float] = Field(
        None, description="OCR confidence score (0-1) for the top candidate"
    )
    lookup: Optional[dict] = Field(
        None, description="Vehicle metadata from demo dataset if a match is found"
    )
    plate_crop_base64: Optional[str] = Field(
        None,
        description="JPEG-encoded base64 crop of the detected plate region for display purposes.",
    )
    web_results: Optional[List[dict]] = Field(
        None,
        description=(
            "Optional list of web search results containing public references to the detected plate."
        ),
    )


class AnalyzeResponse(BaseModel):
    plates: List[PlateMatch]
    image_size: dict
    annotated_image_base64: Optional[str] = Field(
        None, description="JPEG-encoded base64 version of the image with bounding boxes."
    )


class ALPRService:
    """High-level orchestrator for plate detection, OCR, and lookup."""

    def __init__(
        self,
        detector: PlateDetector | None = None,
        ocr: PlateOCR | None = None,
        lookup: VehicleLookup | None = None,
    ) -> None:
        self.detector = detector or PlateDetector()
        self.ocr = ocr or PlateOCR()
        self.lookup = lookup or VehicleLookup()

    def analyze(self, image: np.ndarray) -> AnalyzeResponse:
        detections = self.detector.detect(image)
        annotated = self.detector.draw_detections(image, detections) if detections else image

        plate_matches: List[PlateMatch] = []

        for detection in detections:
            crop = detection.crop(image)
            try:
                ocr_results = self.ocr.read_plate(crop, candidates=1)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"OCR failed: {exc}") from exc

            if ocr_results:
                text, confidence = ocr_results[0]
                metadata = self.lookup.lookup(text)
                web_results = search_public_info(text) or None
            else:
                text, confidence, metadata, web_results = None, None, None, None

            try:
                crop_encoded = _encode_image(crop)
            except ValueError:
                crop_encoded = None

            plate_matches.append(
                PlateMatch(
                    bbox=list(detection.bbox),
                    text=text,
                    confidence=confidence,
                    lookup=metadata,
                    plate_crop_base64=crop_encoded,
                    web_results=web_results,
                )
            )

        try:
            annotated_b64 = _encode_image(annotated)
        except ValueError:
            annotated_b64 = None

        return AnalyzeResponse(
            plates=plate_matches,
            image_size={"height": int(image.shape[0]), "width": int(image.shape[1])},
            annotated_image_base64=annotated_b64,
        )


service = ALPRService()

app = FastAPI(
    title="ALPR Demo Service",
    version="0.1.0",
    description=(
        "Educational demo showcasing license plate detection, OCR, and lookup backed by a local dataset. "
        "Do not use in production or for real enforcement scenarios."
    ),
)


@app.get("/", response_class=HTMLResponse)
async def landing_page() -> str:
    """Simple upload form for manual testing."""
    return """
    <html>
        <head>
            <title>ALPR Demo</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; background: #f7f7f9; color: #1f2933; }
                h1 { margin-bottom: 0.25rem; }
                h2 { margin-top: 2rem; }
                form { margin-top: 1rem; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(15,23,42,0.1); }
                button { background: #2563eb; color: white; border: none; padding: 0.65rem 1.2rem; border-radius: 6px; cursor: pointer; font-size: 1rem; }
                button:disabled { opacity: 0.6; cursor: not-allowed; }
                code { background: #e2e8f0; padding: 0.2rem 0.4rem; border-radius: 4px; }
                pre { background: #0f172a; color: #f8fafc; padding: 1rem; border-radius: 8px; overflow-x: auto; }
                .result-card { margin-top: 1.5rem; padding: 1.5rem; border-radius: 10px; background: white; box-shadow: 0 1px 3px rgba(15,23,42,0.1); }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.5rem; margin-top: 1rem; }
                .card { padding: 1rem; background: #f1f5f9; border-radius: 8px; }
                .muted { color: #64748b; font-size: 0.9rem; }
                img { max-width: 100%; border-radius: 8px; border: 1px solid #cbd5f5; }
                footer { margin-top: 3rem; font-size: 0.85rem; color: #475569; }
            </style>
        </head>
        <body>
            <h1>Automatic License Plate Recognition Demo</h1>
            <p class="muted">Upload an image to inspect detection, OCR, and lookup details. No data leaves your machine.</p>
            <form id="upload-form">
                <label>
                    <strong>Select an image:</strong><br />
                    <input type="file" name="file" accept="image/png, image/jpeg" required />
                </label>
                <div style="margin-top: 1rem;">
                    <button type="submit">Analyze Image</button>
                    <span id="status" class="muted" style="margin-left: 0.75rem;"></span>
                </div>
            </form>

            <div id="results" style="display: none;">
                <div class="result-card">
                    <h2>Detection Summary</h2>
                    <p class="muted">Bounding boxes highlight detected license plates. Confidence is the YOLO detection probability (0–1).</p>
                    <div class="grid" id="plate-cards"></div>
                </div>
                <div class="result-card" id="annotated-section" style="display:none;">
                    <h2>Annotated Image</h2>
                    <p class="muted">This image overlays detection boxes produced by the detector.</p>
                    <img id="annotated-image" alt="Annotated detection result"/>
                </div>
                <div class="result-card">
                    <h2>Raw JSON Response</h2>
                    <p class="muted">Structured payload returned by <code>/analyze</code>, formatted for readability.</p>
                    <pre id="json-output"></pre>
                </div>
            </div>

            <div class="result-card" style="margin-top: 2rem;">
                <h2>Field Reference</h2>
                <div class="grid">
                    <div class="card">
                        <strong>plates[]</strong>
                        <p class="muted">Each detected plate, including the bounding box and OCR results.</p>
                    </div>
                    <div class="card">
                        <strong>bbox</strong>
                        <p class="muted">Four integers <code>[x, y, width, height]</code> describing the plate location in pixels.</p>
                    </div>
                    <div class="card">
                        <strong>text & confidence</strong>
                        <p class="muted">Cleaned OCR output and its confidence score (0–1). <em>null</em> indicates OCR failure.</p>
                    </div>
                    <div class="card">
                        <strong>lookup</strong>
                        <p class="muted">Demo metadata retrieved from the local CSV (make, model, year, color, status). Missing if plate not in dataset.</p>
                    </div>
                    <div class="card">
                        <strong>plate_crop_base64</strong>
                        <p class="muted">Base64-encoded JPEG of the cropped plate region. Useful for debugging or rendering inline.</p>
                    </div>
                    <div class="card">
                        <strong>annotated_image_base64</strong>
                        <p class="muted">Entire frame annotated with detection boxes (base64 JPEG) for quick inspection.</p>
                    </div>
                    <div class="card">
                        <strong>web_results</strong>
                        <p class="muted">Optional list of public search hits (title, URL, snippet). Set <code>PLATE_WEB_SEARCH_ENABLED=1</code> before starting the server to activate.</p>
                    </div>
                </div>
            </div>

            <footer>
                <p>API users can POST directly to <code>/analyze</code> with <code>multipart/form-data</code>. See README for more details.</p>
            </footer>

            <script>
                const form = document.getElementById("upload-form");
                const statusEl = document.getElementById("status");
                const resultsEl = document.getElementById("results");
                const jsonOutput = document.getElementById("json-output");
                const plateCards = document.getElementById("plate-cards");
                const annotatedSection = document.getElementById("annotated-section");
                const annotatedImg = document.getElementById("annotated-image");

                form.addEventListener("submit", async (ev) => {
                    ev.preventDefault();
                    const fileInput = form.querySelector("input[type=file]");
                    if (!fileInput.files.length) {
                        statusEl.textContent = "Please choose an image before submitting.";
                        return;
                    }

                    const formData = new FormData();
                    formData.append("file", fileInput.files[0]);
                    resultsEl.style.display = "none";
                    annotatedSection.style.display = "none";
                    statusEl.textContent = "Uploading...";

                    try {
                        const response = await fetch("/analyze", {
                            method: "POST",
                            body: formData,
                        });

                        if (!response.ok) {
                            const err = await response.json().catch(() => ({}));
                            throw new Error(err.detail || "Request failed");
                        }

                        const data = await response.json();
                        statusEl.textContent = "Analysis complete.";
                        resultsEl.style.display = "block";
                        plateCards.innerHTML = "";

                        if (Array.isArray(data.plates) && data.plates.length) {
                            data.plates.forEach((plate, index) => {
                                const card = document.createElement("div");
                                card.className = "card";
                                const webResults = Array.isArray(plate.web_results) ? plate.web_results : [];
                                const webMarkup = webResults.length
                                    ? `<ul>${webResults
                                          .map(
                                              (item) =>
                                                  `<li><a href="${item.url}" target="_blank" rel="noopener noreferrer">${item.title || item.url}</a><br/><span class="muted">${item.snippet || ""}</span></li>`
                                          )
                                          .join("")}</ul>`
                                    : "<p class='muted'>No public web references found or web search disabled.</p>";

                                card.innerHTML = `
                                    <strong>Plate ${index + 1}</strong>
                                    <p class="muted">
                                        <span style="display:block;">BBox: [${plate.bbox.join(", ")}]</span>
                                        <span style="display:block;">OCR: ${plate.text ?? "—"} (${plate.confidence !== null ? plate.confidence.toFixed(2) : "n/a"})</span>
                                    </p>
                                    ${plate.lookup ? `<p><strong>Lookup:</strong><br/>${Object.entries(plate.lookup).map(([k, v]) => `${k}: ${v}`).join("<br/>")}</p>` : "<p class='muted'>No lookup record in demo dataset.</p>"}
                                    <div style="margin-top:0.75rem;">
                                        <strong>Web Mentions</strong>
                                        ${webMarkup}
                                    </div>
                                    ${plate.plate_crop_base64 ? `<img src="data:image/jpeg;base64,${plate.plate_crop_base64}" alt="Plate crop" />` : ""}
                                `;
                                plateCards.appendChild(card);
                            });
                        } else {
                            const card = document.createElement("div");
                            card.className = "card";
                            card.innerHTML = "<strong>No plates detected.</strong><p class='muted'>Try another image or adjust the detector.</p>";
                            plateCards.appendChild(card);
                        }

                        if (data.annotated_image_base64) {
                            annotatedImg.src = `data:image/jpeg;base64,${data.annotated_image_base64}`;
                            annotatedSection.style.display = "block";
                        }

                        jsonOutput.textContent = JSON.stringify(data, null, 2);
                    } catch (error) {
                        console.error(error);
                        statusEl.textContent = `Error: ${error.message}`;
                    }
                });
            </script>
        </body>
    </html>
    """


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    """
    Analyze an uploaded image for license plates.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        image = _image_from_upload(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = service.analyze(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)) if "PORT" in os.environ else 8000,
        reload=False,
    )

