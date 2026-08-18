# God's Eye 👁️

**God's Eye** is an Automatic License Plate Recognition (ALPR) system. It tracks vehicles in video footage using the **frame subtraction (frame differencing) method**, and extracts license plate numbers from vehicle images using contour-based plate localization and OCR.

## How It Works

The project has two stages: **vehicle detection** (from video) and **plate recognition** (from a vehicle image).

### 1. Vehicle Detection — `cdetect.py`

Frame subtraction is a classic, lightweight motion-detection technique — no deep learning model required:

1. **Frame Capture** — Consecutive frames are read from the input video and converted to RGB.
2. **Grayscale Conversion** — Each pair of consecutive frames is converted to grayscale to simplify comparison.
3. **Absolute Difference** — `cv2.absdiff` computes the pixel-wise difference between the current and previous frame. Areas with significant change (i.e., motion) show up as bright regions.
4. **Thresholding** — The difference image is binarized (threshold = 30) to isolate regions of significant change from noise.
5. **Dilation** — A dilation pass fills small gaps in detected motion blobs, making vehicle shapes more contiguous.
6. **Contour Detection** — `cv2.findContours` finds the boundaries of moving regions.
7. **Bounding Boxes** — Contours larger than 20×20 pixels are filtered as vehicles and boxed with a bounding rectangle, filtering out small noise like flickering pixels or shadows.
8. **Output** — Annotated frames are displayed live and written to `output.avi`.

### 2. License Plate Recognition — `driver.py`

Given a vehicle image, this stage locates and reads the license plate:

1. **Preprocessing** — The image is converted to grayscale and binarized with a fixed threshold.
2. **Noise Reduction** — A bilateral filter smooths the image while preserving edges.
3. **Edge Detection** — Canny edge detection highlights object boundaries.
4. **Contour Analysis** — Contours are extracted and sorted by area (largest first, top 10 kept). The algorithm scans for the first roughly **4-sided (rectangular) contour**, which is assumed to be the plate.
5. **Plate Isolation** — A mask is drawn from the detected 4-sided contour, and the image is cropped tightly around that region.
6. **OCR** — [Tesseract](https://github.com/tesseract-ocr/tesseract) (via `pytesseract`) reads the cropped plate region, restricted to alphanumeric characters (`--psm 8`, single-word mode).
7. **Cleanup** — Non-alphanumeric OCR noise is stripped from the final output string.

There's also a `unsharp_mask` utility for image sharpening (defined but not currently wired into the main pipeline).

## Project Structure

| File | Description |
|---|---|
| `cdetect.py` | Vehicle detection from video using frame subtraction. Draws bounding boxes around moving vehicles and writes annotated output to `output.avi`. |
| `driver.py` | License plate recognition (ALPR) on a single vehicle image. Locates the plate via contour detection and reads its text via Tesseract OCR. |

## Getting Started

### Prerequisites

- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and available on your system PATH (required by `pytesseract`)
- A video file of traffic/vehicle footage (for `cdetect.py`)
- A vehicle image (for `driver.py`)

### Installation

```bash
git clone https://github.com/<your-username>/gods-eye.git
cd gods-eye
pip install -r requirements.txt
```

**Dependencies include:**
- `opencv-python`
- `numpy`
- `pytesseract`
- `imutils`
- `matplotlib`

> `pytesseract` is a wrapper around the Tesseract binary — installing the Python package alone isn't enough. Install Tesseract itself separately ([Windows installer](https://github.com/UB-Mannheim/tesseract/wiki), or `apt install tesseract-ocr` / `brew install tesseract` on Linux/Mac).

### Usage

**Vehicle detection**

1. Place your input video at `assets/carVideo.mp4` (or update the path in `cdetect.py`).
2. Run:

```bash
python cdetect.py
```

- Detected vehicles are outlined with green bounding boxes in the live preview window.
- Annotated output is saved to `output.avi`.
- Press `Esc` to quit.

**License plate recognition**

1. Place a vehicle image at `assets/car1.jpg` (or update the path in `driver.py`).
2. Run:

```bash
python driver.py
```

- Several intermediate images (original, thresholded, edge-detected, isolated plate) are displayed step-by-step via `matplotlib` — close each window to proceed to the next.
- The detected plate number is printed to the console.

## Notes & Known Limitations

- **The two stages aren't connected yet** — `cdetect.py` detects vehicles in video, and `driver.py` reads plates from a separate static image. Wiring them together (auto-cropping detected vehicles from video and feeding them into the plate-reading pipeline) is the natural next step.
- Frame subtraction is sensitive to camera movement, lighting changes, and shadows; it works best with a **static camera** and stable lighting.
- Plate localization assumes the plate is the most prominent 4-sided contour in the top 10 largest contours — this can fail on cluttered backgrounds, angled shots, or non-rectangular plate framing.
- OCR accuracy depends heavily on image quality, resolution, and plate angle.
- The `cdetect.py` output `VideoWriter` uses codec `-1` (prompts for codec selection on some systems) and dimensions `640x480` — update these to match your input video's resolution and your platform's supported codecs (e.g., `XVID`, `MJPG`) if you run into issues.

## Roadmap

- [ ] Connect vehicle detection output directly to the plate recognition pipeline
- [ ] Vehicle tracking across frames (persistent IDs, not just per-frame detection)
- [ ] Improve plate localization robustness (perspective correction, multiple candidate contours)
- [ ] Configurable input source (webcam / RTSP stream / file path via CLI args)
- [ ] Batch processing for multiple vehicle images

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

Add your license here (e.g., MIT).
