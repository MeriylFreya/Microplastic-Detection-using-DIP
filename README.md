# MicroScan — Microplastic Detection Platform

A web-based Digital Image Processing (DIP) platform for detecting and classifying microplastic particles in circular petri-dish microscope images. All image processing occurs entirely in memory — no files are ever saved to disk.

---

## Features

| Feature | Description |
|---|---|
| **Circular ROI Detection** | Hough Circle Transform automatically locates and crops the petri dish region |
| **Full DIP Pipeline** | 8-stage pipeline: Enhancement → CLAHE → Denoising → Spatial Filter → Thresholding → Morphology → Contour Detection |
| **Particle Classification** | Fiber / Fragment / Pellet based on circularity, aspect ratio, and solidity |
| **Contamination Score** | `(microplastic_area / dish_area) × 100` — Low / Medium / High levels |
| **Single Image Mode** | Full analysis with step-by-step visualisation (opens in new tab) |
| **Bulk Mode** | Process many images at once with comparison charts and a combined PDF report |
| **Session Dashboard** | History of all analyses this session, with re-view and PDF download |
| **PDF Reports** | ReportLab-generated PDF with images, tables, and charts — base64, no disk save |
| **Zero disk I/O** | All images handled as base64 strings in memory |

---

## Project Structure

```
project/
│── app.py              # Flask routes and session management
│── processing.py       # Full DIP pipeline (OpenCV)
│── utils.py            # PDF generation, graphs, thumbnail helper
│── requirements.txt    # Python dependencies
│── templates/
│    ├── base.html       # Shared layout (nav, loader, styles)
│    ├── index.html      # Home page with mode selection
│    ├── single.html     # Single image upload + results
│    ├── single_steps.html # Step-by-step DIP visualisation
│    ├── bulk.html       # Bulk upload form
│    ├── bulk_result.html # Bulk results table + charts
│    └── dashboard.html  # Session history dashboard
└── README.md
```

---

## Quick Setup

### 1. Prerequisites

- Python 3.9 or higher
- pip

### 2. Install Dependencies

```bash
cd project
pip install -r requirements.txt
```

> **Note for Linux users:** If `opencv-python-headless` has issues, install system deps:
> ```bash
> sudo apt-get install -y libgl1 libglib2.0-0
> ```

### 3. Run the App

```bash
python app.py
```

Then open your browser at: **http://localhost:5000**

---

## How to Use

### Single Image Analysis
1. Click **Single Image** on the home page
2. Upload a petri-dish microscope image (JPG, PNG, BMP, TIFF)
3. Click **Run DIP Analysis**
4. View the detection result, stats, and classification breakdown
5. Click **Show Processing Steps** to see all 8 pipeline stages in a new tab
6. Click **Download PDF Report** to get a full report

### Bulk Upload
1. Click **Bulk Upload** on the home page
2. Select multiple images (Ctrl+click or Shift+click)
3. Click **Process All Images**
4. Review the results table, comparison charts, and thumbnail grid
5. Click **Download Bulk PDF Report**

### Dashboard
1. Click **Dashboard** in the navigation bar
2. All analyses from your current browser session are listed
3. Click **View Result** to re-examine any past analysis
4. Click **PDF** to download the report for that image
5. History is cleared when the browser session ends

---

## DIP Pipeline Details

| Step | Method | Notes |
|---|---|---|
| 1. Circle Detection | `cv2.HoughCircles` + fallback center crop | Detects petri dish; masks outside pixels |
| 2. Enhancement | Auto LAB contrast stretch + unsharp mask | Amplifies particle-background contrast |
| 3. CLAHE | `cv2.createCLAHE(clipLimit=2.5)` | Adaptive local contrast equalisation |
| 4. Denoising | `cv2.fastNlMeansDenoisingColored(h=7)` | Preserves edges while reducing noise |
| 5. Spatial Filter | Sharpening kernel + Laplacian overlay | Accentuates particle boundaries |
| 6. Thresholding | Otsu OR Adaptive Gaussian | Combined mask captures more particle types |
| 7. Morphology | Open → Close → Dilate | Removes noise, fills holes, connects regions |
| 8. Contour Detection | `cv2.findContours` + feature extraction | Classification by aspect ratio, circularity, solidity |

---

## Classification Rules

| Type | Criteria |
|---|---|
| **Pellet** | Circularity > 0.70 **AND** Solidity > 0.80 |
| **Fiber** | Aspect Ratio > 3.0 |
| **Fragment** | All other irregular shapes |

---

## Contamination Scoring

```
score = (total_microplastic_pixel_area / total_dish_pixel_area) × 100
```

| Level | Score Range |
|---|---|
| 🟢 Low | 0 – 20% |
| 🟡 Medium | 20 – 50% |
| 🔴 High | 50 – 100% |

---

## Technical Notes

- **Session storage:** Flask's signed cookie session stores a session ID; actual history data lives in a server-side Python dict (`_history_store` in `app.py`). History is tied to the browser session and cleared on restart.
- **No disk writes:** Every image is encoded as a base64 PNG string. PDFs are generated into `io.BytesIO` buffers and base64-encoded before transmission.
- **Memory usage:** For large bulk uploads (50+ images), RAM usage can be significant since all images are held in memory. Recommend processing ≤ 30 images per bulk job on typical hardware.
- **Image size:** Very large images (>4000×4000 px) will be processed but may be slow. Consider resizing inputs to 1024×1024 for best performance.

---

## Troubleshooting

**"No circle detected"** — The fallback crops to the image center at 80% of the shorter dimension. Ensure the petri dish is reasonably centred and fills most of the frame.

**Few or no particles detected** — Try images with higher contrast between particles and the dish background. Very transparent particles may need manual threshold tuning in `processing.py`.

**PDF download doesn't work** — Ensure `reportlab` is installed correctly: `pip install reportlab`.

**Slow processing** — Install OpenCV with OpenCL support, or reduce input image resolution. Denoising (`fastNlMeansDenoisingColored`) is the most expensive step.

---

## License

MIT — Free to use for research and educational purposes.
# Microplastic-Detection-using-DIP
