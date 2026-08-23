# MicroScan — Microplastic Detection Platform

A web-based Digital Image Processing (DIP) platform for detecting and classifying microplastic particles in circular petri-dish microscope images. All image processing occurs entirely in memory — no files are ever saved to disk.

https://microplastic-detection-kbg8.onrender.com/
---

## Features

| Feature | Description |
|---|---|
| **Circular ROI Detection** | Hough Circle Transform automatically locates and crops the petri dish region |
| **Full DIP Pipeline** | 8-stage pipeline: Enhancement → CLAHE → Denoising → Spatial Filter → Multi-Method Thresholding → Advanced Morphology → Contour Detection |
| **Particle Classification** | Fiber / Fragment / Pellet based on circularity, aspect ratio, and solidity |
| **Contamination Score** | `(microplastic_area / dish_area) × 100` — Low / Medium / High levels |
| **Single Image Mode** | Full analysis with step-by-step visualisation (opens in new tab) |
| **Transparent Material Detection** | Improved algorithm detects transparent & semi-transparent microplastics on dark and light backgrounds |
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
| 2. Enhancement | Percentile-based LAB stretch + aggressive unsharp mask | Better handling of dark/light backgrounds |
| 3. CLAHE | `cv2.createCLAHE(clipLimit=2.5)` | Adaptive local contrast equalisation |
| 4. Denoising | `cv2.fastNlMeansDenoisingColored(h=7)` | Preserves edges while reducing noise |
| 5. Spatial Filter | Sharpening kernel + Laplacian overlay | Accentuates particle boundaries |
| 6. Thresholding (IMPROVED) | **Multi-method:** Otsu + Adaptive + Morphological Gradient + Canny Edge | Detects opaque, semi-transparent, and transparent particles |
| 7. Morphology (IMPROVED) | Multi-scale Open → Close → Dilate → Close → Open | Removes noise while preserving small particles; better for transparent materials |
| 8. Contour Detection | `cv2.findContours` + feature extraction | Classification by aspect ratio, circularity, solidity |

---

## Classification Rules

| Type | Criteria |
|---|---|
| **Pellet** | Circularity > 0.65 **AND** Solidity > 0.75 |
| **Fiber** | Aspect Ratio > 2.5 |
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

**Few or no particles detected** — The improved multi-method thresholding (Step 6) now detects transparent and semi-transparent particles using edge detection. If still getting low counts, verify image quality and ensure particles are visible under the microscope.

**PDF download doesn't work** — Ensure `reportlab` is installed correctly: `pip install reportlab`.

**Slow processing** — Install OpenCV with OpenCL support, or reduce input image resolution. Denoising and multi-method thresholding are the most expensive steps. Typical processing time: 2-5 seconds per image on standard hardware.

---

## Recent Improvements (v2.0)

### Algorithm Enhancements for Transparent Microplastic Detection

The detection pipeline has been significantly improved to handle transparent and semi-transparent plastic materials on both dark and light backgrounds:

#### 1. **Enhanced Image Preprocessing**
- Switched from naive min/max contrast stretching to percentile-based (2nd–98th percentile)
- More aggressive unsharp masking (1.8× vs 1.5×) for better edge definition
- Improved handling of non-uniform lighting conditions

#### 2. **Multi-Method Thresholding (Step 6)**
Previous single-method approach missed transparent particles. Now combines four complementary methods:
- **Otsu Thresholding**: Detects dark particles
- **Adaptive Gaussian**: Captures local contrast differences
- **Morphological Gradient** (NEW): Detects particle boundaries and edges
- **Canny Edge Detection** (NEW): Captures fine transparent particle boundaries

All methods combined via OR logic—if any method detects a particle, it's included.

#### 3. **Advanced Morphological Operations (Step 7)**
- Multi-scale opening, closing, and dilation for better artifact removal
- Gentle dilation to connect edge-detected regions (critical for transparent materials)
- Better preservation of small particles while eliminating noise

#### 4. **Improved Contour Filtering**
- Lowered minimum particle size from 30px to 15px (catches smaller microplastics)
- Increased maximum size to 10% of image (fewer false rejections)
- Added perimeter validation to eliminate extremely thin artifacts

#### 5. **Relaxed Classification Thresholds**
More flexible criteria for semi-transparent particles:
- **Pellet**: circularity > 0.65 (was 0.70), solidity > 0.75 (was 0.80)
- **Fiber**: aspect ratio > 2.5 (was 3.0)

### Result
✅ Transparent microplastics now detectable  
✅ Better performance on dark backgrounds  
✅ Better performance on light backgrounds  
✅ Fewer false negatives overall  

---

## License

MIT — Free to use for research and educational purposes.
# Microplastic-Detection-using-DIP
