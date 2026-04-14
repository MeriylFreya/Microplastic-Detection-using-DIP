"""
processing.py - Digital Image Processing Pipeline for Microplastic Detection
All operations are done in memory using base64-encoded images.
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import math


# ─────────────────────────────────────────────
# UTILITY: numpy array ↔ base64
# ─────────────────────────────────────────────

def img_to_b64(img_array):
    """Convert a numpy BGR/Gray image to base64 PNG string."""
    success, buf = cv2.imencode('.png', img_array)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf).decode('utf-8')


def b64_to_img(b64_str):
    """Decode a base64 string to a numpy array (BGR)."""
    img_data = base64.b64decode(b64_str)
    arr = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img


def file_to_b64(file_obj):
    """Read an uploaded file object and return base64 string."""
    data = file_obj.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode uploaded image")
    return img_to_b64(img)


# ─────────────────────────────────────────────
# STEP 1: Detect circular dish and crop ROI
# ─────────────────────────────────────────────

def detect_and_crop_circle(img):
    """
    Detect the largest circular petri dish in the image.
    Returns:
        cropped_img  – square-cropped region around the circle
        mask         – binary circle mask (same size as cropped_img)
        circle_params – (cx, cy, r) in original image coords
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Try Hough Circle detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) * 0.4,
        param1=80,
        param2=40,
        minRadius=int(min(h, w) * 0.15),
        maxRadius=int(min(h, w) * 0.55)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Pick the largest circle
        best = sorted(circles[0], key=lambda c: c[2], reverse=True)[0]
        cx, cy, r = int(best[0]), int(best[1]), int(best[2])
    else:
        # Fallback: assume circle fills ~80% of the shorter dimension, centered
        r = int(min(h, w) * 0.40)
        cx, cy = w // 2, h // 2

    # Pad radius slightly
    r = int(r * 1.02)

    # Crop bounding box
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)

    cropped = img[y1:y2, x1:x2].copy()

    # Build mask in cropped coordinates
    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    local_cx = cx - x1
    local_cy = cy - y1
    local_r = r
    cv2.circle(mask, (local_cx, local_cy), local_r, 255, -1)

    return cropped, mask, (cx, cy, r)


def apply_circle_mask(img, mask):
    """Apply a circular mask so area outside the dish is black."""
    result = img.copy()
    mask3 = cv2.merge([mask, mask, mask])
    result = cv2.bitwise_and(result, mask3)
    return result


# ─────────────────────────────────────────────
# STEP 2: Image Enhancement (Improved)
# ─────────────────────────────────────────────

def enhance_image(img):
    """
    Enhanced contrast and brightness for varying backgrounds.
    Works better with both dark and light backgrounds.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Improved auto-stretch: use percentile instead of min/max to avoid outliers
    l_min = np.percentile(l, 2)
    l_max = np.percentile(l, 98)
    
    if l_max > l_min:
        # More aggressive stretching for better contrast
        l = np.clip((l.astype(np.float32) - l_min) / (l_max - l_min) * 255, 0, 255).astype(np.uint8)

    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Stronger unsharp mask for edge enhancement
    gaussian = cv2.GaussianBlur(enhanced, (5, 5), 1.2)
    enhanced = cv2.addWeighted(enhanced, 1.8, gaussian, -0.8, 0)
    enhanced = np.uint8(np.clip(enhanced, 0, 255))
    
    return enhanced


# ─────────────────────────────────────────────
# STEP 3: Histogram Equalization (CLAHE)
# ─────────────────────────────────────────────

def histogram_equalization(img):
    """Apply CLAHE to the L channel of LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# STEP 4: Noise Removal
# ─────────────────────────────────────────────

def remove_noise(img):
    """Fast Non-Local Means denoising."""
    denoised = cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7,
                                                templateWindowSize=7,
                                                searchWindowSize=21)
    return denoised


# ─────────────────────────────────────────────
# STEP 5: Spatial Filtering (Sharpening + Edge)
# ─────────────────────────────────────────────

def spatial_filtering(img):
    """Sharpening kernel + Laplacian edge overlay."""
    sharpen_kernel = np.array([
        [0, -1,  0],
        [-1,  5, -1],
        [0, -1,  0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(img, -1, sharpen_kernel)

    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
    laplacian_bgr = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

    result = cv2.addWeighted(sharpened, 0.8, laplacian_bgr, 0.2, 0)
    return result


# ─────────────────────────────────────────────
# STEP 6: Thresholding (Improved for transparency)
# ─────────────────────────────────────────────

def threshold_image(img, mask=None):
    """
    Improved thresholding for transparent microplastics.
    Uses: Otsu + Adaptive + Morphological Gradient + Canny Edge.
    Returns binary image (uint8, 0/255).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Method 1: Otsu Thresholding (good for dark objects) ---
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Method 2: Adaptive Thresholding (local contrast) ---
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=8
    )

    # --- Method 3: Morphological Gradient (edges) ---
    # Good for detecting boundaries of transparent objects
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_grad)
    _, grad_thresh = cv2.threshold(grad, 20, 255, cv2.THRESH_BINARY)

    # --- Method 4: Canny Edge Detection ---
    # Very sensitive to edges, helps with transparent materials
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to make them more robust
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)

    # --- Combine all methods: at least 2 methods must detect ---
    # This reduces false positives while catching transparent materials
    combined = cv2.bitwise_or(otsu, adaptive)
    combined = cv2.bitwise_or(combined, grad_thresh)
    combined = cv2.bitwise_or(combined, edges_dilated)

    # Apply dish mask to remove outside-circle detections
    if mask is not None:
        combined = cv2.bitwise_and(combined, mask)

    return combined


# ─────────────────────────────────────────────
# STEP 7: Morphological Operations (Improved)
# ─────────────────────────────────────────────

def morphological_ops(binary_img):
    """
    Advanced morphological operations for microplastic isolation.
    Removes small noise while preserving transparent particles.
    """
    kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Step 1: Remove VERY tiny noise (single pixels)
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_tiny, iterations=1)

    # Step 2: Fill small holes in particles
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # Step 3: Gentle dilation to connect nearby detected edges
    # (important for transparent materials that may have been detected as separate edges)
    dilated = cv2.dilate(closed, kernel_small, iterations=1)

    # Step 4: Remove holes in larger structures
    closed2 = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_med, iterations=1)

    # Step 5: Final gentle opening to clean up artifacts
    final = cv2.morphologyEx(closed2, cv2.MORPH_OPEN, kernel_small, iterations=1)

    return final


# ─────────────────────────────────────────────
# STEP 8: Contour Detection & Feature Extraction
# ─────────────────────────────────────────────

def compute_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4 * math.pi * area) / (perimeter ** 2)


def compute_solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return area / hull_area


def classify_particle(contour):
    """
    Classify a contour as Fiber, Fragment, or Pellet.
    Improved rules for transparent and semi-transparent materials.

    Rules:
      - Pellet:   circularity > 0.65  AND  solidity > 0.75  (slightly relaxed)
      - Fiber:    aspect_ratio > 2.5  (slightly relaxed from 3.0)
      - Fragment: everything else
    """
    area = cv2.contourArea(contour)
    if area < 5:
        return "Unknown"

    # Bounding rect aspect ratio
    x, y, ww, hh = cv2.boundingRect(contour)
    if hh == 0:
        return "Unknown"
    aspect_ratio = max(ww, hh) / (min(ww, hh) + 1e-6)

    circularity = compute_circularity(contour)
    solidity = compute_solidity(contour)

    # Relaxed thresholds to catch transparent particles better
    if circularity > 0.65 and solidity > 0.75:
        return "Pellet"
    elif aspect_ratio > 2.5:
        return "Fiber"
    else:
        return "Fragment"


def detect_contours(binary_img, min_area=15, max_area=None):
    """
    Find and filter contours from a binary image with improved heuristics.
    Returns list of (contour, classification, area, circularity).
    
    Parameters:
    - min_area: minimum particle size in pixels (reduced to 15 to catch small particles)
    - max_area: maximum particle size (prevents detecting dish artifacts)
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = binary_img.shape[:2]
    if max_area is None:
        max_area = (h * w) * 0.10  # no single particle > 10% of image area (increased from 5%)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Filter by contour length: must have a reasonable perimeter
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 10:  # too small to be a real object
            continue
        
        # Classify the particle
        cls = classify_particle(cnt)
        if cls == "Unknown":  # skip unknown particles
            continue
            
        circ = compute_circularity(cnt)
        results.append({
            "contour": cnt,
            "classification": cls,
            "area": area,
            "circularity": circ,
        })

    return results


# ─────────────────────────────────────────────
# STEP 9: Draw Detection Result
# ─────────────────────────────────────────────

COLOR_MAP = {
    "Pellet":   (0, 255, 100),   # green
    "Fiber":    (255, 80,  80),  # blue
    "Fragment": (0, 180, 255),   # orange-yellow
    "Unknown":  (180, 180, 180),
}


def draw_detections(img, detections):
    """Draw contours + labels on a copy of img."""
    vis = img.copy()
    for d in detections:
        cnt = d["contour"]
        cls = d["classification"]
        color = COLOR_MAP.get(cls, (200, 200, 200))
        cv2.drawContours(vis, [cnt], -1, color, 2)

        # Label near centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(vis, cls[0], (cx - 4, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return vis


# ─────────────────────────────────────────────
# STEP 10: Contamination Score
# ─────────────────────────────────────────────

def compute_contamination(detections, mask):
    """
    score = (microplastic_area / total_visible_area) * 100
    """
    total_area = int(np.sum(mask > 0))
    if total_area == 0:
        return 0.0, "Low"

    plastic_area = sum(d["area"] for d in detections)
    score = (plastic_area / total_area) * 100
    score = min(score, 100.0)

    if score < 20:
        level = "Low"
    elif score < 50:
        level = "Medium"
    else:
        level = "High"

    return round(score, 3), level


# ─────────────────────────────────────────────
# FULL PIPELINE (Single Image)
# ─────────────────────────────────────────────

def run_pipeline(b64_input):
    """
    Run the full DIP pipeline on a base64-encoded input image.

    Returns a dict with:
        steps        – dict of {step_name: base64_png}
        detections   – list of detection dicts (without raw contour)
        count        – int
        score        – float
        level        – str
        class_counts – dict {Fiber, Fragment, Pellet}
    """
    original = b64_to_img(b64_input)

    # 1. Detect & crop circle
    cropped, mask, circle_params = detect_and_crop_circle(original)
    masked_crop = apply_circle_mask(cropped, mask)

    # 2. Enhancement
    enhanced = enhance_image(masked_crop)

    # 3. Histogram equalization
    hist_eq = histogram_equalization(enhanced)

    # 4. Noise removal
    denoised = remove_noise(hist_eq)

    # 5. Spatial filtering
    filtered = spatial_filtering(denoised)

    # 6. Thresholding
    binary = threshold_image(filtered, mask)

    # 7. Morphology
    morph = morphological_ops(binary)

    # 8. Contour detection
    detections = detect_contours(morph)

    # 9. Draw result
    result_vis = draw_detections(masked_crop, detections)

    # 10. Contamination score
    score, level = compute_contamination(detections, mask)

    # Classification counts
    class_counts = {"Fiber": 0, "Fragment": 0, "Pellet": 0, "Unknown": 0}
    for d in detections:
        class_counts[d["classification"]] = class_counts.get(d["classification"], 0) + 1

    # Serialisable detection list (remove raw contour numpy array)
    det_list = [
        {
            "classification": d["classification"],
            "area": round(d["area"], 1),
            "circularity": round(d["circularity"], 3),
        }
        for d in detections
    ]

    # Build overlay for thresholding step (colorise binary)
    binary_bgr = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)

    steps = {
        "cropped_roi":  img_to_b64(masked_crop),
        "enhancement":  img_to_b64(enhanced),
        "histogram":    img_to_b64(hist_eq),
        "noise_removal": img_to_b64(denoised),
        "spatial":      img_to_b64(filtered),
        "threshold":    img_to_b64(binary_bgr),
        "morphology":   img_to_b64(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)),
        "final":        img_to_b64(result_vis),
    }

    return {
        "steps": steps,
        "detections": det_list,
        "count": len(detections),
        "score": score,
        "level": level,
        "class_counts": class_counts,
    }
