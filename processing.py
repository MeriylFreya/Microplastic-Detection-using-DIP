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
        is_valid_dish – bool indicating if a petri dish is confidently detected
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blurred, 40, 120)

    # Try Hough Circle detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) * 0.4,
        param1=80,
        param2=34,
        minRadius=int(min(h, w) * 0.20),
        maxRadius=int(min(h, w) * 0.55)
    )

    if circles is None:
        # No circular dish confidently found: return an invalid marker.
        return img.copy(), np.zeros((h, w), dtype=np.uint8), None, False

    circles = np.uint16(np.around(circles))

    best = None
    best_score = -1.0
    for c in circles[0]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        if r <= 0:
            continue

        rr = float(r)
        ring_outer = np.zeros((h, w), dtype=np.uint8)
        ring_inner = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring_outer, (cx, cy), int(rr * 1.03), 255, -1)
        cv2.circle(ring_inner, (cx, cy), int(rr * 0.97), 255, -1)
        ring = cv2.subtract(ring_outer, ring_inner)

        ring_pixels = int(np.sum(ring > 0))
        if ring_pixels == 0:
            continue

        edge_ratio = float(np.sum((edges > 0) & (ring > 0))) / ring_pixels
        center_offset = math.hypot(cx - (w / 2.0), cy - (h / 2.0)) / (min(h, w) / 2.0)
        radius_ratio = rr / float(min(h, w))

        center_score = max(0.0, 1.0 - center_offset)
        radius_score = 1.0 if (0.23 <= radius_ratio <= 0.52) else 0.0
        score = (1.6 * edge_ratio) + (0.8 * center_score) + (0.8 * radius_score)

        if score > best_score:
            best_score = score
            best = (cx, cy, r, edge_ratio, radius_ratio, center_offset)

    if best is None:
        return img.copy(), np.zeros((h, w), dtype=np.uint8), None, False

    cx, cy, r, edge_ratio, radius_ratio, center_offset = best
    is_valid_dish = (
        edge_ratio > 0.04 and
        0.23 <= radius_ratio <= 0.52 and
        center_offset < 0.60
    )

    if not is_valid_dish:
        return img.copy(), np.zeros((h, w), dtype=np.uint8), None, False

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

    return cropped, mask, (cx, cy, r), True


def build_inner_analysis_mask(mask, border_ratio=0.10):
    """Shrink the dish mask to exclude the rim and wall reflections."""
    if mask is None or mask.size == 0:
        return mask

    min_dim = min(mask.shape[:2])
    kernel_size = max(7, int(min_dim * border_ratio))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inner = cv2.erode(mask, kernel, iterations=1)

    # Fallback to original mask if erosion collapses too aggressively.
    if np.sum(inner > 0) < 0.20 * np.sum(mask > 0):
        return mask.copy()

    return inner


def build_empty_result(img, warning):
    """Return a clean zero-result when the image is not a valid petri-dish sample."""
    h, w = img.shape[:2]
    black = np.zeros((h, w, 3), dtype=np.uint8)
    img_b64 = img_to_b64(img)
    black_b64 = img_to_b64(black)
    return {
        "steps": {
            "cropped_roi": img_b64,
            "enhancement": img_b64,
            "histogram": img_b64,
            "noise_removal": img_b64,
            "spatial": img_b64,
            "threshold": black_b64,
            "morphology": black_b64,
            "final": img_b64,
        },
        "detections": [],
        "count": 0,
        "score": 0.0,
        "level": "Low",
        "class_counts": {"Fiber": 0, "Fragment": 0, "Pellet": 0, "Unknown": 0},
        "warning": warning,
    }


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
    Uses: Otsu + Adaptive + Morphological Gradient + Canny Edge +
    background-compensated detail maps.
    Returns binary image (uint8, 0/255).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize local contrast before thresholding so faint flakes are not lost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

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

    # --- Method 5: Background-compensated detail maps ---
    # Top-hat and black-hat boost small bright or dark flakes against uneven backgrounds.
    kernel_detail = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_detail)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_detail)
    detail = cv2.max(tophat, blackhat)
    detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)
    _, detail_thresh = cv2.threshold(detail, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Small local deviations from a median-smoothed background often capture translucent flakes.
    median_bg = cv2.medianBlur(gray, 5)
    deviation = cv2.absdiff(gray, median_bg)
    _, deviation_thresh = cv2.threshold(deviation, 8, 255, cv2.THRESH_BINARY)

    # Small bright/dark square-like particles are often low-contrast and tiny.
    # A local box-blur residual highlights these micro-features.
    local_bg = cv2.blur(gray, (7, 7))
    micro_detail = cv2.absdiff(gray, local_bg)
    _, micro_thresh = cv2.threshold(micro_detail, 7, 255, cv2.THRESH_BINARY)

    # --- Combine all methods ---
    combined = cv2.bitwise_or(otsu, adaptive)
    combined = cv2.bitwise_or(combined, grad_thresh)
    combined = cv2.bitwise_or(combined, edges_dilated)
    combined = cv2.bitwise_or(combined, detail_thresh)
    combined = cv2.bitwise_or(combined, deviation_thresh)
    combined = cv2.bitwise_or(combined, micro_thresh)

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


def build_essential_roi_mask(img_shape, margin_ratio=0.08):
    """Build a central ROI mask to ignore non-essential outer image regions."""
    h, w = img_shape[:2]
    cx, cy = w // 2, h // 2

    ax = max(20, int((w * (1.0 - margin_ratio * 2)) / 2))
    ay = max(20, int((h * (1.0 - margin_ratio * 2)) / 2))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return mask


def remove_border_components(binary_img, border_px=12, min_area_keep=6):
    """Remove connected components touching the outer border area."""
    h, w = binary_img.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    cleaned = np.zeros_like(binary_img)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_keep:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])

        touches_border = (
            x <= border_px or
            y <= border_px or
            (x + ww) >= (w - border_px) or
            (y + hh) >= (h - border_px)
        )
        if touches_border:
            continue

        cleaned[labels == label] = 255

    return cleaned


def filter_detections(detections, analysis_mask, border_px=12):
    """Filter detections outside the essential ROI or touching image borders."""
    h, w = analysis_mask.shape[:2]
    filtered = []

    for d in detections:
        cnt = d["contour"]
        x, y, ww, hh = cv2.boundingRect(cnt)

        if x <= border_px or y <= border_px or (x + ww) >= (w - border_px) or (y + hh) >= (h - border_px):
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue
        if analysis_mask[cy, cx] == 0:
            continue

        component_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(component_mask, [cnt], -1, 255, -1)
        inside_pixels = int(np.sum((component_mask > 0) & (analysis_mask > 0)))
        total_pixels = int(np.sum(component_mask > 0))
        if total_pixels == 0:
            continue

        overlap_ratio = inside_pixels / float(total_pixels)
        if overlap_ratio < 0.90:
            continue

        filtered.append(d)

    return filtered


def fallback_particle_mask(img, mask=None):
    """
    Build a more sensitive binary mask for faint, bright, or translucent particles.
    This is used only when the main pipeline returns no detections.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)

    _, bright = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, dark = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    detail = cv2.max(tophat, blackhat)
    detail = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)
    _, detail_thresh = cv2.threshold(detail, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    median_bg = cv2.medianBlur(gray, 7)
    deviation = cv2.absdiff(gray, median_bg)
    _, deviation_thresh = cv2.threshold(deviation, 8, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(blur, 15, 60)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    combined = cv2.bitwise_or(bright, dark)
    combined = cv2.bitwise_or(combined, edges)
    combined = cv2.bitwise_or(combined, detail_thresh)
    combined = cv2.bitwise_or(combined, deviation_thresh)

    if mask is not None:
        combined = cv2.bitwise_and(combined, mask)

    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)

    return combined


def last_resort_particle_detections(img, mask=None):
    """
    Extract tiny visible blobs using connected components on a loose foreground map.
    This is intentionally sensitive and only used when the regular passes undercount.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)

    bg = cv2.medianBlur(gray, 9)
    diff = cv2.absdiff(gray, bg)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # Slightly boost bright/dark specks independently.
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    detail = cv2.max(diff, cv2.max(tophat, blackhat))

    _, binary = cv2.threshold(detail, 10, 255, cv2.THRESH_BINARY)
    if mask is not None:
        binary = cv2.bitwise_and(binary, mask)

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    detections = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 3:
            continue

        component = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        cls = classify_particle(cnt)
        if cls == "Unknown":
            cls = "Fragment"

        detections.append({
            "contour": cnt,
            "classification": cls,
            "area": float(area),
            "circularity": compute_circularity(cnt),
        })

    return detections, binary


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
    if area < 2:
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


def detect_contours(binary_img, min_area=4, max_area=None):
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
        if perimeter < 4:  # too small to be a real object
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


def estimate_confidence(area, circularity, image_area):
    """Estimate a YOLO-like confidence score from contour shape and size."""
    if image_area <= 0:
        return 0.5

    size_score = min(1.0, max(0.0, area / (image_area * 0.01)))
    shape_score = min(1.0, max(0.0, circularity))
    confidence = 0.35 + (0.35 * size_score) + (0.30 * shape_score)
    return round(max(0.5, min(confidence, 0.99)), 3)


def _detection_center(detection):
    cnt = detection["contour"]
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        x, y, ww, hh = cv2.boundingRect(cnt)
        return (x + ww / 2.0, y + hh / 2.0)
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def merge_detection_sets(*detection_sets, distance_px=10):
    """Merge overlapping detections from multiple passes."""
    merged = []

    for detection_set in detection_sets:
        for detection in detection_set:
            center_x, center_y = _detection_center(detection)
            area = float(detection.get("area", 0.0))

            matched_index = None
            for index, existing in enumerate(merged):
                existing_x, existing_y = _detection_center(existing)
                if math.hypot(center_x - existing_x, center_y - existing_y) <= distance_px:
                    matched_index = index
                    break

            if matched_index is None:
                merged.append(detection)
                continue

            existing = merged[matched_index]
            existing_area = float(existing.get("area", 0.0))
            if area > existing_area:
                merged[matched_index] = detection

    return merged


# ─────────────────────────────────────────────
# STEP 9: Draw Detection Result
# ─────────────────────────────────────────────

COLOR_MAP = {
    "Pellet":   (180, 80, 255),   # purple
    "Fiber":    (180, 80, 255),   # purple
    "Fragment": (180, 80, 255),   # purple
    "Unknown":  (180, 180, 180),
}


def draw_detections(img, detections):
    """Draw bounding boxes + labels on a copy of img."""
    vis = img.copy()
    green = (0, 255, 0)
    black = (0, 0, 0)
    image_area = float(img.shape[0] * img.shape[1])
    
    for d in detections:
        cnt = d["contour"]
        cls = d["classification"]
        confidence = d.get("confidence")
        if confidence is None:
            confidence = estimate_confidence(d.get("area", 0.0), d.get("circularity", 0.0), image_area)
        confidence_text = f"{int(round(confidence * 100))}%"
        
        # Get bounding rectangle
        x, y, ww, hh = cv2.boundingRect(cnt)
        
        # Slight padding so the box is easier to read
        pad = 2
        x = max(0, x - pad)
        y = max(0, y - pad)
        ww = ww + (2 * pad)
        hh = hh + (2 * pad)

        # Draw a clean green box and a single class-confidence label
        cv2.rectangle(vis, (x, y), (x + ww, y + hh), green, 2)

        label = f"{cls[0]} {confidence_text}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        label_y = max(14, y - 6)
        cv2.rectangle(vis, (x, label_y - text_size[1] - 4),
                  (x + text_size[0] + 4, label_y + 4), green, -1)
        cv2.putText(vis, label, (x + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, black, 1, cv2.LINE_AA)
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

    dish_crop, dish_mask, _, is_valid_dish = detect_and_crop_circle(original)
    if is_valid_dish:
        masked_crop = dish_crop
        analysis_mask = build_inner_analysis_mask(dish_mask, border_ratio=0.05)
        border_px = max(4, int(min(masked_crop.shape[:2]) * 0.02))
    else:
        masked_crop = original.copy()
        analysis_mask = np.ones(masked_crop.shape[:2], dtype=np.uint8) * 255
        border_px = 0

    # 2. Enhancement
    enhanced = enhance_image(masked_crop)

    # 3. Histogram equalization
    hist_eq = histogram_equalization(enhanced)

    # 4. Noise removal
    denoised = remove_noise(hist_eq)

    # 5. Spatial filtering
    filtered = spatial_filtering(denoised)

    # 6. Thresholding
    binary = threshold_image(filtered, analysis_mask)

    # 7. Morphology
    morph = morphological_ops(binary)
    if is_valid_dish:
        morph = remove_border_components(morph, border_px=border_px, min_area_keep=6)

    # 8. Contour detection
    detections = detect_contours(morph, min_area=10)
    detections = filter_detections(detections, analysis_mask, border_px=border_px)

    # Only use the more sensitive passes when the primary pass fails completely.
    if len(detections) == 0:
        fallback_binary = fallback_particle_mask(masked_crop, analysis_mask)
        if is_valid_dish:
            fallback_binary = remove_border_components(fallback_binary, border_px=border_px, min_area_keep=6)
        fallback_detections = detect_contours(fallback_binary, min_area=12)
        fallback_detections = filter_detections(fallback_detections, analysis_mask, border_px=border_px)

        if len(fallback_detections) == 0:
            rescue_detections, rescue_binary = last_resort_particle_detections(masked_crop, analysis_mask)
            rescue_detections = filter_detections(rescue_detections, analysis_mask, border_px=border_px)
            if is_valid_dish:
                rescue_binary = remove_border_components(rescue_binary, border_px=border_px, min_area_keep=5)
            rescue_binary_detections = detect_contours(rescue_binary, min_area=12)
            rescue_binary_detections = filter_detections(rescue_binary_detections, analysis_mask, border_px=border_px)
            fallback_detections = merge_detection_sets(rescue_detections, rescue_binary_detections)

        detections = fallback_detections

    # Last attempt: if everything is still zero, relax ROI restrictions so tiny inner-dish
    # particles are not suppressed by conservative masks on challenging dark samples.
    if len(detections) == 0:
        relaxed_mask = np.ones(masked_crop.shape[:2], dtype=np.uint8) * 255
        relaxed_border = 0 if not is_valid_dish else max(2, border_px // 2)

        relaxed_binary = threshold_image(filtered, relaxed_mask)
        relaxed_morph = morphological_ops(relaxed_binary)
        if is_valid_dish:
            relaxed_morph = remove_border_components(relaxed_morph, border_px=relaxed_border, min_area_keep=3)

        relaxed_detections = detect_contours(relaxed_morph, min_area=12)
        relaxed_detections = filter_detections(relaxed_detections, relaxed_mask, border_px=relaxed_border)

        if relaxed_detections:
            detections = relaxed_detections
            morph = relaxed_morph
            analysis_mask = relaxed_mask

    # 9. Draw result
    result_vis = draw_detections(masked_crop, detections)

    # 10. Contamination score
    score, level = compute_contamination(detections, analysis_mask)

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
            "confidence": estimate_confidence(d["area"], d["circularity"], masked_crop.shape[0] * masked_crop.shape[1]),
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


class MicroplasticProcessor:
    """Convenience wrapper for API endpoints and batch processing."""

    def process_image(self, b64_image):
        result = run_pipeline(b64_image)
        return {
            "success": True,
            "total_count": result["count"],
            "shape_distribution": result["class_counts"],
            "contamination_level": result["level"],
            "contamination_score": result["score"],
            "detections": result["detections"],
            "steps": result["steps"],
            "count": result["count"],
            "class_counts": result["class_counts"],
            "level": result["level"],
            "score": result["score"],
        }

    def batch_process(self, b64_images):
        results = []
        for index, b64_image in enumerate(b64_images):
            try:
                result = self.process_image(b64_image)
                result["status"] = "success"
                result["index"] = index
                results.append(result)
            except Exception as exc:
                results.append({
                    "status": "error",
                    "index": index,
                    "error": str(exc),
                })
        return results
