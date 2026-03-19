"""
TCG Card Grading Engine — Computer Vision pipeline.

Performs PSA-style grading using pure OpenCV analysis:
  - Card detection & isolation
  - Centering measurement (L/R and T/B ratios)
  - Corner sharpness analysis (all 4 corners)
  - Edge straightness & damage detection
  - Surface defect detection (scratches, print lines, whitening)
  - Color consistency & print quality

Returns objective numeric grades (1-10 scale) for each sub-category
plus an overall composite grade.
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CornerDetail:
    top_left: float
    top_right: float
    bottom_left: float
    bottom_right: float
    average: float


@dataclass
class EdgeDetail:
    top: float
    right: float
    bottom: float
    left: float
    average: float


@dataclass
class CenteringDetail:
    left_right_ratio: str    # e.g. "52/48"
    top_bottom_ratio: str    # e.g. "50/50"
    lr_offset_pct: float     # how far off from 50/50
    tb_offset_pct: float
    grade: float


@dataclass
class SurfaceDetail:
    scratches: int
    print_defects: int
    whitening_areas: int
    color_uniformity: float  # 0-1
    grade: float


@dataclass
class GradeResult:
    overall: float
    centering: CenteringDetail
    corners: CornerDetail
    edges: EdgeDetail
    surface: SurfaceDetail
    confidence: float
    card_detected: bool
    card_width: int
    card_height: int
    defects: list
    raw_metrics: dict


# ---------------------------------------------------------------------------
# Card detection
# ---------------------------------------------------------------------------

def detect_card(img: np.ndarray) -> Optional[np.ndarray]:
    """Detect and extract the card region from the image."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to handle varying lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    # Also try Canny edge detection
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_area = 0
    min_area = h * w * 0.1  # Card should be at least 10% of image

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and area > best_area:
            best_contour = approx
            best_area = area

    # Fallback: try finding largest rectangular-ish contour
    if best_contour is None:
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if 4 <= len(approx) <= 6:
                best_contour = approx
                best_area = area
                break

    if best_contour is None:
        return None

    return best_contour


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left

    return rect


def warp_card(img: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Perspective-transform the card to a flat rectangle."""
    pts = order_points(contour.reshape(-1, 2).astype("float32"))

    # Standard card aspect ratio ~2.5 x 3.5 inches
    card_w, card_h = 500, 700

    dst = np.array([
        [0, 0],
        [card_w - 1, 0],
        [card_w - 1, card_h - 1],
        [0, card_h - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (card_w, card_h))
    return warped


def detect_slab(img: np.ndarray) -> bool:
    """Detect if the image shows a graded card inside a slab holder."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sample the outer border pixels (10% band on each side)
    top_band    = gray[0:h//10, :]
    bottom_band = gray[9*h//10:h, :]
    left_band   = gray[:, 0:w//10]
    right_band  = gray[:, 9*w//10:w]
    border_mean = np.mean([top_band.mean(), bottom_band.mean(),
                           left_band.mean(), right_band.mean()])
    center_crop = gray[h//4:3*h//4, w//4:3*w//4]
    center_mean = center_crop.mean()
    # Slab images have a bright/uniform outer border and a complex inner card
    border_is_bright = border_mean > 200
    center_is_complex = center_crop.std() > 30
    return bool(border_is_bright and center_is_complex)


def crop_inner_card(img: np.ndarray) -> np.ndarray:
    """For slab images: crop out the uniform slab border to isolate the card."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the largest non-white contiguous region
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fall back: crop 12% from each side
        pad_y = int(h * 0.12)
        pad_x = int(w * 0.12)
        return img[pad_y:h-pad_y, pad_x:w-pad_x]
    # Use the bounding box of the largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    # Add small inset to avoid slab border artefacts
    inset_x = max(0, int(cw * 0.04))
    inset_y = max(0, int(ch * 0.04))
    x1 = max(0, x + inset_x)
    y1 = max(0, y + inset_y)
    x2 = min(w, x + cw - inset_x)
    y2 = min(h, y + ch - inset_y)
    if x2 - x1 < 50 or y2 - y1 < 50:
        pad_y = int(h * 0.12)
        pad_x = int(w * 0.12)
        return img[pad_y:h-pad_y, pad_x:w-pad_x]
    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Centering analysis
# ---------------------------------------------------------------------------

def analyze_centering(card: np.ndarray) -> CenteringDetail:
    """Measure card centering by detecting the border widths."""
    h, w = card.shape[:2]
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    # Convert to binary to find the printed area vs border
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the bounding box of the non-border content
    # Sample rows/cols to find where content starts
    border_threshold = 220  # Near-white = border

    def find_border(line, from_start=True):
        """Find where border ends in a 1D intensity profile."""
        if not from_start:
            line = line[::-1]
        for i, val in enumerate(line):
            if val < border_threshold:
                return i
        return len(line) // 2

    # Sample multiple rows/columns for robustness
    sample_rows = [int(h * r) for r in [0.3, 0.4, 0.5, 0.6, 0.7]]
    sample_cols = [int(w * c) for c in [0.3, 0.4, 0.5, 0.6, 0.7]]

    lefts, rights, tops, bottoms = [], [], [], []

    for r in sample_rows:
        row = gray[r, :]
        lefts.append(find_border(row, True))
        rights.append(find_border(row, False))

    for c in sample_cols:
        col = gray[:, c]
        tops.append(find_border(col, True))
        bottoms.append(find_border(col, False))

    left_border = int(np.median(lefts))
    right_border = int(np.median(rights))
    top_border = int(np.median(tops))
    bottom_border = int(np.median(bottoms))

    # Avoid division by zero
    lr_total = max(left_border + right_border, 1)
    tb_total = max(top_border + bottom_border, 1)

    lr_left_pct = round(left_border / lr_total * 100)
    lr_right_pct = 100 - lr_left_pct
    tb_top_pct = round(top_border / tb_total * 100)
    tb_bottom_pct = 100 - tb_top_pct

    lr_offset = abs(50 - lr_left_pct)
    tb_offset = abs(50 - tb_top_pct)

    # Grade: perfect centering (50/50) = 10, each % off reduces score
    # PSA: 55/45 is about a 9, 60/40 is about a 7, 65/35 is about a 5
    max_offset = max(lr_offset, tb_offset)
    if max_offset <= 2:
        grade = 10.0
    elif max_offset <= 5:
        grade = 9.5 - (max_offset - 2) * 0.15
    elif max_offset <= 10:
        grade = 9.0 - (max_offset - 5) * 0.3
    elif max_offset <= 15:
        grade = 7.5 - (max_offset - 10) * 0.3
    elif max_offset <= 20:
        grade = 6.0 - (max_offset - 15) * 0.3
    else:
        grade = max(1.0, 4.5 - (max_offset - 20) * 0.15)

    return CenteringDetail(
        left_right_ratio=f"{lr_left_pct}/{lr_right_pct}",
        top_bottom_ratio=f"{tb_top_pct}/{tb_bottom_pct}",
        lr_offset_pct=lr_offset,
        tb_offset_pct=tb_offset,
        grade=round(grade, 1),
    )


# ---------------------------------------------------------------------------
# Corner analysis
# ---------------------------------------------------------------------------

def analyze_corners(card: np.ndarray) -> CornerDetail:
    """Analyze all 4 corners for sharpness, wear, and rounding."""
    h, w = card.shape[:2]
    corner_size = int(min(h, w) * 0.1)  # ~10% of card for each corner

    corners = {
        "top_left": card[0:corner_size, 0:corner_size],
        "top_right": card[0:corner_size, w - corner_size:w],
        "bottom_left": card[h - corner_size:h, 0:corner_size],
        "bottom_right": card[h - corner_size:h, w - corner_size:w],
    }

    grades = {}
    for name, region in corners.items():
        grades[name] = grade_corner(region)

    avg = round(sum(grades.values()) / 4, 1)

    return CornerDetail(
        top_left=grades["top_left"],
        top_right=grades["top_right"],
        bottom_left=grades["bottom_left"],
        bottom_right=grades["bottom_right"],
        average=avg,
    )


def grade_corner(region: np.ndarray) -> float:
    """Grade a single corner region (10 = perfect, 1 = destroyed)."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Edge sharpness via Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    # 2. Corner detection — sharp corners have strong Harris response
    corners_detected = cv2.cornerHarris(gray, 2, 3, 0.04)
    corner_strength = corners_detected.max()

    # 3. Check for whitening (wear shows as lighter pixels at the very corner)
    # Sample the outermost corner pixels
    corner_pixel_region = gray[0:max(1, h // 4), 0:max(1, w // 4)]
    inner_region = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]

    corner_brightness = corner_pixel_region.mean() if corner_pixel_region.size > 0 else 128
    inner_brightness = inner_region.mean() if inner_region.size > 0 else 128

    whitening = max(0, corner_brightness - inner_brightness)

    # 4. Check roundness of the corner
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / (h * w * 255)

    # Scoring
    score = 10.0

    # Sharpness: low variance = blurry/worn corner
    if sharpness < 50:
        score -= 3.0
    elif sharpness < 100:
        score -= 2.0
    elif sharpness < 200:
        score -= 1.0
    elif sharpness < 400:
        score -= 0.5

    # Whitening penalty
    if whitening > 30:
        score -= 3.0
    elif whitening > 20:
        score -= 2.0
    elif whitening > 10:
        score -= 1.0
    elif whitening > 5:
        score -= 0.5

    # Edge density: too few edges = rounded/worn, too many = damaged
    if edge_density < 0.05:
        score -= 1.0
    elif edge_density > 0.4:
        score -= 0.5

    return round(max(1.0, min(10.0, score)), 1)


# ---------------------------------------------------------------------------
# Edge analysis
# ---------------------------------------------------------------------------

def analyze_edges(card: np.ndarray) -> EdgeDetail:
    """Analyze all 4 edges for straightness, nicks, and whitening."""
    h, w = card.shape[:2]
    edge_width = int(min(h, w) * 0.04)  # ~4% strip along each edge

    edges = {
        "top": card[0:edge_width, edge_width:w - edge_width],
        "bottom": card[h - edge_width:h, edge_width:w - edge_width],
        "left": card[edge_width:h - edge_width, 0:edge_width],
        "right": card[edge_width:h - edge_width, w - edge_width:w],
    }

    grades = {}
    for name, region in edges.items():
        grades[name] = grade_edge(region, name in ("left", "right"))

    avg = round(sum(grades.values()) / 4, 1)

    return EdgeDetail(
        top=grades["top"],
        right=grades["right"],
        bottom=grades["bottom"],
        left=grades["left"],
        average=avg,
    )


def grade_edge(region: np.ndarray, is_vertical: bool) -> float:
    """Grade a single edge strip."""
    if region.size == 0:
        return 5.0

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Edge straightness — detect the card boundary line
    edges = cv2.Canny(gray, 50, 150)

    # Use HoughLines to find straight lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                            minLineLength=min(h, w) // 3, maxLineGap=5)

    straightness_score = 10.0
    if lines is not None and len(lines) > 0:
        # Check how straight the dominant line is
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
            if is_vertical:
                angle = abs(angle - 90)
            angles.append(angle)

        avg_deviation = np.mean(angles) if angles else 0
        if avg_deviation > 5:
            straightness_score -= 2.0
        elif avg_deviation > 2:
            straightness_score -= 1.0
        elif avg_deviation > 1:
            straightness_score -= 0.5
    else:
        straightness_score -= 1.0  # Can't detect clear edge = slight concern

    # 2. Whitening detection along edge
    edge_strip = gray[:, 0:max(1, w // 3)] if not is_vertical else gray[0:max(1, h // 3), :]
    inner_strip = gray[:, w // 3:2 * w // 3] if not is_vertical else gray[h // 3:2 * h // 3, :]

    edge_brightness = edge_strip.mean() if edge_strip.size > 0 else 128
    inner_brightness = inner_strip.mean() if inner_strip.size > 0 else 128
    whitening = max(0, edge_brightness - inner_brightness)

    if whitening > 25:
        straightness_score -= 2.5
    elif whitening > 15:
        straightness_score -= 1.5
    elif whitening > 8:
        straightness_score -= 0.5

    # 3. Nick/dent detection — sudden intensity changes along the edge
    profile = gray.mean(axis=0) if not is_vertical else gray.mean(axis=1)
    if len(profile) > 10:
        diffs = np.abs(np.diff(profile.astype(float)))
        nicks = np.sum(diffs > 15)
        if nicks > 10:
            straightness_score -= 2.0
        elif nicks > 5:
            straightness_score -= 1.0
        elif nicks > 2:
            straightness_score -= 0.5

    return round(max(1.0, min(10.0, straightness_score)), 1)


# ---------------------------------------------------------------------------
# Surface analysis
# ---------------------------------------------------------------------------

def analyze_surface(card: np.ndarray) -> SurfaceDetail:
    """Analyze the card surface for scratches, defects, and print quality."""
    h, w = card.shape[:2]
    margin = int(min(h, w) * 0.08)
    interior = card[margin:h - margin, margin:w - margin]

    gray = cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape

    # 1. Scratch detection — long thin bright/dark lines
    # Use morphological operations to isolate linear defects
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    blackhat_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
    blackhat_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_v)

    scratch_mask = cv2.add(blackhat_h, blackhat_v)
    _, scratch_binary = cv2.threshold(scratch_mask, 30, 255, cv2.THRESH_BINARY)
    scratch_count = cv2.countNonZero(scratch_binary)
    scratch_ratio = scratch_count / max(1, ih * iw)

    # Count distinct scratches
    contours_s, _ = cv2.findContours(scratch_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scratches = len([c for c in contours_s if cv2.contourArea(c) > 20])

    # 2. Print defect detection — blobs of wrong color
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    diff = cv2.absdiff(gray, blur)
    _, defect_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    contours_d, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print_defects = len([c for c in contours_d if 30 < cv2.contourArea(c) < 2000])

    # 3. Whitening / fading areas
    hsv = cv2.cvtColor(interior, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    low_sat_mask = (saturation < 20).astype(np.uint8) * 255
    # Exclude naturally white areas by checking if value is also high
    value = hsv[:, :, 2]
    whitening_mask = cv2.bitwise_and(low_sat_mask, (value > 215).astype(np.uint8) * 255)
    whitening_ratio = cv2.countNonZero(whitening_mask) / max(1, ih * iw)

    contours_w, _ = cv2.findContours(whitening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    whitening_areas = len([c for c in contours_w if cv2.contourArea(c) > 50])

    # 4. Color uniformity — std dev of color channels in smooth regions
    b, g, r = cv2.split(interior)
    color_std = np.mean([b.std(), g.std(), r.std()])
    # Normalize: lower std = more uniform (but cards have varied content, so be lenient)
    color_uniformity = max(0, min(1.0, 1.0 - (color_std - 40) / 100))

    # Compute surface grade
    grade = 10.0

    # Scratches
    if scratches > 20:
        grade -= 4.0
    elif scratches > 10:
        grade -= 2.5
    elif scratches > 5:
        grade -= 1.5
    elif scratches > 2:
        grade -= 0.5

    # Print defects
    if print_defects > 15:
        grade -= 3.0
    elif print_defects > 8:
        grade -= 2.0
    elif print_defects > 3:
        grade -= 1.0
    elif print_defects > 0:
        grade -= 0.5

    # Whitening
    if whitening_ratio > 0.15:
        grade -= 3.0
    elif whitening_ratio > 0.08:
        grade -= 2.0
    elif whitening_ratio > 0.03:
        grade -= 1.0

    return SurfaceDetail(
        scratches=scratches,
        print_defects=print_defects,
        whitening_areas=whitening_areas,
        color_uniformity=round(color_uniformity, 3),
        grade=round(max(1.0, min(10.0, grade)), 1),
    )


# ---------------------------------------------------------------------------
# Main grading function
# ---------------------------------------------------------------------------

def grade_card_image(img_data: bytes) -> GradeResult:
    """
    Grade a card image from raw bytes.
    Returns comprehensive grading result.
    """
    # Decode image
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return GradeResult(
            overall=0, centering=CenteringDetail("0/0", "0/0", 0, 0, 0),
            corners=CornerDetail(0, 0, 0, 0, 0), edges=EdgeDetail(0, 0, 0, 0, 0),
            surface=SurfaceDetail(0, 0, 0, 0, 0), confidence=0, card_detected=False,
            card_width=0, card_height=0, defects=["Could not decode image"], raw_metrics={},
        )

    h, w = img.shape[:2]

    # Detect slab and extract inner card first
    is_slab = detect_slab(img)
    if is_slab:
        img = crop_inner_card(img)

    # Try to detect and extract card
    contour = detect_card(img)
    if contour is not None and len(contour) == 4:
        card = warp_card(img, contour)
        card_detected = True
    else:
        # Fallback: use the whole image (assume it's cropped to the card)
        card = cv2.resize(img, (500, 700))
        card_detected = contour is not None

    ch, cw = card.shape[:2]

    # Run all analyses
    centering = analyze_centering(card)
    corners = analyze_corners(card)
    edges = analyze_edges(card)
    surface = analyze_surface(card)

    # Compute overall grade (PSA weighting)
    # PSA weights: Centering ~15%, Corners ~25%, Edges ~25%, Surface ~35%
    overall = (
        centering.grade * 0.15 +
        corners.average * 0.25 +
        edges.average * 0.25 +
        surface.grade * 0.35
    )

    # Round to nearest 0.5 (PSA style)
    overall = round(overall * 2) / 2
    overall = max(1.0, min(10.0, overall))

    # Confidence based on card detection quality
    confidence = 0.85 if card_detected else 0.5
    # Boost confidence if all sub-grades are close
    grade_spread = max(centering.grade, corners.average, edges.average, surface.grade) - \
                   min(centering.grade, corners.average, edges.average, surface.grade)
    if grade_spread < 1.5:
        confidence += 0.1
    confidence = min(0.95, confidence)

    # Collect defect list
    defects = []
    if centering.lr_offset_pct > 5:
        defects.append(f"Off-center L/R: {centering.left_right_ratio}")
    if centering.tb_offset_pct > 5:
        defects.append(f"Off-center T/B: {centering.top_bottom_ratio}")
    if corners.average < 7:
        worst = min(corners.top_left, corners.top_right, corners.bottom_left, corners.bottom_right)
        defects.append(f"Corner wear detected (worst: {worst})")
    if edges.average < 7:
        defects.append("Edge wear or damage detected")
    if surface.scratches > 2:
        defects.append(f"{surface.scratches} surface scratches detected")
    if surface.print_defects > 2:
        defects.append(f"{surface.print_defects} print defects detected")
    if surface.whitening_areas > 3:
        defects.append(f"Surface whitening detected ({surface.whitening_areas} areas)")

    raw_metrics = {
        "image_size": f"{w}x{h}",
        "card_size": f"{cw}x{ch}",
        "card_detected": card_detected,
        "is_slab": is_slab,
    }

    return GradeResult(
        overall=overall,
        centering=centering,
        corners=corners,
        edges=edges,
        surface=surface,
        confidence=round(confidence, 2),
        card_detected=card_detected,
        card_width=cw,
        card_height=ch,
        defects=defects,
        raw_metrics=raw_metrics,
    )


def grade_card_file(path: str) -> GradeResult:
    """Grade a card from a file path."""
    with open(path, "rb") as f:
        return grade_card_image(f.read())
