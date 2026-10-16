"""Image preprocessing for mobile-captured financial documents (Objective 2.1, Gap 3).

Why this module exists:
- Phone photos of payslips and bills are skewed, unevenly lit and noisy; raw OCR on
  them is unreliable. Each step below targets one capture artefact.
- Every step can be switched on or off, so the benchmark can measure what each one
  contributes (preprocessing ablation, experiment E1).
- Every geometric step is recorded as a 3x3 homography, so bounding boxes found on the
  cleaned image can be mapped back to the ORIGINAL photo (contract rule: bbox is in
  original-image pixels; Gap 7 traceability).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import cv2
import numpy as np

ALL_STEPS: Tuple[str, ...] = ("denoise", "resize", "perspective", "deskew", "clahe", "threshold")
DEFAULT_STEPS: Tuple[str, ...] = ALL_STEPS


@dataclass
class PreprocessResult:
    image: np.ndarray                     # image to pass to OCR (gray or binary)
    homography: np.ndarray                # 3x3, maps ORIGINAL pixel -> processed pixel
    steps_applied: List[str] = field(default_factory=list)
    skew_angle: float = 0.0               # degrees corrected by deskew (+ = counter-clockwise)
    original_shape: Tuple[int, int] = (0, 0)

    def to_original_bbox(self, bbox: Sequence[int]) -> List[int]:
        """Map [x, y, w, h] on the processed image back to the original image."""
        return map_bbox(bbox, np.linalg.inv(self.homography), self.original_shape)


# ---------------------------------------------------------------- geometry helpers

def map_bbox(bbox: Sequence[int], H: np.ndarray, clip_shape: Tuple[int, int] | None = None) -> List[int]:
    """Transform the 4 corners of [x, y, w, h] with H and return their bounding box."""
    x, y, w, h = bbox
    pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    x0, y0 = out.min(axis=0)
    x1, y1 = out.max(axis=0)
    if clip_shape is not None:
        hh, ww = clip_shape
        x0, x1 = np.clip([x0, x1], 0, ww)
        y0, y1 = np.clip([y0, y1], 0, hh)
    return [int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))]


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------- individual steps

def resize_long_side(gray: np.ndarray, target: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Bring every image to a similar resolution so later parameters behave the same.
    Tesseract works best at roughly 300 dpi; a 2000 px long side is close for A4/A5."""
    h, w = gray.shape[:2]
    s = target / max(h, w)
    if abs(s - 1) < 0.05:
        return gray, np.eye(3)
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    out = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=interp)
    return out, np.diag([s, s, 1.0])


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _edge_support(edges: np.ndarray, quad: np.ndarray, samples: int = 400) -> float:
    """Share of points along the quad outline that lie on an edge. A real page border is a
    long continuous line (high support); the outline of a block of text is not."""
    pts = np.vstack([quad, quad[:1]])
    hits, total = 0, 0
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, samples // 4, endpoint=False):
            x, y = (a + t * (b - a)).astype(int)
            total += 1
            if 0 <= y < edges.shape[0] and 0 <= x < edges.shape[1] and edges[y, x]:
                hits += 1
    return hits / max(total, 1)


def correct_perspective(gray: np.ndarray, min_area_ratio: float = 0.3,
                        min_edge_support: float = 0.6) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Find the page outline (largest convex 4-corner contour backed by a continuous edge)
    and warp it to a flat rectangle. Returns (image, H, applied). Skips safely when no
    clear page edge is visible (scans, or photos already cropped to the page)."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.dilate(cv2.Canny(blur, 50, 150), np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < min_area_ratio * h * w:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        src = _order_corners(approx)
        if _edge_support(edges, src) < min_edge_support:
            continue  # outline of a text block, not a page border
        tl, tr, br, bl = src
        W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        Hh = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if W < 100 or Hh < 100 or (W > 0.97 * w and Hh > 0.97 * h):
            continue  # degenerate, or the "page" is just the image border
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, Hh - 1], [0, Hh - 1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(gray, H, (W, Hh), flags=cv2.INTER_CUBIC, borderValue=255), H, True
    return gray, np.eye(3), False


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Text pixels = 255. A LOCAL threshold is used because a global one (Otsu) marks
    whole shadowed regions of a phone photo as ink, which breaks skew and page detection."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ink = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # drop speckle noise


def estimate_skew(gray: np.ndarray, max_angle: float = 20.0) -> float:
    """Projection-profile skew estimate: text lines give the sharpest row-sum profile
    when they are horizontal. Coarse search then fine search. Returns degrees to rotate
    counter-clockwise to straighten the text (0.0 if there is too little text)."""
    small = gray
    if max(gray.shape) > 1000:
        s = 1000 / max(gray.shape)
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ink = _ink_mask(small)
    if ink.mean() < 0.5:          # almost no text pixels: nothing reliable to measure
        return 0.0
    h, w = ink.shape
    centre = (w / 2, h / 2)

    def score(a: float) -> float:
        M = cv2.getRotationMatrix2D(centre, a, 1.0)
        rot = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rot.sum(axis=1)))

    coarse = np.arange(-max_angle, max_angle + 0.01, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.01, 0.1)
    return float(round(float(np.clip(max(fine, key=score), -max_angle, max_angle)), 2))


def rotate_bound(gray: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate without cropping corners (canvas grows). New corner areas are filled with the
    image's median brightness (its typical background) rather than pure white, so no strong
    seam is created next to the text that Tesseract could read as a stroke. (Replicating
    edge pixels was tried and rejected: it creates streaks that are read as dashes.)"""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    fill = int(np.median(gray))
    out = cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=fill)
    return out, np.vstack([M, [0, 0, 1]])


def estimate_noise(gray: np.ndarray) -> float:
    """Noise standard deviation (Immerkaer, 1996): a Laplacian-like kernel cancels smooth
    image content, so what remains is mostly noise. Used to set the denoise strength."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    resp = cv2.filter2D(gray.astype(np.float64), -1, k)
    h, w = gray.shape
    return float(np.sqrt(np.pi / 2) * np.abs(resp[1:-1, 1:-1]).sum() / (6 * (w - 2) * (h - 2)))


def denoise(gray: np.ndarray, strength: float | None = None) -> np.ndarray:
    """Non-local means removes sensor noise and JPEG blocks while keeping stroke edges,
    which matters for thin Sinhala/Tamil strokes. Strength follows the measured noise
    (strength=None) so clean scans are barely touched and dark photos are cleaned harder."""
    if strength is None:
        strength = float(np.clip(1.1 * estimate_noise(gray), 3, 30))
    return cv2.fastNlMeansDenoising(gray, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def apply_clahe(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Local contrast normalisation: evens out shadows and dim areas from phone lighting."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)


def adaptive_threshold(gray: np.ndarray, block: int = 31, c: int = 15) -> np.ndarray:
    """Local (Gaussian) threshold: one global threshold fails under uneven lighting.
    block must be odd; larger blocks suit larger text."""
    block = block if block % 2 else block + 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)


def remove_specks(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Delete tiny black blobs left by noise. Tesseract slows down and invents characters
    when the background is speckled. min_area is scaled to resolution by the caller so
    full stops and decimal points survive."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(255 - binary, connectivity=8)
    small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
    out = binary.copy()
    out[np.isin(labels, small)] = 255
    return out


# ---------------------------------------------------------------- pipeline

def preprocess(img: np.ndarray, steps: Sequence[str] = DEFAULT_STEPS, *, target_long_side: int = 2000,
               denoise_strength: float | None = None, threshold_block: int = 31, threshold_c: int = 15) -> PreprocessResult:
    """Run the selected steps in a fixed order. Unknown step names raise ValueError so a
    typo in a benchmark config cannot silently skip a step."""
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise ValueError(f"unknown preprocessing steps: {sorted(unknown)}")
    gray = _to_gray(img)
    res = PreprocessResult(image=gray, homography=np.eye(3), original_shape=gray.shape[:2])
    H = np.eye(3)

    # Denoise FIRST, at the photo's own resolution: upscaling noisy pixels turns them into
    # blobs that later look like ink. Denoising does not move pixels, so H is unchanged.
    if "denoise" in steps:
        gray = denoise(gray, denoise_strength)
        res.steps_applied.append("denoise")
    if "resize" in steps:
        gray, Hs = resize_long_side(gray, target_long_side)
        H = Hs @ H
        res.steps_applied.append("resize")
    if "perspective" in steps:
        gray, Hp, ok = correct_perspective(gray)
        if ok:
            H = Hp @ H
            res.steps_applied.append("perspective")
    if "deskew" in steps:
        angle = estimate_skew(gray)
        if abs(angle) >= 0.3:
            gray, Hr = rotate_bound(gray, angle)
            H = Hr @ H
            res.skew_angle = angle
            res.steps_applied.append("deskew")
    if "clahe" in steps:
        gray = apply_clahe(gray)
        res.steps_applied.append("clahe")
    if "threshold" in steps:
        gray = adaptive_threshold(gray, threshold_block, threshold_c)
        speck = max(4, int(round((max(gray.shape) / 1000) ** 2 * 6)))
        gray = remove_specks(gray, speck)
        res.steps_applied.append("threshold")

    res.image, res.homography = gray, H
    return res
