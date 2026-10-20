"""Why: preprocessing changes geometry; these tests guarantee skew is corrected and that
boxes found on the cleaned image still point at the right place in the original photo."""
import cv2
import numpy as np
import pytest

from preprocessing import ALL_STEPS, map_bbox, preprocess
from preprocessing.pipeline import correct_perspective, estimate_noise, estimate_skew, rotate_bound

LINES = ["PAY SLIP SEPTEMBER 2026", "Basic Salary 120,000.00", "Gross Salary 150,000.00",
         "EPF Employee 8% 12,000.00", "APIT 3,500.00", "Net Salary 134,500.00"]


def make_page(w=900, h=700):
    img = np.full((h, w), 255, np.uint8)
    for i, t in enumerate(LINES):
        cv2.putText(img, t, (40, 80 + i * 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    return img


@pytest.mark.parametrize("angle", [-8.0, -3.0, 4.0, 10.0])
def test_estimate_skew_recovers_rotation(angle):
    rotated, _ = rotate_bound(make_page(), -angle)   # tilt the page by `angle`
    est = estimate_skew(rotated)
    assert abs(est - angle) <= 0.6, (angle, est)


def test_bbox_maps_back_to_original():
    page = make_page()
    tilted, _ = rotate_bound(page, 6.0)
    tilted = cv2.cvtColor(tilted, cv2.COLOR_GRAY2BGR)
    res = preprocess(tilted, steps=("resize", "deskew"))
    assert "deskew" in res.steps_applied
    # Map a box forward and back. Its centre must return to the same spot; the box itself
    # grows slightly because each rotation takes the axis-aligned bounding rectangle.
    box = [100, 150, 300, 60]
    back = res.to_original_bbox(map_bbox(box, res.homography))
    centre = lambda b: (b[0] + b[2] / 2, b[1] + b[3] / 2)
    assert np.allclose(centre(back), centre(box), atol=3), (box, back)
    assert back[2] >= box[2] - 2 and back[3] >= box[3] - 2


def test_threshold_output_is_binary():
    res = preprocess(make_page(), steps=("threshold",))
    assert set(np.unique(res.image)) <= {0, 255}


def test_no_steps_is_identity():
    page = make_page()
    res = preprocess(page, steps=())
    assert res.steps_applied == [] and np.array_equal(res.image, page)
    assert np.allclose(res.homography, np.eye(3))


def test_unknown_step_rejected():
    with pytest.raises(ValueError):
        preprocess(make_page(), steps=("deskw",))


def test_full_pipeline_runs_on_colour_photo():
    noisy = cv2.cvtColor(make_page(), cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.7
    noisy = np.clip(noisy + np.random.default_rng(0).normal(0, 15, noisy.shape), 0, 255).astype(np.uint8)
    res = preprocess(noisy, steps=ALL_STEPS)
    assert res.image.ndim == 2 and res.image.dtype == np.uint8


def test_perspective_flattens_page_photographed_at_an_angle():
    page = make_page(600, 800)
    table = np.full((1100, 1000), 60, np.uint8)            # dark desk around the page
    src = np.float32([[0, 0], [599, 0], [599, 799], [0, 799]])
    dst = np.float32([[180, 120], [820, 170], [860, 980], [140, 940]])
    M = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(page, M, (1000, 1100), dst=table.copy(), borderMode=cv2.BORDER_TRANSPARENT)
    flat, H, applied = correct_perspective(photo)
    assert applied
    h, w = flat.shape
    assert 0.6 < w / h < 0.9                                 # portrait page restored (600x800 = 0.75)


def test_perspective_skips_plain_text_without_page_edge():
    _, _, applied = correct_perspective(make_page())
    assert not applied


def test_noise_estimate_increases_with_noise():
    page = make_page().astype(np.float32)
    rng = np.random.default_rng(0)
    low = np.clip(page + rng.normal(0, 3, page.shape), 0, 255).astype(np.uint8)
    high = np.clip(page + rng.normal(0, 20, page.shape), 0, 255).astype(np.uint8)
    assert estimate_noise(high) > estimate_noise(low) + 5


def test_perspective_ignores_ruled_table_on_the_page():
    """A payslip table with a printed border must not be taken as the page: the header
    above it (issuer, name, NIC) would be cropped away before OCR and PII masking."""
    page = np.full((1000, 800), 235, np.uint8)
    cv2.putText(page, "HEADER LINE WITH NAME", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    cv2.rectangle(page, (40, 200), (760, 960), 0, 4)       # ruled table border
    for y in range(260, 940, 60):
        cv2.putText(page, "Label 12,345.00", (70, y), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
    _, _, applied = correct_perspective(page)
    assert not applied
