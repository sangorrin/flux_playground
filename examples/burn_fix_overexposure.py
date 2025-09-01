#!/usr/bin/env python3
"""
Photoshop-like AUTO LEVELS for overexposed images (no parameters).

What it does:
  1) For each RGB channel (in sRGB space), find in_black = P1 and in_white = P99.5.
  2) Levels transform per channel: x' = ((x - in_black)/(in_white - in_black)) ** gamma
     (clamped to [0,1], out_black=0, out_white=1).
  3) Convert to LAB, apply a *very* mild unsharp on L to reveal detail.

Usage:
  python ps_levels_auto.py input.jpg output.jpg

Install:
  pip install opencv-python numpy
"""
import sys
import cv2
import numpy as np

def levels_per_channel_rgb(img_bgr_u8: np.ndarray) -> np.ndarray:
    # BGR -> RGB float [0,1]
    rgb = cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Percentile-based black/white points per channel
    in_blacks = np.percentile(rgb.reshape(-1, 3), 1.0, axis=0)
    in_whites = np.percentile(rgb.reshape(-1, 3), 99.5, axis=0)

    # Protect against tiny ranges
    ranges = np.maximum(in_whites - in_blacks, 1e-6)

    # Gamma > 1 darkens mids (good for overexposed images)
    gamma = 1.45

    # Apply Levels per-channel in sRGB space (Photoshop-like behavior)
    rgb = (rgb - in_blacks) / ranges
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.power(rgb, gamma)  # midtone slider

    # Back to BGR uint8
    out = (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def mild_unsharp_luminance(bgr_u8: np.ndarray) -> np.ndarray:
    # LAB: sharpen only L to avoid color shifts
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Very mild unsharp to reveal folds/texture without halos
    blur = cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=1.0, sigmaY=1.0)
    amount = 0.35
    L2 = cv2.addWeighted(L, 1 + amount, blur, -amount, 0)
    L2 = np.clip(L2, 0, 255).astype(np.uint8)

    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def auto_levels_ps(bgr: np.ndarray) -> np.ndarray:
    leveled = levels_per_channel_rgb(bgr)
    # Optional: detail pop
    return mild_unsharp_luminance(leveled)

def main():
    if len(sys.argv) < 3:
        print("Usage: python ps_levels_auto.py input.jpg output.jpg")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]
    img = cv2.imread(inp, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Cannot read image: {inp}", file=sys.stderr)
        sys.exit(2)

    fixed = auto_levels_ps(img)
    ok = cv2.imwrite(out, fixed)
    if not ok:
        print(f"Failed to write: {out}", file=sys.stderr)
        sys.exit(3)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
