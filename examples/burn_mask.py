#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
from pathlib import Path

def make_overexposed_mask(img_bgr: np.ndarray,
                          luma_thresh: int = 250,
                          rgb_thresh: int = 250,
                          open_ksize: int = 3,
                          close_ksize: int = 5) -> np.ndarray:
    """
    Returns a uint8 mask (0/255) where overexposed/clipped areas are 255 (white).
    """
    # Luma (perceptual brightness) ~ Rec.601
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    luma_mask = (gray >= luma_thresh)

    # True clip: all channels high
    b, g, r = cv2.split(img_bgr)
    rgb_clip_mask = (r >= rgb_thresh) & (g >= rgb_thresh) & (b >= rgb_thresh)

    # Combine (OR): anything very bright OR truly clipped
    mask = (luma_mask | rgb_clip_mask).astype(np.uint8) * 255

    # Optional: clean up noise with morphology
    if open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return mask

def main():
    ap = argparse.ArgumentParser(description="Create BW mask for overexposed/clipped regions.")
    ap.add_argument("--input", required=True, help="Path to original image")
    ap.add_argument("--output", default="overexposed_mask.jpg", help="Path to save mask")
    ap.add_argument("--luma_thresh", type=int, default=250, help="0–255; higher = stricter")
    ap.add_argument("--rgb_thresh", type=int, default=250, help="0–255; higher = stricter")
    ap.add_argument("--open_ksize", type=int, default=3, help="Morphological open kernel size (0 disables)")
    ap.add_argument("--close_ksize", type=int, default=5, help="Morphological close kernel size (0 disables)")
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.input}")

    mask = make_overexposed_mask(
        img,
        luma_thresh=args.luma_thresh,
        rgb_thresh=args.rgb_thresh,
        open_ksize=args.open_ksize,
        close_ksize=args.close_ksize,
    )

    # ensure white=overexposed, black=others
    cv2.imwrite(args.output, mask)
    print(f"Saved mask to: {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
