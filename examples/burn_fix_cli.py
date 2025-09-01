"""
Burn-Fix CLI — create a burned-highlights mask with OpenCV and inpaint it
via BFL's Flux Fill model (flux-pro-1.0-fill).

Install:
  pip install opencv-python pillow requests pydantic argdantic

Env:
  export BFL_API_KEY="sk-......"

Usage:
  python burn_fix_cli.py --input-image img/quemada.png \
                         --save-debug-mask seed_mask.png
"""
# pylint: disable=no-member

import os
import io
import sys
import time
import base64
import textwrap
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image
from pydantic import BaseModel, Field
from argdantic import ArgParser


# --------------------------- OpenCV seed-mask ---------------------------

def build_seed_mask(
    rgb: np.ndarray,
    th_chan: float = 0.97,   # was 0.985
    th_luma: float = 0.96,   # was 0.975
    kernel_sz: int = 5,
    min_area: int = 150,
    expand_px: int = 18,     # NEW: expand mask outward
    feather_px: int = 3,     # NEW: light blur to smooth edges (for preview; we still send binary)
) -> np.ndarray:
    """
    Create an aggressive mask for blown-out/near-blown highlights.
    255 = EDIT (to fill), 0 = KEEP.
    """
    srgb = rgb.astype(np.float32) / 255.0
    linear = np.power(np.clip(srgb, 0, 1), 2.2)

    luminance = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]

    # seed: any channel near 1 OR high luminance  OR very low local contrast (halo)
    near_white_any = (linear[..., 0] >= th_chan) | \
                     (linear[..., 1] >= th_chan) | \
                     (linear[..., 2] >= th_chan)
    near_white_luma = luminance >= th_luma

    # bonus: pixels that are bright AND low-gradient (typical bloom around the sun/sky)
    gx = cv2.Sobel((luminance*255).astype(np.uint8), cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel((luminance*255).astype(np.uint8), cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.absdiff(gx, 0), 1, cv2.absdiff(gy, 0), 1, 0))
    low_texture = grad < 6  # tweak if needed

    seed = (near_white_any | near_white_luma | (low_texture & (luminance > 0.90))).astype(np.uint8) * 255

    # Morphology
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_sz, kernel_sz))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k, iterations=1)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN,  k, iterations=1)

    # Remove tiny specks
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
    mask = np.zeros_like(seed)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255

    # Distance-based expansion (stronger than fixed dilate)
    inv = (mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    # Pixels within 'expand_px' of current foreground become foreground
    expand = (dist <= expand_px).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask, expand)

    if feather_px > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather_px)

    # Ensure binary for the API (Flux Fill expects binary mask; 255 = edit)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return mask


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as RGB numpy array (HWC, uint8)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_image_base64_pil(im: Image.Image, fmt: str = "JPEG") -> str:
    """Encode PIL image to base64 string."""
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------- CLI + API call ----------------------------

class Args(BaseModel):
    """CLI args and config."""
    input_image: str = Field(..., description="Path to the source photo")
    output_image: str = Field("output.jpg", description="Where to save the final image")
    prompt: str = Field(
        textwrap.dedent("""
            Fix clipped highlights in this photo using the provided mask.
            Make skies light blue and match surrounding colors.
        """).strip(),
        description="Text guidance for Flux Fill",
    )
    # OpenCV detector knobs (optional)
    th_chan: float = 0.985
    th_luma: float = 0.975
    kernel_sz: int = 5
    min_area: int = 150
    # BFL generation knobs
    steps: int = 50
    guidance: int = 80
    output_format: str = "png"  # "jpeg" or "png"
    timeout: int = 90            # seconds to poll before giving up
    save_debug_mask: Optional[str] = None  # e.g. "seed_mask.png"


def flux_fill_inpaint(args: Args) -> None:
    """Run the burn-fix inpainting flow."""
    api_key = os.environ.get("BFL_API_KEY")
    if not api_key:
        print("ERROR: set env var BFL_API_KEY")
        sys.exit(1)

    # 1) Load input
    in_path = Path(args.input_image)
    rgb = load_image_rgb(in_path)

    # 2) Build mask (255 = to fill)
    mask = build_seed_mask(
        rgb,
        th_chan=args.th_chan,
        th_luma=args.th_luma,
        kernel_sz=args.kernel_sz,
        min_area=args.min_area,
    )

    # Optional debug save
    if args.save_debug_mask:
        cv2.imwrite(args.save_debug_mask, mask)

    # 3) Prepare base64 inputs for Flux Fill
    pil_img = Image.fromarray(rgb)
    pil_mask = Image.fromarray(mask)  # single-channel, 255=paint area
    img_str = encode_image_base64_pil(pil_img, "JPEG" if args.output_format == "jpeg" else "PNG")
    mask_str = encode_image_base64_pil(pil_mask, "PNG")  # keep as PNG to preserve mask

    # 4) Submit job
    try:
        r = requests.post(
            "https://api.bfl.ai/v1/flux-pro-1.0-fill",
            headers={
                "x-key": api_key,
                "Content-Type": "application/json",
                "accept": "application/json",
            },
            json={
                "prompt": args.prompt,
                "image": img_str,
                "mask": mask_str,
                "steps": args.steps,
                "guidance": args.guidance,
                "output_format": args.output_format,
            },
            timeout=60,
        )
        r.raise_for_status()
        job = r.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(1)

    request_id = job.get("id")
    polling_url = job.get("polling_url")
    if not (request_id and polling_url):
        print(f"Unexpected API response: {job}")
        sys.exit(1)

    # 5) Poll
    start = time.time()
    while True:
        if time.time() - start > args.timeout:
            print("Timeout waiting for generation.")
            sys.exit(2)

        try:
            res = requests.get(
                polling_url,
                headers={
                    "accept": "application/json",
                    "x-key": api_key,
                },
                params={"id": request_id},
                timeout=30,
            ).json()
        except requests.RequestException as e:
            print(f"Polling failed: {e}")
            sys.exit(1)

        status = res.get("status", "")
        if status == "Ready":
            image_url = res["result"]["sample"]
            seed = res["result"].get("seed")
            try:
                img_bytes = requests.get(image_url, timeout=30).content
                with open(args.output_image, "wb") as f:
                    f.write(img_bytes)
            except requests.RequestException as e:
                print(f"Download failed: {e}")
                sys.exit(1)
            print(f"Saved → {args.output_image}")
            if seed is not None:
                print(f"Seed: {seed}")
            break

        if status in ("Error", "Failed"):
            print(f"Generation failed: {res}")
            sys.exit(1)

        time.sleep(0.6)


cli = ArgParser()

@cli.command(singleton=True)
def main(config: Args):
    """CLI entrypoint."""
    flux_fill_inpaint(config)

if __name__ == "__main__":
    cli()
