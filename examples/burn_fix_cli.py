"""
Burn-Fix CLI — create a burned-highlights mask with OpenCV and inpaint it
via BFL's Flux Fill model (flux-pro-1.0-fill).

Install:
  pip install opencv-python pillow requests pydantic argdantic

Env:
  export BFL_API_KEY="sk-......"

Usage:
  python burn_fix_cli.py --input-image img/quemada.png \
                         --prompt "blue sky with light clouds" \
                         --output-image fixed.jpg \
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
    th_chan: float = 0.985,
    th_luma: float = 0.975,
    kernel_sz: int = 5,
    min_area: int = 150,
) -> np.ndarray:
    """
    Create a high-recall mask for clipped whites (burned highlights).
    Returns a uint8 mask where 255 = region to inpaint (burned), 0 = keep.
    """
    # sRGB [0,255] -> [0,1] float
    srgb = rgb.astype(np.float32) / 255.0
    # approx inverse gamma to linearize (good enough for thresholding)
    linear = np.power(np.clip(srgb, 0, 1), 2.2)

    # Luminance (BT.709)
    luminance = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]

    # High-recall seed: any channel near 1.0 OR luminance near 1.0
    near_white_any = (linear[..., 0] >= th_chan) | \
                     (linear[..., 1] >= th_chan) | \
                     (linear[..., 2] >= th_chan)
    near_white_luma = luminance >= th_luma
    seed = (near_white_any | near_white_luma).astype(np.uint8) * 255

    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_sz, kernel_sz))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k, iterations=1)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, k, iterations=1)

    # Remove tiny blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
    mask = np.zeros_like(seed)
    for i in range(1, n):  # 0 is background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255

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
            Recover realistic detail in blown-out white areas while keeping the rest of the
            image intact. If the mask covers sky, recreate a natural blue sky with subtle
            clouds matching scene lighting and perspective. If the mask touches fabric or
            walls, generate plausible texture and shading coherent with nearby edges.
        """).strip(),
        description="Text guidance for Flux Fill",
    )
    # OpenCV detector knobs (optional)
    th_chan: float = 0.985
    th_luma: float = 0.975
    kernel_sz: int = 5
    min_area: int = 150
    # BFL generation knobs
    steps: int = 40
    guidance: int = 20
    output_format: str = "jpeg"  # "jpeg" or "png"
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
