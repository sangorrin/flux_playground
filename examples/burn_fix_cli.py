"""
Burn-Fix CLI — Use a pre-made mask image with BFL's Flux Fill model (flux-pro-1.0-fill)
to fix burned highlights in photos.

Install:
  pip install pillow requests pydantic argdantic

Env:
  export BFL_API_KEY="sk-......"

Usage:
  python burn_fix_cli.py --input-image img/photo.png \
                        --mask-image img/mask.png
"""

import os
import io
import sys
import time
import base64
import textwrap
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from pydantic import BaseModel, Field
from argdantic import ArgParser


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as RGB numpy array (HWC, uint8)."""
    img = Image.open(path)
    return np.array(img.convert('RGB'))


def encode_image_base64_pil(im: Image.Image, fmt: str = "JPEG") -> str:
    """Encode PIL image to base64 string."""
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------- CLI + API call ----------------------------

class Args(BaseModel):
    """CLI args and config."""
    input_image: str = Field(..., description="Path to the source photo")
    mask_image: str = Field(..., description="Path to the black & white mask image (255=edit area)")
    output_image: str = Field("output.jpg", description="Where to save the final image")
    prompt: str = Field(
        textwrap.dedent("""
            Beautiful tanned skin, blue big eyes lady and man.
        """).strip(),
        description="Text guidance for Flux Fill",
    )
    steps: int = 50
    guidance: int = 80
    output_format: str = "jpeg"  # "jpeg" or "png"
    timeout: int = 90           # seconds to poll before giving up


def flux_fill_inpaint(args: Args) -> None:
    """Run the burn-fix inpainting flow."""
    api_key = os.environ.get("BFL_API_KEY")
    if not api_key:
        print("ERROR: set env var BFL_API_KEY")
        sys.exit(1)

    # 1) Load input and mask
    in_path = Path(args.input_image)
    mask_path = Path(args.mask_image)
    rgb = load_image_rgb(in_path)
    mask = np.array(Image.open(mask_path).convert('L'))  # Load as grayscale

    # 2) Prepare base64 inputs for Flux Fill
    pil_img = Image.fromarray(rgb)
    pil_mask = Image.fromarray(mask)  # single-channel, 255=paint area
    img_str = encode_image_base64_pil(pil_img, "JPEG" if args.output_format == "jpeg" else "PNG")
    mask_str = encode_image_base64_pil(pil_mask, "PNG")  # keep as PNG to preserve mask

    # 3) Submit job
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

    # 4) Poll
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
