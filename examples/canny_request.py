"""Flux CLI Tool to generate images using the FLUX.1 Canny model via the BFL API."""

import os
import sys
import time
import base64
import textwrap
from io import BytesIO
from typing import Optional
import requests
from PIL import Image
from pydantic import BaseModel
from argdantic import ArgParser
import cv2

def preprocess_sketch(path: str) -> str:
    """Preprocesses the pencil sketch to prepare a better control image for Canny-based generation."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Invert so pencil marks are dark on white
    img = cv2.bitwise_not(img)

    # Normalize lighting and boost contrast
    img = cv2.equalizeHist(img)

    # Optional: Remove horizontal lines (not perfect, but helps)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

    # Threshold to get hard binary ink-like lines
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # Invert back: white background, black lines
    img = cv2.bitwise_not(img)

    # Resize or pad to square if needed (optional)

    # Encode to base64
    _, buffer = cv2.imencode(".jpg", img)
    encoded = base64.b64encode(buffer).decode()
    return encoded


class Args(BaseModel):
    """Command-line arguments for Flux Canny image generation."""
    input_image: str = "img/robber.jpg"  # Path to the image to extract edges from
    output_image: str = "output.jpg"  # File path to save the final generated image
    prompt: str = textwrap.dedent('''
        A clean inked version of the original sketch, preserving the character's unique hairstyle, 
        confident smirk, mole, and neck tattoo. The final image should look like it was inked by 
        hand with a black pen: no grayscale, no halftones, no digital brush effects. Just bold, 
        confident linework and real human inking texture on a white background. All lines must be clean, 
        dark, and sharp, with high contrast. Do not change the character’s pose, proportions, hairstyle, 
        or facial features from the sketch.
    ''').strip()
    canny_low_threshold: int = 150  # Canny edge detection low threshold
    canny_high_threshold: int = 250  # Canny edge detection high threshold
    seed: Optional[int] = None  # Random seed for reproducibility (optional)
    prompt_upsampling: bool = True  #  If True, upscales the result using prompt guidance
    safety_tolerance: int = 3  # Between 0 and 3; 0=strict, 3=very loose
    output_format: str = "png"  # Output file format: "jpeg" or "png"
    timeout: int = 60  # Time (in seconds) to poll the result before giving up

def edit_image(args: Args):
    """Sends Canny-based image generation request to Flux API and polls for result."""
    # Load and encode input image either in JPEG or PNG format
    # image = Image.open(args.input_image).convert("RGB")
    # buffered = BytesIO()
    # image.save(buffered, format="JPEG" if args.input_image.lower().endswith('.jpg') else "PNG")
    # img_str = base64.b64encode(buffered.getvalue()).decode()
    with open(args.input_image, "rb") as f:
        img_bytes = f.read()
    img_str = base64.b64encode(img_bytes).decode()

    # img_str = preprocess_sketch(args.input_image)

    # Submit request
    try:
        response = requests.post(
            'https://api.bfl.ai/v1/flux-pro-1.0-canny',
            headers={
                'accept': 'application/json',
                'x-key': os.environ.get("BFL_API_KEY"),
                'Content-Type': 'application/json',
            },
            json={
                'prompt': args.prompt,
                'control_image': img_str,
                'canny_low_threshold': args.canny_low_threshold,
                'canny_high_threshold': args.canny_high_threshold,
                'seed': args.seed,
                'prompt_upsampling': args.prompt_upsampling,
                'safety_tolerance': args.safety_tolerance,
                'output_format': args.output_format,
            },
            timeout=60
        )
        response.raise_for_status()
        request = response.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(1)

    request_id = request.get("id")
    polling_url = request.get("polling_url")

    # Polling loop
    start_time = time.time()
    while True:
        if time.time() - start_time > args.timeout:
            print("Timeout waiting for image generation")
            break

        try:
            result = requests.get(
                polling_url,
                headers={
                    'accept': 'application/json',
                    'x-key': os.environ.get("BFL_API_KEY"),
                },
                params={'id': request_id},
                timeout=60
            ).json()

            if result['status'] == 'Ready':
                image_url = result['result']['sample']
                image_data = requests.get(image_url, timeout=10).content
                with open(args.output_image, 'wb') as f:
                    f.write(image_data)
                print(f"Image saved to {args.output_image}")
                print(f"Seed: {result['result']['seed']}")
                break

            if result['status'] in ['Error', 'Failed']:
                print(f"Generation failed: {result}")
                break

        except requests.RequestException as e:
            print(f"Polling failed: {e}")
            break

        time.sleep(0.5)

cli = ArgParser()

@cli.command(singleton=True)
def main(config: Args):
    """Main CLI entry point to generate image using FLUX.1 Canny."""
    edit_image(config)

if __name__ == "__main__":
    cli()
