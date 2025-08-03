"""Flux CLI Tool to edit images via the BFL API."""

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

class Args(BaseModel):
    """Command-line arguments for Flux image editing."""
    input_image: str = "img/robber.jpg"  # Path to the image to be edited
    output_image: str = "output.jpg"  # File path to save the final generated image
    prompt: str = textwrap.dedent('''
        A full-body character design of a young man with medium-length, messy hair and asymmetrical bangs,
        wearing a layered outfit with a shirt, jacket, and visible tattoo on his neck. His expression is
        slightly mischievous or confident, with a mole on his cheek and sharp, angular features.
        He should have a cyberpunk or gritty urban style, similar to the world of Akira.
        He has a yakuza tattoo on his neck.
        The artwork must be black and white, fully inked, with bold, expressive linework and strong contrast,
        using detailed shadows and textures reminiscent of 1980s seinen manga.
        The character stands in a neutral pose, clearly visible from head to toe,
        with careful attention to anatomy, folds in the clothing, and realistic proportions.
        The background should be plain or minimal to emphasize the character.
        Studio-quality comic illustration. No colors, no greyscale, just pure black ink on white background.
    ''').strip()  # Description of the visual transformation to apply to the input image
    aspect_ratio: str = "3:6"  # Desired output image aspect ratio, e.g., "1:1", "3:4"
    seed: Optional[int] = None  # Random seed for reproducibility (optional)
    prompt_upsampling: bool = True  # Enables enhanced text interpretation if True
    safety_tolerance: int = 2  # Controls moderation: 0 = strict, 2 = relaxed
    output_format: str = "jpeg"  # Output file format: "jpeg" or "png"
    timeout: int = 60  # Time (in seconds) to poll the result before giving up

def edit_image(args: Args):
    """Sends image editing request to Flux API and polls for result."""
    # Load and encode image
    image = Image.open(args.input_image)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Make initial request
    try:
        response = requests.post(
            'https://api.bfl.ai/v1/flux-kontext-pro',
            headers={
                'accept': 'application/json',
                'x-key': os.environ.get("BFL_API_KEY"),
                'Content-Type': 'application/json',
            },
            json={
                'prompt': args.prompt,
                'input_image': img_str,
                'aspect_ratio': args.aspect_ratio,
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

    request_id = request["id"]
    polling_url = request["polling_url"]

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
    """Main CLI entry point to edit image."""
    edit_image(config)

if __name__ == "__main__":
    cli()
