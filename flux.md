Flux

# References

Black Forest Labs
https://dashboard.bfl.ai/
https://playground.bfl.ai/
https://bfl.ai/pricing/api
https://docs.bfl.ai/quick_start/introduction
https://huggingface.co/black-forest-labs

Copainter
https://www.copainter.ai/en

Frameplanner
https://frameplanner-e5569.web.app/

Octocomics
https://www.animeshorts.ai/ai-generator/comic
https://piyo-piyo-piyo.com/14168/

Civitai
https://civitai.com/models

TensorArt
https://tensor.art/

OpemArt
https://openart.ai/

SeaArt
https://www.seaart.ai/es

Comfy UI
https://www.comfy.org/
https://www.ipentec.com/document/ai-image/flux-1-kontext-inking-from-lineart

RunPod
https://www.runpod.io

Fal
https://fal.ai/

# BFL API

## Example

Prepare a conda environment and install the dependencies
```
$ conda create -n flux python
$ conda activate flux
$ python -m pip install --upgrade pip setuptools wheel
$ pip install requests Pillow argdantic
$ vi ~/.zshrc
    export BFL_API_KEY="your_actual_key_here"
```

Try the example
```
$ cd examples
$ python api_request.py
    Image saved to output.jpg
    Seed: 1351337727 <-- you can use it as input during the next interation.
$ open output.jpg
```

## Create the request

Examples:
    - https://docs.bfl.ai/kontext/kontext_image_editing

This is just a requests.post with
- post URL: https://api.bfl.ai/v1/${model}
- headers: x-key with the API key
- json: prompt, image_str (image converted to base64 string)

It returns
- an id
- a polling url

## Wait for the result

Next all you need to do is polling until the result is ready
- requests.get(polling_url, request_id)

Example response
```
{
    'id': '12ef21b0-5823-4c9e-8558-fa8b374de26d',
    'status': 'Ready',
    'result': {
        'sample': 'https://delivery-us1.bfl.ai/results/c5/9633896182c4ab/6182c4abf8974b0e9be908b209dfb4ea/sample.jpeg?se=2025-08-03T10%3A39%3A23Z&sp=r&sv=2024-11-04&sr=b&rsct=image/jpeg&sig=RGPXlnY930OOAsG6XH7ipKabsfaWjwipg6IopgotGdA%3D',
        'prompt': 'Add a full-body to the young man with medium-length, messy hair and asymmetrical bangs, wearing a layered outfit with a shirt, jacket, and visible yakuza tattoo on his neck. His expression is slightly mischievous or confident, with a mole on his cheek and sharp, angular features. Style as cyberpunk or gritty urban, similar to the world of Akira. Use only white (#FFFFFFFF) background and black opaque (#000000FF) for the man. Convert from pencils to ink, ready for B/W comic printing, using an inking style similar to the world of Akira.',
        'seed': 2400964722,
        'start_time': 1754216958.9739745,
        'end_time': 1754216963.9304962,
        'duration': 4.95652174949646
    },
    'progress': None,
    'details': None,
    'preview': None
}
```