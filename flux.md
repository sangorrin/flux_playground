Flux

# References

Black Forest Labs
https://dashboard.bfl.ai/
https://playground.bfl.ai/
https://bfl.ai/pricing/api
https://docs.bfl.ai/quick_start/introduction

https://huggingface.co/black-forest-labs

Civitai
https://civitai.com/models

Flux-AI
https://flux-ai.io/

Comfy UI
https://www.comfy.org/

RunPod
https://www.runpod.io

Fal
https://fal.ai/

# BFL API

## Environment

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
- a poll url

## Wait for the result

