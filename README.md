# Optimizing cat2dog

Our research team has trained a new state-of-the-art model that converts cats into cartoon dogs:

<p align="center">
  <img src="images/img1.jpg" width="45%">
  <img src="images/out1.jpg" width="45%">
</p>

We are now interested to ship this model to our users, but the current implementation is way too slow.

Your job is to profile and optimize our current research implementation, minimizing inference latency without sacrificing model quality.

## Setup

Create a virtual environment and install dependencies:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision einops jaxtyping
```

Download our pretrained model checkpoints:

```bash
wget https://cdn.marble.worldlabs.ai/takehomes/re-opt/models.tar
tar -xvf models.tar
rm models.tar
```

## Running inference

You can run inference on one of our pretrained checkpoints like this:

```bash
python run.py --model models/500m-256 --input-image images/img1.jpg
```

This should reproduce the result on this page, saving the output image to `out.png`.

We have provided several model checkpoints that vary both the model size and the image resolution; they follow the convention `models/{size}-{resolution}`.

We have also provided several sample input images in the `images/` directory.

## Your Task

Your goal is to optimize the inference performance of these models on H100 GPUs.

Make whatever changes you want to the inference code in order to **minimize the end-to-end latency** of generating results from the model, without drastically reducing the quality of outputs from the model.

At present, only the `500m-256` model can run without crashing. You should be able to successfully run inference on all models we provide.

This model will eventually be embedded in a backend service to run inference on an incoming stream of many user requests; as such we are willing to accept some additional startup time in exchange for serving inference results faster.

## What to Submit

You should submit a zipfile or tarbell containing:

1. All your modified and optimized inference code
2. A brief writeup (~1-2 pages) explaining your changes
3. Your submission SHOULD NOT include any model checkpoints; if you need to modify any of our pretrained checkpoints, include a script and instructions for generating your modified checkpoints from our provided checkpoints.

Your writeup should include the following:
- A description of each change you made and why it matters
- For each model checkpoint:
  - The **total startup time** for your optimized inference code, measured in milliseconds
  - The **average runtime** of running inference across the 5 examples we provide, measured in milliseconds
  - The **average throughput** of running inference across the 5 examples we provide, measured in TFLOP/sec and MFU
  - The **average difference** between the unoptimized model outputs and your optimized model outputs for the 5 examples we provide, measured in PSNR. You should achieve an average PSNR > 30 dB.

## Using GPUs

We will reimburse up to $200 of expenses for using cloud GPUs.

On AWS, a `p5.4xlarge` instance has 1xH100 GPU and costs about $7/hour.

You can probably find lower prices from some other providers; e.g. Lambda Labs sometimes has H100 GPUs available for closer to $3/hour.
