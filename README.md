# xDDPM: Explainable Denoising Diffusion Probabilistic Model for scientific modeling
## Denoising Diffusion Probabilistic Model, in Pytorch

Implementation of xDDPM: Explainable Denoising Diffusion Probabilistic Model for scientific modelingin Pytorch. First diffusion-based model for explainable generation. We introduce a simple variant to the DDPM method that incorporates Information Bottleneck for attribution and is able to generate samples that exclusively contain the signal-relevant components.


## Install

```bash
$ pip install -r requirement.txt
```

## Usage

```python
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
)

training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)
```

Or, if you simply want to pass in a folder name and the desired image dimensions, you can use the `Trainer` class to easily train a model.

```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

import torch
from code.denoising_diffusion_pytorch.denoising_diffusion_pytorch_tension import Unet, GaussianDiffusion,Trainer, UnetZ, Decode
import pickle
from torch import autograd
from torch import nn

def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file

model = Unet(
    dim = 64,
    channels = 20, 
    dim_mults = (1, 2, 4, 8),
    flash_attn = False
    
)
modelz = UnetZ(
    dim = 64,
    channels = 20, 
    dim_mults = (1, 2, 4, 8),
    flash_attn = False   
)
decode = Decode()


diffusion = GaussianDiffusion(
    model,
    modelz,
    decode, 
    image_size = 32,
    timesteps = 1000,    # number of steps
    objective = 'pred_noise'
)




data_path = "/yourdatapath"



trainer = Trainer(
    diffusion,
    data_path,
    train_batch_size = 250,  #7291
    train_lr = 8e-5,
    train_num_steps = 50000,       # total training steps
    save_and_sample_every = 1000,
    results_folder = data_path,
    gradient_accumulate_every = 8,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True,             # whether to calculate fid during training
    num_fid_samples = 25000
)

trainer.train()
```

Samples and model checkpoints will be logged to `./results` periodically

## Multi-GPU Training

The `Trainer` class is now equipped with <a href="https://huggingface.co/docs/accelerate/accelerator">🤗 Accelerator</a>. You can easily do multi-gpu training in two steps using their `accelerate` CLI

At the project root directory, where the training script is, run

```python
$ accelerate config
```

Then, in the same directory

```python
$ accelerate launch train.py
```

