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






