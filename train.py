from cgi import test
import sys 
import torch 
from utils import readyaml 
from utils.readwrite import load_checkpoint
import torch.nn as nn
from dataset import dataloader
from torch.utils.data import DataLoader
import torch.optim as optim 
from tqdm import tqdm
from models.discriminator import Discriminator
from models.generator import Generator
from torchvision.utils import save_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path_name = 'config.yaml'
config = readyaml.read_yaml_file(path_name)

lr = config['HYPERPARAMS']['LEARNING_RATE']
num_epochs = config['HYPERPARAMS']['NUM_EPOCHS']

load_model = config['TRAINING']['LOAD_MODEL']
save_model = config['TRAINING']['SAVE_MODEL']
train_dir = config['TRAINING']['TRAIN_DIR']
test_dir = config['TRAINING']['TRAIN_DIR']

ckpt_genH = config['TRAINING']['CHECKPOINT_GEN_H']
ckpt_genZ = config['TRAINING']['CHECKPOINT_GEN_Z']
ckpt_discH = config['TRAINING']['CHECKPOINT_CRITIC_H']
ckpt_dicZ = config['TRAINING']['CHECKPOINT_CRITIC_Z']

def main():
    disc_H = Discriminator(in_channels=3).to(device)
    disc_Z = Discriminator(in_channels=3).to(device)
    gen_H = Generator(img_channels=3).to(device)
    gen_Z = Generator(img_channels=3).to(device)

    opt_disc = optim.Adam(
        list(disc_H.parameters())+list(disc_Z.parameters()),
        lr = lr,
        betas = (0.5, 0.999)
    )

    opti_gen = optim.Adam(
        list(gen_H.parameters())+list(gen_Z.parameters()),
        lr = lr,
        betas = (0.5,0.999)
    )
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if load_model:
        load_checkpoint(
            ckpt_genH, gen_H, opti_gen, lr
        )
 