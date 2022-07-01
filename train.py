from cgi import test
from locale import locale_alias
import sys
from numpy import identity 
import torch 
from utils import readyaml 
from utils.readwrite import load_checkpoint, save_checkpoint
import torch.nn as nn
from dataset.dataloader import HorseZebras
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
batch_size = config['HYPERPARAMS']['BATCH_SIZE']
num_workers = config['HYPERPARAMS']['NUM_WORKERS']

load_model = config['TRAINING']['LOAD_MODEL']
save_model = config['TRAINING']['SAVE_MODEL']
train_dir = config['TRAINING']['TRAIN_DIR']
test_dir = config['TRAINING']['TRAIN_DIR']

ckpt_genH = config['TRAINING']['CHECKPOINT_GEN_H']
ckpt_genZ = config['TRAINING']['CHECKPOINT_GEN_Z']
ckpt_discH = config['TRAINING']['CHECKPOINT_CRITIC_H']
ckpt_discZ = config['TRAINING']['CHECKPOINT_CRITIC_Z']

lambda_cycle = config['HYPERPARAMS']['LAMBDA_CYCLE']
lambda_identity = config['HYPERPARAMS']['LAMBDA_IDENTITY']

def train_fn(disc_H, disc_Z, gen_H, gen_Z, loader, opti_disc, opti_gen, L1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(device)
        horse = horse.to(device)

        # Train discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_fake_loss+D_H_real_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real,torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake,torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_fake_loss+D_Z_real_loss

            D_Loss = (D_H_loss+D_Z_loss)/2 
        
        opti_disc.zero_grad()
        d_scaler.scale(D_Loss).backward()
        d_scaler.step(opti_disc)
        d_scaler.update()

        # Train generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_gen_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_gen_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss 
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = L1(zebra, cycle_zebra)
            cycle_horse_loss = L1(horse, cycle_horse)

            # identity loss 
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = L1(zebra, identity_zebra)
            identity_horse_loss = L1(horse, identity_horse)

            G_loss = loss_gen_H + loss_gen_Z + (cycle_zebra_loss+cycle_horse_loss)*lambda_cycle 
        
        opti_gen.zero_grad()
        g_scaler.scale(G_loss)
        g_scaler.step(opti_gen)
        g_scaler.update()

        if idx%200 == 0:
            save_image(fake_horse, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra, f"saved_images/zebra_{idx}.png")
 
def main():
    disc_H = Discriminator(in_channels=3).to(device)
    disc_Z = Discriminator(in_channels=3).to(device)
    gen_H = Generator(img_channels=3).to(device)
    gen_Z = Generator(img_channels=3).to(device)

    opti_disc = optim.Adam(
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
        load_checkpoint(ckpt_genH, gen_H, opti_gen, lr, device)
        load_checkpoint(ckpt_genZ, gen_Z, opti_gen, lr, device)
        load_checkpoint(ckpt_discH, disc_H, opti_disc, lr, device)
        load_checkpoint(ckpt_discZ, disc_Z, opti_disc, lr, device)

    dataset = HorseZebras(
        root_zebra=train_dir+'trainZ', root_horse=train_dir+'trainH'
        )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):

        train_fn(disc_H, disc_Z, gen_H, gen_Z, loader, opti_disc, opti_gen, L1, mse, d_scaler, g_scaler)

        if save_model:
            save_checkpoint(disc_H, opti_disc, ckpt_discH)
            save_checkpoint(disc_Z, opti_disc, ckpt_discZ)
            save_checkpoint(gen_H, opti_gen, ckpt_genH)
            save_checkpoint(gen_Z, opti_gen, ckpt_genZ)

if __name__ == "__main__":
    main()