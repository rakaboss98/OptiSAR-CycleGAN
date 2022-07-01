import torch 
from utils.readyaml import read_yaml_file

def save_checkpoint(model, optimizer, filename):
    print('==> Saving checkpoint')
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print('==> Loading checkpoint')
    ckpt = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
