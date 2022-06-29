import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class HorseZebras(Dataset):
    def __init__(self, root_zebra, root_horse):
        self.root_zebra = root_zebra
        self.root_horse = root_horse

        self.zebra_image = os.listdir(self.root_zebra)
        self.horse_image = os.listdir(self.root_horse)
        self.data_length = max(len(self.zebra_image), len(self.horse_image))
        self.horse_len = len(self.horse_image)
        self.zebra_len = len(self.zebra_image)

    def __len__(self):
        return self.data_length

    def __getitem__(self, item):
        zebra_img_name = self.zebra_image[item % self.zebra_len]
        horse_img_name = self.horse_image[item % self.horse_len]

        zebra_path = os.path.join(self.root_zebra, zebra_img_name)
        horse_path = os.path.join(self.root_horse, horse_img_name)

        zebra_image = np.array(Image.open(zebra_path))
        horse_image = np.array(Image.open(horse_path))

        return zebra_image, horse_image