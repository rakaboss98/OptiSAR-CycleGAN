# Creating a PatchGAN discriminator architecture
# PatchGAN: https://sahiltinky94.medium.com/understanding-patchgan-9f3c8380c207
# Instance Norm: https://medium.com/techspace-usict/normalization-techniques-in-deep-neural-networks-9121bf100d8
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, stride=1, kernel_size=4,
                      padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=4,
                      padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, stride=2, kernel_size=4,
                      padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=512, stride=2, kernel_size=4,
                      padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=1, stride=1, kernel_size=4,
                      padding=1, padding_mode="reflect"),
            nn.Sigmoid()
        )

    def forward(self, tensor):
        return self.conv(tensor)


# Writing the test case
if __name__ == '__main__':
    test_tensor = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    predictions = model(test_tensor)
    print(predictions.shape)
