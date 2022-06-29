import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        nn.Module.__init__(self)
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.Conv(tensor)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.Block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, kernel_size=3, use_act=False, padding=1)
        )

    def forward(self, tensor):
        return tensor + self.Block(tensor)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        nn.Module.__init__(self)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU()
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1)
            ]
        )
        self.residual_block = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
                ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1)
            ]
        )

        self.final = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, tensor):
        tensor = self.initial(tensor)
        for each_block in self.down_blocks:
            tensor = each_block(tensor)
        tensor = self.residual_block(tensor)
        for each_block in self.up_blocks:
            tensor = each_block(tensor)
        tensor = self.final(tensor)
        tensor = torch.tanh(tensor)
        return tensor


if __name__ == "__main__":
    image_channels = 3
    image_size = 256
    image = torch.randn((2, image_channels, image_size, image_size))
    genx = Generator(img_channels=image_channels)
    image2 = genx.forward(image)
    print(image2.shape)
