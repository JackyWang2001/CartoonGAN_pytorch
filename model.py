import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


# model for one residual block
class residual_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.norm1 = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.norm2 = nn.InstanceNorm2d(channel)

        utils.initialize_weights(self)

    def forward(self, inputs):
        c1 = self.conv1(inputs)
        n1 = self.norm1(c1)
        r1 = F.relu(n1, True)

        c2 = self.conv2(r1)
        outputs = self.norm2(c2)

        return inputs + outputs  # Elementwise sum


# model for the generator
class generator(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, filters=64, res_num=8):
        """
            filters: the number of filters/feature maps in the first and last conv layers
            res_num: the number of residual blocks
        """
        super(generator, self).__init__()

        # down convolution
        self.down_convs = nn.Sequential(
            nn.Conv2d(in_channel, filters, 7, 1, 3),  # k7n64s1
            nn.InstanceNorm2d(filters),
            nn.ReLU(True),

            nn.Conv2d(filters, filters * 2, 3, 2, 1),  # k3n128s2
            nn.Conv2d(filters * 2, filters * 2, 3, 1, 1),  # k3n128s1
            nn.InstanceNorm2d(filters * 2),
            nn.ReLU(True),

            nn.Conv2d(filters * 2, filters * 4, 3, 2, 1),  # k3n256s1
            nn.Conv2d(filters * 4, filters * 4, 3, 1, 1),  # k3n256s1
            nn.InstanceNorm2d(filters * 4),
            nn.ReLU(True),
        )

        # residual blocks
        self.residual_blocks = []
        for i in range(res_num):
            self.residual_blocks.append(residual_block(filters * 4, 3, 1, 1))

        self.residual_blocks = nn.Sequential(*self.residual_blocks)

        # up convolution
        self.up_convs = nn.Sequential(
            nn.ConvTranspose2d(filters * 4, filters * 2, 3, 2, 1, 1),  # k3n128s1/2
            nn.Conv2d(filters * 2, filters * 2, 3, 1, 1),  # k3n128s1
            nn.InstanceNorm2d(filters * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(filters * 2, filters, 3, 2, 1, 1),  # k3n64s1/2
            nn.Conv2d(filters, filters, 3, 1, 1),  # k3n64s1
            nn.InstanceNorm2d(filters),
            nn.ReLU(True),

            nn.Conv2d(filters, out_channel, 7, 1, 3),  # k7n3s1
            nn.Tanh(),
        )

        utils.initialize_weights(self)

    def forward(self, input):
        down = self.down_convs(input)
        res = self.residual_blocks(down)
        output = self.up_convs(res)
        return output


# model for the discriminator
class discriminator(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, filters=32):
        """
            filters: the number of filters/feature maps in the first conv layers
        """
        super(discriminator, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, filters, 3, 1, 1),  # k3n32s1
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(filters, filters * 2, 3, 2, 1),  # k3n64s2
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(filters * 2, filters * 4, 3, 1, 1),  # k3n128s1
            nn.InstanceNorm2d(filters * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(filters * 4, filters * 4, 3, 2, 1),  # k3n128s2
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(filters * 4, filters * 8, 3, 1, 1),  # k3n256s1
            nn.InstanceNorm2d(filters * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(filters * 8, filters * 8, 3, 1, 1),  # k3n256s1
            nn.InstanceNorm2d(filters * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(filters * 8, out_channel, 3, 1, 1),  # k3n1s1
            nn.Sigmoid(),
        )

        utils.initialize_weights(self)

    def forward(self, input):
        output = self.convs(input)
        return output
