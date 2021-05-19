import torch.nn as nn


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            print("Conv2d")
        elif isinstance(m, nn.ConvTranspose2d):
            print("ConvTranspose2d")
        elif isinstance(m, nn.Linear):
            print("Linear")
        elif isinstance(m, nn.InstanceNorm2d):
            print("instanceNorm2d")
