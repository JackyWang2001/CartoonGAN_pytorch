import os
import json

import torch

from experiment import Experiment

# read in config file
config_file = "config.json"
if os.path.isfile(config_file):
    with open(config_file) as json_file:
        config = json.load(json_file)

train_epoch = config["num_epoch"]

exp = Experiment()

# training
D_losses = []
G_losses = []
Content_losses = []
for epoch in range(train_epoch):
    D_loss, G_loss, Content_loss = exp.train()
    D_losses += D_loss
    G_losses += G_loss
    Content_losses += Content_loss

    print("Epoch: %s, Discriminator loss: %.3f, Generator loss: %.3f, Content loss: %.3f" %
          (epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.mean(Content_losses))))



