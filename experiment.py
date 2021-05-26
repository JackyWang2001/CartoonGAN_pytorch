import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import model
import utils
from dataset import MyDataset


class Experiment:
    def __init__(self, config_file="config.json"):
        # read config.json file
        if os.path.isfile(config_file):
            with open(config_file) as json_file:
                config = json.load(json_file)
                self.config = config
        else:
            raise Exception("file does not exist: %s" % config_file)
        # read in
        root = config["dataset"]["root"]
        self.root = os.path.abspath(root)
        self.num_epoch = config["num_epoch"]
        self.batch_size = config["dataset"]["batch_size"]
        self.G_path = config["model"]["G_path"]
        self.D_path = config["model"]["D_path"]
        # initialize dataset
        train_real_dataset = MyDataset(self.root, style="real", mode="train")
        train_anim_dataset = MyDataset(self.root, style="violet", mode="train")
        # TODO: edge promoting
        if not os.path.isdir(os.path.join('data', 'edge_smoothed')):
            src_dir = os.path.join('data', 'real')
            target_dir = os.path.join('data', 'edge_smoothed')
            utils.edge_promoting(src_dir, target_dir)
        else:
            print("edge-promoting already done %s" % os.path.join(self.root, "edge_smoothed"))
        # initialize dataset
        train_real_dataset = MyDataset(self.root, style="real", mode="train")
        train_anim_dataset = MyDataset(self.root, style="edge_smoothed", mode="")

        val_real_dataset = MyDataset(self.root, style="real", mode="valid")
        val_anim_dataset = MyDataset(self.root, style="violet", mode="valid")
        test_dataset = MyDataset(self.root, style="real", mode="test")
        self.train_real_loader = DataLoader(train_real_dataset, batch_size=self.batch_size, shuffle=True,
                                            num_workers=48)
        self.train_anim_loader = DataLoader(train_anim_dataset, batch_size=self.batch_size, shuffle=True,
                                            num_workers=48)
        self.val_real_loader = ...
        self.val_anim_loader = ...

        self.test_loader = ...

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device: ", self.device)

        # initialize Discriminator and Generator
        self.D = model.discriminator()
        self.D.to(self.device)
        self.D.train()      # put model to training mode
        self.G = model.generator()
        self.G.to(self.device)
        self.G.train()

        # initialize optimizer
        self.D_optimizer = optim.Adam(self.D.parameters(), config["optim"]["D_lr"], betas=(0.9, 0.99))
        self.G_optimizer = optim.Adam(self.G.parameters(), config["optim"]["G_lr"], betas=(0.9, 0.99))

        # initialize loss function
        self.BCE_loss = nn.BCELoss().to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)

        # initialize scheduler
        self.D_scheduler = MultiStepLR(self.D_optimizer, config["optim"]["D_step"], config["optim"]["D_gamma"])
        self.G_scheduler = MultiStepLR(self.G_optimizer, config["optim"]["G_step"], config["optim"]["G_gamma"])

    def _train(self):
        """
        train the model for 1 epoch
        :return:
        """

        for src, anim in (self.train_real_loader, self.train_anim_loader):
            src, anim = src.to(self.device), anim.to(self.device)

            # train discriminator...

            # discriminate real anime image
            D_real = self.D(anim)
            D_real_loss = self.BCE_loss(D_real)

            # discriminate generated/fake anime image
            fake_anim = self.G(src)
            D_fake = self.D(fake_anim)
            D_fake_loss = self.BCE_loss(D_fake)

            # TODO: discriminate real anime image without clear edges
            D_edge = 0
            D_edge_loss = self.BCE_loss(D_edge)

            D_loss = D_real_loss + D_fake_loss + D_edge_loss
            self.D_optimizer.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()

            # train generator...

            # generated/fake anime image
            fake_anim = self.G(src)
            D_fake = self.D(fake_anim)
            D_fake_loss = self.BCE_loss(D_fake)

            # TODO: content loss (L1)
            src_feature = 0
            G_feature = 0
            Content_loss = self.L1_loss(G_feature, src_feature)

            G_loss = D_fake_loss + Content_loss
            self.G_optimizer.zero_grad()
            G_loss.backward()
            self.G_optimizer.step()

        return

    def _valid(self):
        save_results = self.config["valid"]["save_results"]
        save_num = self.config["valid"]["save_num"]
        for src, tgt in self.val_real_loader, self.val_anim_loader:
            src, tgt = src.to(self.device), tgt.to(self.device)
            # TODO: compute loss
            outputs = self.G(src)
            outputs = outputs.to("cpu").numpy()  # [B, C=3, H=256, W=256]
            B, C, H, W = outputs.shape
            # if save is true in config
            if save_results:
                idx = np.random.shuffle()
        return

    def run(self):
        for i in range(self.num_epoch):
            self._train()
            self._valid()
        self._test()

    def _save_model(self, epoch, D_state, G_state, D_optim_state, G_optim_state):
        """ save model """
        torch.save({"epoch": epoch, "D_state": D_state, "D_optim_state": D_optim_state}, os.path.join(self.D_path))
        torch.save({"epoch": epoch, "G_state": G_state, "G_optim_state": G_optim_state}, os.path.join(self.G_path))

    def _test(self):
        return