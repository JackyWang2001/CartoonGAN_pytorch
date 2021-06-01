import os
import json
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torchvision

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
        # edge promoting
        if not os.path.isdir(os.path.join(self.root, "edge_smoothed")):
            src_dir = os.path.join(self.root, "violet", "train")
            target_dir = os.path.join(self.root, "edge_smoothed")
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
        self.val_real_loader = DataLoader(val_real_dataset, batch_size=self.batch_size, shuffle=True, num_workers=48)
        self.val_anim_loader = DataLoader(val_anim_dataset, batch_size=self.batch_size, shuffle=True, num_workers=48)

        self.test_loader = ...

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Using device: ", self.device)

        # initialize Discriminator and Generator
        self.D = model.discriminator()
        self.D.to(self.device)

        self.G = model.generator()
        self.G.to(self.device)

        # initialize vgg19 pretrained model
        self.vgg19 = torchvision.models.mobilenet_v2(pretrained=True)
        self.vgg19.to(self.device)
        self.vgg19.eval()

        # initialize optimizer
        self.D_optimizer = optim.Adam(self.D.parameters(), config["optim"]["D_lr"], betas=(0.9, 0.99))
        self.G_optimizer = optim.Adam(self.G.parameters(), config["optim"]["G_lr"], betas=(0.9, 0.99))

        # masks
        self.real_mask = torch.ones(self.batch_size, 1, 256 // 4, 256 // 4).to(self.device)
        self.fake_mask = torch.zeros(self.batch_size, 1, 256 // 4, 256 // 4).to(self.device)

        # initialize loss function
        self.BCE_loss = nn.BCELoss().to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)
        self.content_loss_lambda = 1

        # initialize scheduler
        self.D_scheduler = MultiStepLR(self.D_optimizer, config["optim"]["D_step"], config["optim"]["D_gamma"])
        self.G_scheduler = MultiStepLR(self.G_optimizer, config["optim"]["G_step"], config["optim"]["G_gamma"])

    def _train(self):
        """
        train the model for 1 epoch
        :return:
        """
        # put model to training mode
        self.D.train()
        self.G.train()

        # arrays to store the losses
        D_losses = []
        G_losses = []
        Content_losses = []

        for i, data in enumerate(zip(self.train_real_loader, self.train_anim_loader)):
            src, anim = data[0], data[1]
            origin_anim = anim[:, :, :, :256]
            edge_smooth_anim = anim[:, :, :, 256:]
            src = src.to(self.device)
            edge_smooth_anim, origin_anim = edge_smooth_anim.to(self.device), origin_anim.to(self.device)

            # train discriminator...

            # discriminate real anime image
            D_real = self.D(origin_anim)
            D_real_loss = self.BCE_loss(D_real, self.real_mask)

            # discriminate generated/fake anime image
            fake_anim = self.G(src)
            D_fake = self.D(fake_anim)
            D_fake_loss = self.BCE_loss(D_fake, self.fake_mask)

            # discriminate real anime image without clear edges
            D_edge = self.D(edge_smooth_anim)
            D_edge_loss = self.BCE_loss(D_edge, self.fake_mask)

            D_loss = D_real_loss + D_fake_loss + D_edge_loss
            self.D_optimizer.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()

            # train generator...

            # generated/fake anime image
            fake_anim = self.G(src)
            D_fake = self.D(fake_anim)
            D_fake_loss = self.BCE_loss(D_fake, self.real_mask)

            # content loss (L1)
            src_feature = self.vgg19((src + 1) / 2)
            G_feature = self.vgg19((fake_anim + 1) / 2)
            Content_loss = self.content_loss_lambda * self.L1_loss(G_feature, src_feature.detach())

            G_loss = D_fake_loss + Content_loss
            self.G_optimizer.zero_grad()
            G_loss.backward()
            self.G_optimizer.step()

            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())
            Content_losses.append(Content_loss.item())

        print("Discriminator loss: %.3f, Generator loss: %.3f, Content loss: %.3f" % (np.mean(D_losses),
                                                                                      np.mean(G_losses),
                                                                                      np.mean(Content_losses)))

        self.G_scheduler.step()
        self.D_scheduler.step()
        return D_losses, G_losses, Content_losses

    def _valid(self, e):  # use e for image names
        self.G.eval()
        save_results = self.config["valid"]["save_results"]
        save_path = os.path.join(self.config["valid"]["save_path"])
        save_num = self.config["valid"]["save_num"]

        # arrays to store the losses
        D_losses = []
        G_losses = []
        for src, tgt in self.val_real_loader, self.val_anim_loader:
            src, tgt = src.to(self.device), tgt.to(self.device)
            outputs = self.G(src)

            # masks
            # real_mask = torch.ones(src.shape[0], 3, 256 // 4, 256 // 4, device=self.device)
            # fake_mask = torch.zeros(tgt.shape[0], 3, 256 // 4, 256 // 4, device=self.device)

            # # discriminate real anime image
            # D_real = self.D(tgt)
            # D_real_loss = self.BCE_loss(D_real, torch.ones_like(D_real, device=self.device))
            #
            # # discriminate generated/fake anime image
            # fake_anim = self.G(src)
            # D_fake = self.D(fake_anim)
            # D_fake_loss = self.BCE_loss(D_fake, torch.zeros_like(D_fake, device=self.device))
            #
            # D_loss = D_real_loss + D_fake_loss
            # D_losses.append(D_loss.item())
            #
            # # generated/fake anime image
            # fake_anim = self.G(src)
            # D_fake = self.D(fake_anim)
            # D_fake_loss = self.BCE_loss(D_fake, torch.ones_like(D_fake, device=self.device))
            #
            # # content loss (L1)
            # src_feature = self.vgg19((src + 1) / 2)
            # G_feature = self.vgg19((fake_anim + 1) / 2)
            # Content_loss = self.content_loss_lambda * self.L1_loss(G_feature, src_feature.detach())
            #
            # G_loss = D_fake_loss + Content_loss
            # G_losses.append(G_loss.item())
            # Content_losses.append(Content_loss.item())

            # visualize generated samples
            B, C, H, W = outputs.shape
            outputs = outputs.to("cpu").numpy()  # [B, C=3, H=256, W=256]
            outputs = outputs.transpose(0, 2, 3, 1)
            # if save is true in config
            if save_results:
                idx = np.arange(B)
                np.random.shuffle(idx)
                idx = idx[:save_num]
                for i in range(save_num):
                    filename = "%s_%s_%s.png" % (e, batch_idx, i)
                    output = outputs[idx[i], :, :, :]
                    output = Image.fromarray(output, mode="RGB")
                    output.save(os.path.join(save_path, filename))

    def run(self):
        for e in range(self.num_epoch):
            self._train()
            self._valid(e)
        # self._test()

    def _save_model(self, epoch, D_state, G_state, D_optim_state, G_optim_state):
        """ save model """
        torch.save({"epoch": epoch, "D_state": D_state, "D_optim_state": D_optim_state}, os.path.join(self.D_path))
        torch.save({"epoch": epoch, "G_state": G_state, "G_optim_state": G_optim_state}, os.path.join(self.G_path))

    def _test(self):
        return