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
import matplotlib.pyplot as plt

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
        self.warmup_epoch = config["train"]["G_warming"]
        self.batch_size = config["dataset"]["batch_size"]
        self.G_path = config["model"]["G_path"]
        self.D_path = config["model"]["D_path"]
        # edge promoting
        if not os.path.isdir(os.path.join(self.root, "edge_smoothed")):
            src_dir = os.path.join(self.root, "violet", "train")
            target_dir = os.path.join(self.root, "edge_smoothed")
            utils.edge_promoting(src_dir, target_dir)
        else:
            print("edge-promoting already done %s" %
                  os.path.join(self.root, "edge_smoothed"))
        # initialize dataset
        train_real_dataset = MyDataset(self.root, style="real", mode="train")
        train_anim_dataset = MyDataset(
            self.root, style="edge_smoothed", mode="")

        val_real_dataset = MyDataset(self.root, style="real", mode="valid")
        val_anim_dataset = MyDataset(self.root, style="violet", mode="valid")
        test_dataset = MyDataset(self.root, style="real", mode="test")
        self.train_real_loader = DataLoader(train_real_dataset, batch_size=self.batch_size, shuffle=True,
                                            num_workers=12)
        self.train_anim_loader = DataLoader(train_anim_dataset, batch_size=self.batch_size, shuffle=True,
                                            num_workers=12)
        self.val_real_loader = DataLoader(
            val_real_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12)
        self.val_anim_loader = DataLoader(
            val_anim_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12)

        self.test_loader = ...

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Using device: ", self.device)

        # initialize Discriminator and Generator
        self.D = model.discriminator()
        self.D.to(self.device)

        self.G = model.generator()
        self.G.to(self.device)

        # initialize vgg19 pretrained model
        self.vgg19 = torchvision.models.vgg19(pretrained=True)
        self.vgg19.to(self.device)
        self.vgg19.eval()

        # initialize optimizer
        self.D_optimizer = optim.Adam(
            self.D.parameters(), config["optim"]["D_lr"], betas=(0.5, 0.99))
        self.G_optimizer = optim.Adam(
            self.G.parameters(), config["optim"]["G_lr"], betas=(0.5, 0.99))

        # initialize loss function
        self.BCE_loss = nn.BCELoss().to(self.device)
        self.L1_loss = nn.L1Loss().to(self.device)
        self.content_loss_lambda = 10

        # initialize scheduler
        self.D_scheduler = MultiStepLR(
            self.D_optimizer, config["optim"]["D_step"], config["optim"]["D_gamma"])
        self.G_scheduler = MultiStepLR(
            self.G_optimizer, config["optim"]["G_step"], config["optim"]["G_gamma"])

    def _train(self, e):
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
            edge_smooth_anim, origin_anim = edge_smooth_anim.to(
                self.device), origin_anim.to(self.device)

            # train discriminator...

            # discriminate real anime image
            D_real = self.D(origin_anim)
            D_real_loss = self.BCE_loss(
                D_real, torch.ones_like(D_real, device=self.device))

            # discriminate generated/fake anime image
            fake_anim = self.G(src)
            D_fake = self.D(fake_anim)
            D_fake_loss = self.BCE_loss(
                D_fake, torch.zeros_like(D_fake, device=self.device))

            # discriminate real anime image without clear edges
            D_edge = self.D(edge_smooth_anim)
            D_edge_loss = self.BCE_loss(
                D_edge, torch.zeros_like(D_edge, device=self.device))

            D_loss = D_real_loss + D_fake_loss + D_edge_loss
            self.D_optimizer.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()

            # train generator...

            # generated/fake anime image
            fake_anim = self.G(src)
            D_fake = self.D(fake_anim)
            D_fake_loss = self.BCE_loss(
                D_fake, torch.ones_like(D_fake, device=self.device))

            # content loss (L1)
            src_feature = self.vgg19((src + 1) / 2)
            G_feature = self.vgg19((fake_anim + 1) / 2)
            Content_loss = self.content_loss_lambda * self.L1_loss(
                G_feature, src_feature.detach())

            G_loss = D_fake_loss + Content_loss
            self.G_optimizer.zero_grad()
            G_loss.backward()
            self.G_optimizer.step()

            print("Epoch: %s, Index: %s, Discriminator loss: %.3f, Generator loss: %.3f, Content loss: %.3f" % (e, i,
                                                                                                                D_loss.item(), G_loss.item(), Content_loss.item()))
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())
            Content_losses.append(Content_loss.item())

        average_D_loss = np.mean(D_losses)
        average_G_loss = np.mean(G_losses)
        average_content_loss = np.mean(Content_losses)

        print()
        print("Average: Epoch: %s, Discriminator loss: %.3f, Generator loss: %.3f, Content loss: %.3f" % (e, average_D_loss,
                                                                                                          average_G_loss,
                                                                                                          average_content_loss))
        print()

        self.G_scheduler.step()
        self.D_scheduler.step()
        return average_D_loss, average_G_loss, average_content_loss

    def _train_warming(self, e):
        """
        warm up the model for 1 epoch
        :return:
        """
        # put generator to training mode
        self.G.train()

        # arrays to store the losses
        Content_losses = []

        for i, src in enumerate(self.train_real_loader):
            src = src.to(self.device)

            # train generator

            # generated/fake anime image
            fake_anim = self.G(src)

            # content loss (L1)
            src_feature = self.vgg19((src + 1) / 2)
            G_feature = self.vgg19((fake_anim + 1) / 2)
            Content_loss = self.content_loss_lambda * self.L1_loss(
                G_feature, src_feature.detach())

            self.G_optimizer.zero_grad()
            Content_loss.backward()
            self.G_optimizer.step()

            print("Epoch: %s, Index: %s, Content loss: %.3f" %
                  (e, i, Content_loss.item()))
            Content_losses.append(Content_loss.item())

        average_content_loss = np.mean(Content_losses)
        print()
        print("Epoch: %s, Average content loss: %.3f" %
              (e, average_content_loss))
        print()
        return average_content_loss

    def _valid(self, e, pretrain=False):  # use e for image names
        save_path = os.path.join(self.config["valid"]["save_path"])
        with torch.no_grad():
            self.G.eval()
            for i, src in enumerate(self.val_real_loader):
                src = src.to(self.device)
                generated_img = self.G(src)
                result = torch.cat((src[0], generated_img[0]), 2)
                result = (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2

                filename = ""
                if pretrain == True:
                    filename = "pretrain_%s_%s.png" % (e, i)
                elif pretrain == False:
                    filename = "during_train_%s_%s.png" % (e, i)
                path = os.path.join(save_path, filename)

                plt.imsave(path, result)
                if i == 6:
                    break

    def run(self):
        warm_up_content_losses = []      # store the average loss at each epoch
        training_D_losses = []
        training_G_losses = []
        training_content_losses = []

        print("start warming up")
        for e in range(self.warmup_epoch):
            curr_content_loss = self._train_warming(e)
            warm_up_content_losses.append(curr_content_loss)
            self._valid(e, True)

        print("start training and validating")
        for e in range(self.num_epoch):
            curr_D_loss, curr_G_loss, curr_content_loss = self._train(e)
            training_D_losses.append(curr_D_loss)
            training_G_losses.append(curr_G_loss)
            training_content_losses.append(curr_content_loss)
            self._valid(e, False)

        return warm_up_content_losses, training_D_losses, training_G_losses, training_content_losses

    def _save_model(self, epoch, D_state, G_state, D_optim_state, G_optim_state):
        """ save model """
        torch.save({"epoch": epoch, "D_state": D_state,
                   "D_optim_state": D_optim_state}, os.path.join(self.D_path))
        torch.save({"epoch": epoch, "G_state": G_state,
                   "G_optim_state": G_optim_state}, os.path.join(self.G_path))

    def _test(self):
        return
