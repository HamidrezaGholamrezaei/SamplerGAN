"""
Generator and discriminator implementation for InfoGAN

"""

import torch
import torch.nn as nn

from models import modelLayers
from utils import initialize_weights


class Generator(nn.Module):
    def __init__(self, num_gpu, input_dim, output_dim, label_dim, code_dim, image_size, dataset_name):
        """InfoCGAN generator implementation for the image domain

        Notes:
            This architecture is inspired from https://arxiv.org/abs/1606.03657

        Args:
            num_gpu (int): Number of gpu.
            input_dim (int): Dimension of the input noise vector to generate fake data.
            output_dim (int): Dimension of output (equal to channel_dim of images in the dataset).
            label_dim (int): Number of classes in the dataset.
            code_dim (int): Dimension of continuous conditional information.
            image_size (int): Size of input image.
            dataset_name (str): Name of the dataset: MNIST/CIFAR10.
        """

        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.gen_input_dim = input_dim + label_dim + code_dim
        self.output_dim = output_dim
        self.image_size = image_size
        self.dataset_name = dataset_name

        if self.dataset_name == "MNIST":
            self.fc = nn.Sequential(
                *modelLayers.linear_block(self.gen_input_dim, 1024),
                *modelLayers.linear_block(1024, 128 * (self.image_size // 4) * (self.image_size // 4)),
            )
            self.model = nn.Sequential(
                *modelLayers.convT_block(128, 64),
                nn.ConvTranspose2d(64, self.output_dim, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
            initialize_weights(self)
        elif self.dataset_name == "CIFAR10":
            self.fc = nn.Sequential(
                *modelLayers.linear_block(self.gen_input_dim, 1024),
                *modelLayers.linear_block(1024, 256 * (self.image_size // 8) * (self.image_size // 8)),
            )
            self.model = nn.Sequential(
                *modelLayers.convT_block(256, 128),
                *modelLayers.convT_block(128, 64),
                nn.ConvTranspose2d(64, self.output_dim, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
            initialize_weights(self)

    def forward(self, noise, labels, codes=None):
        """
        Args:
            noise: Input noise vector, of shape (batch_size, input_dim).
            labels: Input label vector, of shape (batch_size, label_dim).
            codes: Input continuous condition vector, of shape (batch_size, code_dim).
        Return:
            out: Generated image data, of shape (batch_size, output_dim, image_size, image_size).
        """

        x = torch.cat([noise, labels], 1)
        if codes is not None:
            x = torch.cat([x, codes], 1)
        if noise.is_cuda and self.num_gpu > 1:
            h = nn.parallel.data_parallel(self.fc, x, range(self.ngpu))
            if self.dataset_name == "MNIST":
                h = h.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
            elif self.dataset_name == "CIFAR10":
                h = h.view(-1, 256, (self.image_size // 8), (self.image_size // 8))
            out = nn.parallel.data_parallel(self.model, h, range(self.ngpu))
        else:
            h = self.fc(x)
            if self.dataset_name == "MNIST":
                h = h.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
            elif self.dataset_name == "CIFAR10":
                h = h.view(-1, 256, (self.image_size // 8), (self.image_size // 8))
            out = self.model(h)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_gpu, input_dim, output_dim, label_dim, code_dim, image_size, dataset_name):
        """InfoCGAN discriminator implementation for the image domain

        Notes:
            This architecture is inspired from https://arxiv.org/abs/1606.03657

        Args:
            num_gpu (int): Number of gpu.
            input_dim (int): Dimension of the input (equal to channel_dim of images in the dataset).
            output_dim (int): Dimension of output.
            label_dim (int): Number of classes in the dataset.
            code_dim (int): Dimension of continuous conditional information.
            image_size (int): Size of input image.
            dataset_name (str): Name of the dataset: MNIST/CIFAR10.
        """

        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.label_dim = label_dim
        self.code_dim = code_dim
        self.image_size = image_size
        self.dataset_name = dataset_name

        if self.dataset_name == "MNIST":
            self.model = nn.Sequential(
                *modelLayers.conv_block(self.input_dim, 64, normalize=False, lrelu=0.1),
                *modelLayers.conv_block(64, 128, lrelu=0.1),
            )
            self.fc = nn.Sequential(
                *modelLayers.linear_block(128 * (self.image_size // 4) * (self.image_size // 4), 1024,
                                          activation="LeakyReLU", lrelu=0.1),
            )
            self.module_d = nn.Sequential(
                nn.Linear(1024, self.output_dim),
                nn.Sigmoid(),
            )
            initialize_weights(self)
        elif self.dataset_name == "CIFAR10":
            self.model = nn.Sequential(
                *modelLayers.conv_block(self.input_dim, 64, normalize=False, lrelu=0.1),
                *modelLayers.conv_block(64, 128, lrelu=0.1),
                *modelLayers.conv_block(128, 256, lrelu=0.1),
            )
            self.fc = nn.Sequential(
                *modelLayers.linear_block(256 * (self.image_size // 8) * (self.image_size // 8), 1024,
                                          activation="LeakyReLU", lrelu=0.1),
            )
            self.module_d = nn.Sequential(
                nn.Linear(1024, 1),
                nn.Sigmoid(),
            )
            initialize_weights(self)

    def forward(self, data):
        """
        Args:
            data: Input image data, of shape (batch_size, input_dim, image_size, image_size).
        Return:
            d_output: Classification result vector, of shape (batch_size, 1).
            shared_output: Data representations for Q-Module, of shape (batch_size, 1024).
        """

        if data.is_cuda and self.num_gpu > 1:
            x = nn.parallel.data_parallel(self.model, data, range(self.ngpu))
            if self.dataset_name == "MNIST":
                x = x.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))
            elif self.dataset_name == "CIFAR10":
                x = x.view(-1, 256 * (self.image_size // 8) * (self.image_size // 8))
            shared_output = nn.parallel.data_parallel(self.fc, x, range(self.ngpu))
            d_output = nn.parallel.data_parallel(self.d, shared_output, range(self.ngpu))
        else:
            x = self.model(data)
            if self.dataset_name == "MNIST":
                x = x.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))
            elif self.dataset_name == "CIFAR10":
                x = x.view(-1, 256 * (self.image_size // 8) * (self.image_size // 8))
            shared_output = self.fc(x)
            d_output = self.module_d(shared_output)

        return d_output, shared_output


class Q_Module(nn.Module):
    def __init__(self, label_dim, code_dim):
        """InfoCGAN classifier (Q_Module) implementation for the image domain

        Notes:
            This architecture is inspired from https://arxiv.org/abs/1606.03657

        Args:
            label_dim (int): Number of classes in the dataset.
            code_dim (int): Dimension of continuous conditional information.
        """

        super(Q_Module, self).__init__()

        self.label_dim = label_dim
        self.code_dim = code_dim
        self.fc = nn.Sequential(
            *modelLayers.linear_block(1024, 128, activation="LeakyReLU", lrelu=0.1),
        )
        self.q_disc = nn.Linear(128, self.label_dim)
        self.q_mu = nn.Linear(128, self.code_dim + 1)
        self.q_sigma = nn.Linear(128, self.code_dim + 1)
        initialize_weights(self)

    def forward(self, data):
        """
        Args:
            data: Input vector, of shape (batch_size, 1024).
        Return:
            disc: Classification result vector for labels, of shape (batch_size, label_dim).
            mu: mean values for the Gaussian distribution of continuous conditions, of shape (batch_size, code_dim + 1).
            sigma: std values for the Gaussian distribution of continuous conditions, of shape (batch_size, code_dim + 1).
        """

        out = self.fc(data)
        disc = self.q_disc(out)
        mu = self.q_mu(out)
        sigma = torch.exp(self.q_sigma(out).squeeze())
        return disc, mu, sigma
