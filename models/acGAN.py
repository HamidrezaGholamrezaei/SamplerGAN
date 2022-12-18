"""
Generator and discriminator implementation for ACGAN

"""

import torch.nn as nn

from models import modelLayers
from utils import initialize_weights


class Generator(nn.Module):
    def __init__(self, num_gpu, input_dim, output_dim, label_dim, image_size, dataset_name):
        """ACGAN generator implementation for the image domain

        Notes:
            This architecture is inspired from https://arxiv.org/abs/1610.09585

        Args:
            num_gpu (int): Number of gpu.
            input_dim (int): Dimension of the input noise vector to generate fake data.
            output_dim (int): Dimension of output (equal to channel_dim of images in the dataset).
            label_dim (int): Number of classes in the dataset.
            image_size (int): Size of input image.
            dataset_name (str): Name of the dataset: MNIST/CIFAR10.
        """

        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.gen_input_dim = input_dim + label_dim
        self.output_dim = output_dim
        self.image_size = image_size
        self.dataset_name = dataset_name

        if self.dataset_name == "MNIST":
            self.fc = nn.Sequential(
                *modelLayers.linear_block(self.gen_input_dim, 128 * (self.image_size // 4) * (self.image_size // 4),
                                          normalize=False),
            )
            self.model = nn.Sequential(
                *modelLayers.convT_block(128, 64),
                nn.ConvTranspose2d(64, self.output_dim, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
            initialize_weights(self)
        elif self.dataset_name == "CIFAR10":
            self.fc = nn.Sequential(
                *modelLayers.linear_block(self.gen_input_dim, 384, normalize=False, activation="None"),
            )
            self.model = nn.Sequential(
                *modelLayers.convT_block(384, 192, kernel_size=4, stride_size=1, padding_size=0),
                *modelLayers.convT_block(192, 96, kernel_size=4, stride_size=2, padding_size=1),
                *modelLayers.convT_block(96, 48, kernel_size=4, stride_size=2, padding_size=1),
                nn.ConvTranspose2d(48, self.output_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh(),
            )
            initialize_weights(self)

    def forward(self, noise):
        """
        Args:
            noise: Input vector containing noise plus labels, of shape (batch_size, input_dim + label_dim).
        Return:
            out: Generated image data, of shape (batch_size, output_dim, image_size, image_size).
        """

        if noise.is_cuda and self.num_gpu > 1:
            h = nn.parallel.data_parallel(self.fc, noise, range(self.ngpu))
            if self.dataset_name == "MNIST":
                h = h.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
            elif self.dataset_name == "CIFAR10":
                h = h.view(-1, 384, 1, 1)
            out = nn.parallel.data_parallel(self.model, h, range(self.ngpu))
        else:
            h = self.fc(noise)
            if self.dataset_name == "MNIST":
                h = h.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
            elif self.dataset_name == "CIFAR10":
                h = h.view(-1, 384, 1, 1)
            out = self.model(h)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_gpu, input_dim, output_dim, label_dim, image_size, dataset_name):
        """ACGAN discriminator implementation for the image domain

        Notes:
            This architecture is inspired from https://arxiv.org/abs/1610.09585

        Args:
            num_gpu (int): Number of gpu.
            input_dim (int): Dimension of the input (equal to channel_dim of images in the dataset).
            output_dim (int): Dimension of output.
            label_dim (int): Number of classes in the dataset.
            image_size (int): Size of input image.
            dataset_name (str): Name of the dataset: MNIST/CIFAR10.
        """

        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.label_dim = label_dim
        self.image_size = image_size
        self.dataset_name = dataset_name

        if self.dataset_name == "MNIST":
            self.model = nn.Sequential(
                *modelLayers.conv_block(self.input_dim, 64, normalize=False),
                *modelLayers.conv_block(64, 128, dropout=0.25),
            )
            self.d = nn.Sequential(
                nn.Linear(128 * (self.image_size // 4) * (self.image_size // 4), self.output_dim),
                nn.Sigmoid(),
            )
            self.q = nn.Sequential(
                nn.Linear(128 * (self.image_size // 4) * (self.image_size // 4), self.label_dim),
                # nn.Softmax(),
            )
            initialize_weights(self)
        elif self.dataset_name == "CIFAR10":
            self.model = nn.Sequential(
                *modelLayers.conv_block(self.input_dim, 16, kernel_size=3, stride_size=2, normalize=False, dropout=0.5),
                *modelLayers.conv_block(16, 32, kernel_size=3, stride_size=1, dropout=0.5),
                *modelLayers.conv_block(32, 64, kernel_size=3, stride_size=2, dropout=0.5),
                *modelLayers.conv_block(64, 128, kernel_size=3, stride_size=1, dropout=0.5),
                *modelLayers.conv_block(128, 256, kernel_size=3, stride_size=2, dropout=0.5),
                *modelLayers.conv_block(256, 512, kernel_size=3, stride_size=1, dropout=0.5),
            )
            self.d = nn.Sequential(
                nn.Linear(512 * (self.image_size // 8) * (self.image_size // 8), self.output_dim),
                nn.Sigmoid(),
            )
            self.q = nn.Sequential(
                nn.Linear(512 * (self.image_size // 8) * (self.image_size // 8), self.label_dim),
                # nn.Softmax(),
            )
            initialize_weights(self)

    def forward(self, data):
        """
        Args:
            data: Input image data, of shape (batch_size, input_dim, image_size, image_size).
        Return:
            d_output: Classification result vector, of shape (batch_size, 1).
            q_output: Classification result vector for labels, of shape (batch_size, label_dim).
        """

        if data.is_cuda and self.num_gpu > 1:
            out = nn.parallel.data_parallel(self.model, data, range(self.ngpu))
            if self.dataset_name == "MNIST":
                out = out.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))
            elif self.dataset_name == "CIFAR10":
                out = out.view(-1, 512 * (self.image_size // 8) * (self.image_size // 8))
            d_output = nn.parallel.data_parallel(self.d, out, range(self.ngpu))
            q_output = nn.parallel.data_parallel(self.q, out, range(self.ngpu))
        else:
            out = self.model(data)
            if self.dataset_name == "MNIST":
                out = out.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))
            elif self.dataset_name == "CIFAR10":
                out = out.view(-1, 512 * (self.image_size // 8) * (self.image_size // 8))
            d_output = self.d(out)
            q_output = self.q(out)

        return d_output, q_output
