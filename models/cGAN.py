"""
Generator and discriminator implementation for CGAN

"""

import torch
import torch.nn as nn

from models import modelLayers
from utils import initialize_weights


class Generator(nn.Module):
    def __init__(self, num_gpu, input_dim, output_dim, label_dim, dataset_name):
        """CGAN generator implementation for the image domain

        Args:
            num_gpu (int): Number of gpu.
            input_dim (int): Dimension of the input noise vector to generate fake data.
            output_dim (int): Dimension of output (equal to channel_dim of images in the dataset).
            label_dim (int): Number of classes in the dataset.
            dataset_name (str): Name of the dataset: MNIST/CIFAR10.
        """

        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.label_dim = label_dim
        self.dataset_name = dataset_name

        if self.dataset_name == "MNIST":
            self.input_layer_noise = nn.Sequential(
                *modelLayers.convT_block(self.input_dim, 256, 4, 1, 0),
            )
            self.input_layer_labels = nn.Sequential(
                *modelLayers.convT_block(self.label_dim, 256, 4, 1, 0),
            )
            self.model = nn.Sequential(
                *modelLayers.convT_block(512, 256),
                *modelLayers.convT_block(256, 128),
                nn.ConvTranspose2d(128, self.output_dim, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
            initialize_weights(self)
        elif self.dataset_name == "CIFAR10":
            self.input_layer_noise = nn.Sequential(
                *modelLayers.convT_block(self.input_dim, 512, 4, 1, 0),
            )
            self.input_layer_labels = nn.Sequential(
                *modelLayers.convT_block(self.label_dim, 512, 4, 1, 0),
            )
            self.model = nn.Sequential(
                *modelLayers.convT_block(1024, 512),
                *modelLayers.convT_block(512, 256),
                *modelLayers.convT_block(256, 128),
                nn.ConvTranspose2d(128, self.output_dim, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            )
            initialize_weights(self)

    def forward(self, noise, labels):
        """
        Args:
            noise: Input noise vector, of shape (batch_size, input_dim, 1, 1).
            labels: Input label vector, of shape (batch_size, label_dim, 1, 1).
        Return:
            out: Generated image data, of shape (batch_size, output_dim, image_size, image_size).
        """

        if noise.is_cuda and self.num_gpu > 1:
            h1 = nn.parallel.data_parallel(self.input_layer_noise, noise, range(self.ngpu))
            h2 = nn.parallel.data_parallel(self.input_layer_labels, labels, range(self.ngpu))
            x = torch.cat([h1, h2], 1)
            out = nn.parallel.data_parallel(self.model, x, range(self.ngpu))
        else:
            h1 = self.input_layer_noise(noise)
            h2 = self.input_layer_labels(labels)
            x = torch.cat([h1, h2], 1)
            out = self.model(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_gpu, input_dim, output_dim, label_dim, dataset_name):
        """CGAN discriminator implementation for the image domain

        Args:
            num_gpu (int): Number of gpu.
            input_dim (int): Dimension of the input (equal to channel_dim of images in the dataset).
            output_dim (int): Dimension of output.
            label_dim (int): Number of classes in the dataset.
            dataset_name (str): Name of the dataset: MNIST/CIFAR10.
        """

        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.label_dim = label_dim
        self.dataset_name = dataset_name

        if self.dataset_name == "MNIST":
            self.input_layer_data = nn.Sequential(
                *modelLayers.conv_block(self.input_dim, 64, normalize=False),
            )
            self.input_layer_labels = nn.Sequential(
                *modelLayers.conv_block(self.label_dim, 64, normalize=False),
            )
            self.model = nn.Sequential(
                *modelLayers.conv_block(128, 256),
                *modelLayers.conv_block(256, 512),
                nn.Conv2d(512, self.output_dim, kernel_size=4, stride=1, padding=0),
                nn.Sigmoid(),
            )
            initialize_weights(self)
        elif self.dataset_name == "CIFAR10":
            self.input_layer_data = nn.Sequential(
                *modelLayers.conv_block(self.input_dim, 64, 3, 1, 1, normalize=False),
            )
            self.input_layer_labels = nn.Sequential(
                *modelLayers.conv_block(self.label_dim, 64, 3, 1, 1, normalize=False),
            )
            self.model = nn.Sequential(
                *modelLayers.conv_block(128, 256),
                *modelLayers.conv_block(256, 512),
                *modelLayers.conv_block(512, 1024),
                nn.Conv2d(1024, self.output_dim, kernel_size=4, stride=1, padding=0),
                nn.Sigmoid(),
            )
            initialize_weights(self)

    def forward(self, data, labels):
        """
        Args:
            data: Input image data, of shape (batch_size, input_dim, image_size, image_size).
            labels: Input label vector, of shape (batch_size, label_dim, image_size, image_size).
        Return:
            out: Classification result vector, of shape (batch_size, 1).
        """

        if data.is_cuda and self.num_gpu > 1:
            h1 = nn.parallel.data_parallel(self.input_layer_data, data, range(self.ngpu))
            h2 = nn.parallel.data_parallel(self.input_layer_labels, labels, range(self.ngpu))
            x = torch.cat([h1, h2], 1)
            out = nn.parallel.data_parallel(self.model, x, range(self.ngpu))
        else:
            h1 = self.input_layer_data(data)
            h2 = self.input_layer_labels(labels)
            x = torch.cat([h1, h2], 1)
            out = self.model(x)
        return out
