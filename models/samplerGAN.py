"""
Generator and discriminator implementation for SamplerGAN

"""

import torch
import torch.nn as nn

from models import modelLayers
from models.tcn import TCN
from utils import initialize_weights


class ImageGenerator(nn.Module):
    def __init__(self, num_gpu, input_dim, label_dim, output_dim, image_size, dataset_name):
        """SamplerGAN generator implementation for the image domain

        Args:
            num_gpu (int): Number of gpu.
            input_dim (int): Dimension of the input noise vector to generate fake data.
            label_dim (int): Number of classes in the dataset.
            output_dim (int): Dimension of output (equal to channel_dim of images in the dataset).
            image_size (int): Size of input image.
            dataset_name (str): Name of the dataset: MNIST/CIFAR10.
        """

        super(ImageGenerator, self).__init__()
        self.num_gpu = num_gpu
        self.input_dim = input_dim * label_dim
        self.output_dim = output_dim
        self.image_size = image_size
        self.dataset_name = dataset_name

        if self.dataset_name == "MNIST":
            self.fc = nn.Sequential(
                *modelLayers.linear_block(self.input_dim, 1024),
                *modelLayers.linear_block(1024, 128 * (self.image_size // 4) * (self.image_size // 4))
            )
            self.model = nn.Sequential(
                *modelLayers.convT_block(128, 64),
                nn.ConvTranspose2d(64, self.output_dim, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
            initialize_weights(self)
        elif self.dataset_name == "CIFAR10":
            self.fc = nn.Sequential(
                *modelLayers.linear_block(self.input_dim, 1024 * (self.image_size // 8) * (self.image_size // 8)),
            )
            self.model = nn.Sequential(
                *modelLayers.convT_block(1024, 512),
                *modelLayers.convT_block(512, 256),
                *modelLayers.convT_block(256, 128),
                nn.ConvTranspose2d(128, self.output_dim, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            )
            initialize_weights(self)

    def forward(self, noise):
        """
        Args:
            noise: Input noise vector, of shape (batch_size, noise_dim * label_dim).
        Return:
            out: Generated image data, of shape (batch_size, output_dim, image_size, image_size).
        """

        if noise.is_cuda and self.num_gpu > 1:
            h = nn.parallel.data_parallel(self.fc, noise, range(self.ngpu))
            if self.dataset_name == "MNIST":
                h = h.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
            elif self.dataset_name == "CIFAR10":
                h = h.view(-1, 1024, (self.image_size // 8), (self.image_size // 8))
            out = nn.parallel.data_parallel(self.model, h, range(self.ngpu))
        else:
            h = self.fc(noise)
            if self.dataset_name == "MNIST":
                h = h.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
            elif self.dataset_name == "CIFAR10":
                h = h.view(-1, 1024, (self.image_size // 8), (self.image_size // 8))
            out = self.model(h)
        return out


class TSGeneratorLinear(nn.Module):
    def __init__(self, num_gpu, input_dim, output_dim, label_dim):
        """SamplerGAN generator implementation with linear layer for the time-series domain

        Args:
            num_gpu (int): Number of gpu.
            input_dim (int): Dimension of the input noise vector to generate fake data.
            output_dim (int): Dimension of output (equal to time_step).
            label_dim (int): Number of classes in the dataset.
        """

        super(TSGeneratorLinear, self).__init__()
        self.num_gpu = num_gpu
        self.input_dim = input_dim * label_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            *modelLayers.linear_block(self.input_dim, 500, normalize=False),
            *modelLayers.linear_block(500, 500, normalize=False),
            *modelLayers.linear_block(500, self.output_dim, normalize=False, activation="None"),
        )

    def forward(self, noise):
        """
        Args:
            noise: Input noise vector, of shape (batch_size, noise_dim * label_dim).
        Return:
            out: Generated time-series data, of shape (batch_size, time_step).
        """

        if noise.is_cuda and self.num_gpu > 1:
            out = nn.parallel.data_parallel(self.model, noise, range(self.ngpu))
        else:
            out = self.model(noise)
        return out


class TSGeneratorRNN(nn.Module):
    def __init__(self, input_dim, output_dim, label_dim, hidden_dim, rnn_type="gru"):
        """SamplerGAN generator implementation with RNN cell for the time-series domain

        Args:
            input_dim (int): Dimension of the input noise vector to generate fake data.
            output_dim (int): Dimension of output.
            label_dim (int): Number of classes in the dataset.
            hidden_dim (int):Dimension of RNN's hidden layer.
            rnn_type (str): Type of the RNN cell: gru/lstm.
        """

        super(TSGeneratorRNN, self).__init__()
        self.input_dim = input_dim * label_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type

        self.noise_to_hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.prepare_input = nn.Linear(1, self.hidden_dim)

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim)

        self.prepare_output = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, input, hidden):
        """
        Args:
            input: RNN's input vector, of shape (batch_size, hidden_dim).
            hidden: RNN's input hidden vector, of shape (1, batch_size, hidden_dim).
        Return:
            out: Generated time-series data for one time step, of shape (batch_size, 1).
            out_act: out with ReLU activation, of shape (batch_size, 1).
            hidden: RNN's output hidden vector, of shape (1, batch_size, hidden_dim).
        """

        input = self.prepare_input(input)
        input = input.view(-1, 1, self.hidden_dim)
        input = input.transpose(0, 1)
        output, hidden = self.rnn(input, hidden)
        out = self.prepare_output(output[0])
        out_act = self.relu(out)
        return out, out_act, hidden

    def init_hidden(self, noise):
        """Noise initialization process tp prepare it as RNN's input hidden vector

        Args:
            noise: Input noise vector, of shape (batch_size, noise_dim * time_step).
        Return:
            hidden: RNN's input hidden vector, of shape (1, batch_size, hidden_dim).
        """

        hidden = self.noise_to_hidden(noise)
        hidden = hidden.view(-1, 1, self.hidden_dim)
        hidden = hidden.transpose(0, 1)
        return hidden


class TSGeneratorTCN(nn.Module):
    def __init__(self, input_dim, output_dim, label_dim, time_step, n_layers, n_channel, kernel_size, dropout=0):
        """SamplerGAN generator implementation with TCN cell for the time-series domain

        Args:
            input_dim (int): Dimension of the input noise vector to generate fake data.
            output_dim (int): Dimension of output .
            label_dim (int): Number of classes in the dataset.
            time_step (int): Number of time steps in the dataset.
            n_layers (int): Number of TCN's hidden layers.
            n_channel (int): Number of channels in the hidden layers.
            kernel_size (int): Size of kernel in all the layers.
            dropout: (float in [0-1]): Rate of dropout.
        """

        super(TSGeneratorTCN, self).__init__()
        self.input_dim = input_dim * label_dim
        self.output_dim = output_dim
        self.time_step = time_step
        num_channels = [n_channel] * n_layers

        self.linear = nn.Linear(1, self.time_step)
        self.tcn = TCN(self.input_dim, self.output_dim, num_channels, kernel_size, dropout)

    def forward(self, noise):
        """
        Args:
            noise: Input noise vector, of shape (batch_size, noise_dim * label_dim).
        Return:
            out: Generated time-series data, of shape (batch_size, time_step, 1).
        """

        noise = noise.view(-1, self.input_dim, 1)
        out = self.linear(noise)
        out = self.tcn(out.transpose(1, 2))
        return out


class ImageDiscriminator(nn.Module):
    def __init__(self, num_gpu, input_dim, output_dim, label_dim, image_size, dataset_name):
        """SamplerGAN discriminator implementation for the image domain

        Args:
            num_gpu (int): Number of gpu.
            input_dim (int): Dimension of the input (equal to channel_dim of images in the dataset).
            output_dim (int): Dimension of output.
            label_dim (int): Number of classes in the dataset.
            image_size (int): Size of input image.
            dataset_name (str): Name of the dataset: MNIST/CIFAR10.
        """

        super(ImageDiscriminator, self).__init__()
        self.num_gpu = num_gpu
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.label_dim = label_dim
        self.image_size = image_size
        self.dataset_name = dataset_name

        if self.dataset_name == "MNIST":
            self.input_layer_data = nn.Sequential(
                *modelLayers.conv_block(self.input_dim, 32, normalize=False),
            )
            self.input_layer_labels = nn.Sequential(
                *modelLayers.conv_block(self.label_dim, 32, normalize=False),
            )
            self.model = nn.Sequential(
                *modelLayers.conv_block(64, 128),
            )
            self.fc = nn.Sequential(
                *modelLayers.linear_block(128 * (self.image_size // 4) * (self.image_size // 4), 1024,
                                          activation="LeakyReLU"),
                nn.Linear(1024, self.output_dim),
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
            )
            self.fc = nn.Sequential(
                nn.Linear(1024 * (self.image_size // 8) * (self.image_size // 8), self.output_dim),
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
            h = torch.cat([h1, h2], 1)
            x = nn.parallel.data_parallel(self.model, h, range(self.ngpu))
            if self.dataset_name == "MNIST":
                x = x.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))
            elif self.dataset_name == "CIFAR10":
                x = x.view(-1, 1024 * (self.image_size // 8) * (self.image_size // 8))
            out = nn.parallel.data_parallel(self.fc, x, range(self.ngpu))
        else:
            h1 = self.input_layer_data(data)
            h2 = self.input_layer_labels(labels)
            h = torch.cat([h1, h2], 1)
            x = self.model(h)
            if self.dataset_name == "MNIST":
                x = x.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))
            elif self.dataset_name == "CIFAR10":
                x = x.view(-1, 1024 * (self.image_size // 8) * (self.image_size // 8))
            out = self.fc(x)
        return out


class TSDiscriminatorLinear(nn.Module):
    def __init__(self, num_gpu, input_dim, output_dim, label_dim):
        """SamplerGAN discriminator implementation with linear layer for the time-series domain

        Args:
            num_gpu (int): Number of gpu.
            input_dim (int): Dimension of the input (equal to time_step).
            output_dim (int): Dimension of output.
            label_dim (int): Number of classes in the dataset.
        """

        super(TSDiscriminatorLinear, self).__init__()
        self.num_gpu = num_gpu
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.output_dim = output_dim

        self.input_layer_labels = nn.Sequential(
            *modelLayers.linear_block(self.label_dim, self.input_dim, normalize=False),
        )

        self.model = nn.Sequential(
            *modelLayers.linear_block(self.input_dim + self.input_dim, 500, normalize=False),
            *modelLayers.linear_block(500, 500, normalize=False),
            *modelLayers.linear_block(500, self.output_dim, normalize=False, activation="None"),
            nn.Sigmoid(),
        )

    def forward(self, data, labels):
        """
        Args:
            data: Input time-series data, of shape (batch_size, time_step).
            labels: Input label vector, of shape (batch_size, label_dim).
        Return:
            out: Classification result vector, of shape (batch_size, 1).
        """

        if data.is_cuda and self.num_gpu > 1:
            h = nn.parallel.data_parallel(self.input_layer_labels, labels, range(self.ngpu))
            x = torch.cat([data, h], 1)
            out = nn.parallel.data_parallel(self.model, x, range(self.ngpu))
        else:
            h = self.input_layer_labels(labels.float())
            x = torch.cat([data, h], 1)
            out = self.model(x)
        return out


class TSDiscriminatorRNN(nn.Module):
    def __init__(self, time_step, label_dim, hidden_dim, output_dim, rnn_type="gru"):
        """SamplerGAN discriminator implementation with RNN cell for the time-series domain

        Args:
            time_step (int): Number of time steps in the dataset.
            label_dim (int): Number of classes in the dataset.
            hidden_dim (int):Dimension of RNN's hidden layer.
            output_dim (int): Dimension of output.
            rnn_type (str): Type of the RNN cell: gru/lstm.
        """

        super(TSDiscriminatorRNN, self).__init__()
        self.time_step = time_step
        self.input_dim = 1 + label_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_type = rnn_type

        self.label_embeddings = nn.Embedding(self.label_dim, self.label_dim)

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim)

        self.model = nn.Sequential(
            *modelLayers.linear_block(self.hidden_dim, self.output_dim, normalize=False, activation="None"),
            nn.Sigmoid(),
        )

    def forward(self, data, labels):
        """
        Args:
            data: Input time-series data, of shape (batch_size, time_step, 1).
            labels: Input label vector, of shape (batch_size).
        Return:
            out: Classification result vector, of shape (batch_size, time_step, 1).
        """

        # Creates an extra dimension on the tensor and repeats it throughout
        labels_tiled = labels.view(-1, 1).repeat(1, self.time_step)
        labels_emb = self.label_embeddings(labels_tiled.type(torch.LongTensor).to(labels.device))

        x = torch.cat((data, labels_emb), dim=2)
        h, _ = self.rnn(x.transpose(0, 1))
        out = self.model(h.transpose(0, 1))
        return out


class TSDiscriminatorTCN(nn.Module):
    def __init__(self, time_step, label_dim, output_dim, n_layers, n_channel, kernel_size, dropout=0):
        """SamplerGAN discriminator implementation with TCN cell for the time-series domain

        Args:
            time_step (int): Number of time steps in the dataset.
            label_dim (int): Number of classes in the dataset.
            output_dim (int): Dimension of output.
            n_layers (int): Number of TCN's hidden layers.
            n_channel (int): Number of channels in the hidden layers.
            kernel_size (int): Size of kernel in all the layers.
            dropout: (float in [0-1]): Rate of dropout.
        """

        super(TSDiscriminatorTCN, self).__init__()
        self.time_step = time_step
        self.input_dim = 1 + label_dim
        self.label_dim = label_dim
        self.output_dim = output_dim
        num_channels = [n_channel] * n_layers

        self.label_embeddings = nn.Embedding(self.label_dim, self.label_dim)

        self.tcn = TCN(self.input_dim, self.output_dim, num_channels, kernel_size, dropout)

    def forward(self, data, labels):
        """
        Args:
            data: Input time-series data, of shape (batch_size, time_step, 1).
            labels: Input label vector, of shape (batch_size).
        Return:
            out: Classification result vector, of shape (batch_size, time_step, 1).
        """

        # Creates an extra dimension on the tensor and repeats it throughout
        labels_tiled = labels.view(-1, 1).repeat(1, self.time_step)
        labels_emb = self.label_embeddings(labels_tiled.type(torch.LongTensor).to(labels.device))

        x = torch.cat((data, labels_emb), dim=2)
        out = torch.sigmoid(self.tcn(x))
        return out
