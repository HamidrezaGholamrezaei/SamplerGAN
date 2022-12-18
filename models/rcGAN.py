"""
Generator and discriminator implementation for RCGAN

"""

import torch
import torch.nn as nn

from models import modelLayers
from models.tcn import TCN


class TSGeneratorRNN(nn.Module):
    def __init__(self, input_dim, output_dim, label_dim, time_step, hidden_dim, rnn_type="gru"):
        """RCGAN generator implementation with RNN cell for the time-series domain

        Notes:
            This architecture is inspired from https://github.com/3778/Ward2ICU

        Args:
            input_dim (int): Dimension of the input noise vector to generate fake data.
            output_dim (int): Dimension of output.
            label_dim (int): Number of classes in the dataset.
            time_step (int): Number of time steps in the dataset.
            hidden_dim (int):Dimension of RNN's hidden layer.
            rnn_type (str): Type of the RNN cell: gru/lstm.
        """

        super(TSGeneratorRNN, self).__init__()
        self.input_dim = input_dim + label_dim
        self.noise_size = input_dim
        self.output_dim = output_dim
        self.label_dim = label_dim
        self.time_step = time_step
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type

        self.label_embeddings = nn.Embedding(self.label_dim, self.label_dim)

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim)

        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, noise, labels):
        """
        Args:
            noise: Input noise vector, of shape (batch_size, noise_dim * time_step).
            labels: Input label vector, of shape (batch_size).
        Return:
            out: Generated time-series data, of shape (batch_size, time_step, 1).
        """

        noise = noise.view(-1, self.time_step, self.noise_size)

        # Creates an extra dimension on the labels tensor and repeats it throughout
        labels_tiled = labels.view(-1, 1).repeat(1, self.time_step)
        labels_emb = self.label_embeddings(labels_tiled.type(torch.LongTensor).to(labels.device))

        noise_cond = torch.cat((noise, labels_emb), dim=2)
        noise_cond = noise_cond.transpose(0, 1)

        h, _ = self.rnn(noise_cond)
        out = self.linear(h.transpose(0, 1))
        return out


class TSGeneratorTCN(nn.Module):
    def __init__(self, input_dim, output_dim, label_dim, time_step, n_layers, n_channel, kernel_size, dropout=0):
        """RCGAN generator implementation with TCN cell for the time-series domain

        Notes:
            This architecture is inspired from https://github.com/proceduralia/pytorch-GAN-timeseries

        Args:
            input_dim (int): Dimension of the input noise vector to generate fake data.
            output_dim (int): Dimension of output.
            label_dim (int): Number of classes in the dataset.
            time_step (int): Number of time steps in the dataset.
            n_layers (int): Number of TCN's hidden layers.
            n_channel (int): Number of channels in the hidden layers.
            kernel_size (int): Size of kernel in all the layers.
            dropout: (float in [0-1]): Rate of dropout.
        """

        super(TSGeneratorTCN, self).__init__()
        self.input_dim = input_dim + label_dim
        self.noise_size = input_dim
        self.output_dim = output_dim
        self.label_dim = label_dim
        self.time_step = time_step
        num_channels = [n_channel] * n_layers

        self.label_embeddings = nn.Embedding(self.label_dim, self.label_dim)

        self.tcn = TCN(self.input_dim, self.output_dim, num_channels, kernel_size, dropout)

    def forward(self, noise, labels):
        """
        Args:
            noise: Input noise vector, of shape (batch_size, noise_dim * time_step).
            labels: Input label vector, of shape (batch_size).
        Return:
            out: Generated time-series data, of shape (batch_size, time_step, 1).
        """

        noise = noise.view(-1, self.time_step, self.noise_size)

        # Creates an extra dimension on the labels tensor and repeats it throughout
        labels_tiled = labels.view(-1, 1).repeat(1, self.time_step)
        labels_emb = self.label_embeddings(labels_tiled.type(torch.LongTensor).to(labels.device))

        noise_cond = torch.cat((noise, labels_emb), dim=2)
        out = self.tcn(noise_cond)
        return out


class TSDiscriminatorRNN(nn.Module):
    def __init__(self, time_step, label_dim, hidden_dim, output_dim, rnn_type="gru"):
        """RCGAN discriminator implementation with RNN cell for the time-series domain

        Notes:
            This architecture is inspired from https://github.com/3778/Ward2ICU

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
        """RCGAN discriminator implementation with TCN cell for the time-series domain

        Notes:
            This architecture is inspired from https://github.com/proceduralia/pytorch-GAN-timeseries

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
