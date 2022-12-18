"""
Different layer implementation

"""

import torch.nn as nn


def linear_block(in_feat, out_feat, normalize=True, activation="ReLU", lrelu=0.2):
    """Linear block implementation
    Args:
        in_feat (int): Number of input features.
        out_feat (int): Number of output features.
        normalize (bool): Batch normalization of the block: True/False.
        activation (str): Non-linearity of the block: ReLU/LeakyReLU/None.
        lrelu (float): Rate of LeakyReLU activation.
    Return:
        layer: Implemented linear block.
    """

    layer = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layer.append(nn.BatchNorm1d(out_feat))
    if activation == "ReLU":
        layer.append(nn.ReLU())
    elif activation == "LeakyReLU":
        layer.append(nn.LeakyReLU(lrelu))
    return layer


def conv_block(in_feat, out_feat, kernel_size=4, stride_size=2, padding_size=1, normalize=True, lrelu=0.2, dropout=0):
    """Convolution block implementation
    Args:
        in_feat (int): Number of input features.
        out_feat (int): Number of output features.
        kernel_size (int): Size of kernel in the convolution block.
        stride_size (int): Size of stride in the convolution block.
        padding_size (int): Size of padding in the convolution block.
        normalize (bool): Batch normalization of the block: True/False
        lrelu (float): Rate of LeakyReLU activation.
        dropout: (float in [0-1]): Rate of dropout.
    Return:
        layer: Implemented convolution block.
    """

    layer = [nn.Conv2d(
        in_feat, out_feat, kernel_size=kernel_size, stride=stride_size, padding=padding_size, bias=False
    )]
    if normalize:
        layer.append(nn.BatchNorm2d(out_feat))
    layer.append(nn.LeakyReLU(lrelu, inplace=True))
    if not dropout == 0:
        layer.append(nn.Dropout(dropout, inplace=False))
    return layer


def convT_block(in_feat, out_feat, kernel_size=4, stride_size=2, padding_size=1, normalize=True, dropout=0):
    """Transposed convolution block implementation
    Args:
        in_feat (int): Number of input features.
        out_feat (int): Number of output features.
        kernel_size (int): Size of kernel in the convolution block.
        stride_size (int): Size of stride in the convolution block.
        padding_size (int): Size of padding in the convolution block.
        normalize (bool): Batch normalization of the block: True/False
        dropout: (float in [0-1]): Rate of dropout.
    Return:
        layer: Implemented transposed convolution block.
    """

    layer = [nn.ConvTranspose2d(
        in_feat, out_feat, kernel_size=kernel_size, stride=stride_size, padding=padding_size, bias=False
    )]
    if normalize:
        layer.append(nn.BatchNorm2d(out_feat))
    layer.append(nn.ReLU())
    if not dropout == 0:
        layer.append(nn.Dropout(dropout))
    return layer
