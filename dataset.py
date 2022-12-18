"""
Loading datasets

"""

import os
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torchvision import datasets, transforms


def load_image_dataset(dataroot, dataset_name, dataset_mode, image_size):
    """Image datasets loading

    Args:
        dataroot (str): Root path to image datasets.
        dataset_name (str): Name of the dataset: MNIST/CIFAR10.
        dataset_mode (str): Mode of the dataset: train/test.
        image_size (int): Size of input image.
    """

    dataset = None
    label_dim = None
    if dataset_name == "MNIST":
        if dataset_mode == "train":
            dataset = datasets.MNIST(root=dataroot, download=True, train=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                     ]))
        elif dataset_mode == "test":
            dataset = datasets.MNIST(root=dataroot, download=True, train=False,
                                     transform=transforms.Compose([
                                         transforms.Resize(image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                     ]))
        label_dim = 10
    elif dataset_name == "CIFAR10":
        dataroot = os.path.join(dataroot, "cifar-10")
        # dataroot = dataroot + "/cifar-10"
        if dataset_mode == "train":
            dataset = datasets.CIFAR10(root=dataroot, download=True, train=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
        elif dataset_mode == "test":
            dataset = datasets.CIFAR10(root=dataroot, download=True, train=False,
                                       transform=transforms.Compose([
                                           transforms.Resize(image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))
        label_dim = 10
    return dataset, label_dim


def load_ts_dataset(dataroot, dataset_name, dataset_mode):
    """Time-series datasets loading

    Args:
        dataroot (str): Root path to time-series datasets.
        dataset_name (str): Name of the dataset.
        dataset_mode (str): Mode of the dataset: train/test.
    """

    dataroot = os.path.join(dataroot, dataset_name)
    dataset_path = Path(dataroot)
    if dataset_mode == "train":
        data = np.loadtxt(dataset_path / f'{dataset_name}_TRAIN.tsv', delimiter='\t')
    elif dataset_mode == "test":
        data = np.loadtxt(dataset_path / f'{dataset_name}_TEST.tsv', delimiter='\t')
    encoder = OneHotEncoder(categories='auto', sparse=False)
    labels = encoder.fit_transform(np.expand_dims(data[:, 0], axis=-1))
    dataset = (torch.from_numpy(data[:, 1:]).float(), torch.from_numpy(labels))
    return dataset
