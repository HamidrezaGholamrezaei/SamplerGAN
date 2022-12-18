import os
import random
import shutil

import imageio
import torch.nn as nn


def check_args(args):
    """
    Checking arguments

    """

    models = ["SamplerGAN", "RCGAN", "CGAN", "InfoGAN", "ACGAN"]
    if args.gan_model not in models:
        raise Exception(" [!] There is no option for " + args.gan_model + " model")

    datasets = ["MNIST", "CIFAR10", "SmoothSubspace", "Strawberry", "Crop", "FiftyWords"]
    if args.dataset not in datasets:
        raise Exception(" [!] There is no option for " + args.dataset + " dataset")

    assert args.batch_size >= 1
    assert args.num_epochs >= 1
    assert args.noise_dim >= 1

    check_folder(args.dataroot)
    check_folder(args.result_dir)

    print(" [*] Arguments: ", args)
    return args


def check_folder(dir_name, operation="add"):
    """
    Directories-related operations

    Args:
        dir_name (str): Name of the directory.
        operation (str): Type of the operation: add/delete.
    Return:
        dir_name (str): Name of the operated directory.
    """

    if operation == "add":
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    elif operation == "delete":
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    return dir_name


def set_seed():
    """
    Setting random seed

    """

    manual_seed = 104
    print(" [*] Random Seed: ", manual_seed)
    random.seed(manual_seed)
    return manual_seed


def print_network(net):
    """
    Printing the neural network

    """

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: ", num_params)


def initialize_weights(model):
    """
    Neural network's weights initialization

    """

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.ConvTranspose2d):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.normal_(mean=1.0, std=0.02)
            module.bias.data.zero_()


def denorm_image(x):
    """
    Image de-normalization

    """

    out = (x + 1) / 2
    return out.clamp(0, 1)


def generate_animation(src_dir, dest_dir, gif_name, num_epochs):
    """
    Animations generating for models

    Args:
        src_dir: Source directory path for reading inputs data.
        dest_dir: Destination directory path for saving animation results.
        gif_name: Name of the animation.
        num_epochs: Number of input data to read.
    """

    plots_list = []
    for epoch in range(num_epochs):
        if (epoch+1) % 5 == 0:
            plt_name = epoch + 1
            plots_list.append(imageio.imread(src_dir + "/%d.png" % plt_name))
    imageio.mimsave(dest_dir + "/" + gif_name + ".gif", plots_list, fps=5, loop=1)
