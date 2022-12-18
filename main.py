import argparse

import torch

from samplerGAN import SamplerGAN
from rcGAN import RCGAN
from cGAN import CGAN
from infoGAN import InfoGAN
from acGAN import ACGAN
from evaluation import Evaluation
from utils import check_args
from utils import set_seed


# parse and configure arguments
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--domain", type=str, default="image", choices=["image", "time-series"],
                        help="Name of working domain")
    parser.add_argument("--evaluate_reals", action='store_true', help="Flag for evaluating real datasets")
    parser.add_argument("--dataroot", type=str, default="/ds2", help="Root directory for dataset")
    parser.add_argument("--gan_model", type=str, default="SamplerGAN",
                        choices=["SamplerGAN", "RCGAN", "CGAN", "InfoGAN", "ACGAN"], help="Name of GAN models")
    parser.add_argument("--architecture", type=str, default="TCN", choices=["Linear", "RNN", "TCN"],
                        help="Type of architecture of GAN model")
    parser.add_argument("--dataset", type=str, default="MNIST",
                        choices=["MNIST", "CIFAR10", "SmoothSubspace", "Strawberry", "Crop", "FiftyWords"],
                        help="Name of dataset")
    parser.add_argument("--result_dir", type=str, default="/netscratch/gholamrezaei/Thesis/",
                        help="Directory of results")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size during training")
    parser.add_argument("--noise_dim", type=int, default=100, help="Dimension of z vector")
    parser.add_argument("--code_dim", type=int, default=0, help="Latent code")
    parser.add_argument("--image_size", type=int, default=32, help="Dimension of images")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--num_gpu", type=int, default=1, help="Number of GPUs available. 0 for CPU mode")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizers")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 hyper-parameter for Adam optimizers")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyper-parameter for Adam optimizers")

    return check_args(parser.parse_args())


def main():
    # parse and configure arguments
    args = parse_args()
    if args is None:
        exit()

    # set random seed for reproducibility
    seed = set_seed()
    torch.manual_seed(seed)

    # identify process
    if args.evaluate_reals:
        print("Starting evaluating process...")
        evaluate = Evaluation(args)
        evaluate.evaluate_reals()
    else:
        # define model
        gan = None
        if args.gan_model == "SamplerGAN":
            print(" [*] Model : SamplerGAN")
            gan = SamplerGAN(args)
        elif args.gan_model == "RCGAN":
            if args.domain == "image":
                raise Exception(" [!] There is no option for RCGAN model on the image domain")
            if args.architecture == "Linear":
                raise Exception(" [!] There is no option for RCGAN model based on the linear architecture")
            print(" [*] Model : RCGAN")
            gan = RCGAN(args)
        elif args.gan_model == "CGAN":
            if args.domain == "time-series":
                raise Exception(" [!] There is no option for CGAN model on the time-series domain")
            print(" [*] Model : CGAN")
            gan = CGAN(args)
        elif args.gan_model == "InfoGAN":
            if args.domain == "time-series":
                raise Exception(" [!] There is no option for infoGAN model on the time-series domain")
            print(" [*] Model : InfoGAN")
            gan = InfoGAN(args)
        elif args.gan_model == "ACGAN":
            if args.domain == "time-series":
                raise Exception(" [!] There is no option for ACGAN model on the time-series domain")
            print(" [*] Model : ACGAN")
            gan = ACGAN(args)

        # build model
        gan.build_model()
        print(" [*] Model built!")

        # print model
        gan.print_model()

        # train model
        gan.train_model()
        print(" [*] Training finished!")

        # visualize model progress
        gan.generate_animations()
        print(" [*] Animations generated!")


if __name__ == '__main__':
    main()
