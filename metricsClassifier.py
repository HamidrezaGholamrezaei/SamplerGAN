"""
Classifiers implementation for evaluation process

"""

import torch
import torch.nn as nn
from models.inceptionTime import InceptionBlock


class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, x):
        return x.view(-1, self.output_dim)


class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)


class Classifier(nn.Module):
    def __init__(self, dataset_name, label_dim, time_step=None):
        """
        Classifiers implementation for evaluation process

        Args:
            dataset_name (str): Name of the dataset.
            label_dim (int): Number of classes in the dataset.
            time_step (int): Number of time steps in the dataset.
        """

        super(Classifier, self).__init__()
        self.dataset_name = dataset_name
        self.label_dim = label_dim
        self.time_step = time_step

        if self.dataset_name == "MNIST":
            self.ClassifierModel = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(output_size=1),
                Flatten(out_features=128),
                nn.Linear(128, self.label_dim),
            )

        else:
            self.ClassifierModel = nn.Sequential(
                Reshape(out_shape=(1, self.time_step)),
                InceptionBlock(
                    in_channels=1,
                    n_filters=32,
                    kernel_sizes=[5, 11, 23],
                    bottleneck_channels=32,
                    use_residual=True,
                    activation=nn.ReLU()
                ),
                InceptionBlock(
                    in_channels=32 * 4,
                    n_filters=32,
                    kernel_sizes=[5, 11, 23],
                    bottleneck_channels=32,
                    use_residual=True,
                    activation=nn.ReLU()
                ),
                InceptionBlock(
                    in_channels=32 * 4,
                    n_filters=32,
                    kernel_sizes=[5, 11, 23],
                    bottleneck_channels=32,
                    use_residual=True,
                    activation=nn.ReLU()
                ),
                InceptionBlock(
                    in_channels=32 * 4,
                    n_filters=32,
                    kernel_sizes=[5, 11, 23],
                    bottleneck_channels=32,
                    use_residual=True,
                    activation=nn.ReLU()
                ),
                InceptionBlock(
                    in_channels=32 * 4,
                    n_filters=32,
                    kernel_sizes=[5, 11, 23],
                    bottleneck_channels=32,
                    use_residual=True,
                    activation=nn.ReLU()
                ),
                nn.AdaptiveAvgPool1d(output_size=1),
                Flatten(out_features=32 * 4 * 1),
                nn.Linear(in_features=4 * 32 * 1, out_features=self.label_dim)
            )

        if torch.cuda.is_available():
            self.ClassifierModel.cuda()

    def load(self, metric, domain):
        """
        Loading best trained classifier based on IS and FID metrics

        Args:
            metric: Name of the evaluation metric: IS/FID.
            domain: Name of the domain: image/time-series.
        """

        model_path = "./classifiers/best_classifier_{}.torch".format(self.dataset_name)
        state = 'best_state_' if domain == "image" else 'best_inception_state_'
        if metric == "IS":
            self.ClassifierModel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))
                                                 [state + 'is'], strict=False)
        elif metric == "FID":
            self.ClassifierModel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))
                                                 [state + 'fid'], strict=False)
        return self.ClassifierModel
