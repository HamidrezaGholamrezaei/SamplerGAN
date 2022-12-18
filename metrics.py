"""
Evaluation metrics of IS and FID implementation

"""

import matplotlib.pyplot as plt
import torch.nn as nn
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.inception import inception_v3

from dataset import *
from models.inception import InceptionV3
from metricsClassifier import Classifier
from utils import check_folder


class Metrics(object):
    def __init__(self, argsM, domain, batch_size=50, sample_size=50000, dimensions=2048, auto_regressive=False,
                 REALS=False):
        """Evaluation metrics implementation

        Args:
            argsM (dict): required parameters including dataroot path, dataset name, image size, model name,
                generator model, dimension of noise vector.
            domain (str): Name of the domain: image/time-series.
            batch_size (int): Size of the batch.
            sample_size (int): Number of the samples for evaluating process.
            dimensions (int): Number of features acquired from the pretrained InceptionV3 model.
            auto_regressive (bool): generating style for fake samples.
            REALS (bool): Only evaluating the real datasets.
        """

        # set metric parameters
        self.is_mean = 0
        self.is_std = 0
        self.fid_score = 0

        # set input parameters
        self.dataroot = argsM['dataroot']
        self.dataset = argsM['dataset']
        self.domain = domain
        if self.domain == "image":
            self.image_size = argsM['image_size']
            self.num_workers = argsM['num_workers']
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.dimensions = dimensions
        self.auto_regressive = auto_regressive
        self.REALS = REALS
        self.cuda = True if torch.cuda.is_available() else False
        if not self.REALS:
            self.gan_model = argsM['gan_model']
            self.generator = argsM['generator']
            self.noise_dim = argsM['noise_dim']
            self.evaluation_dir = argsM['evaluation_dir']
            if domain == "image":
                self.code_dim = argsM['code_dim']
            if self.cuda:
                self.generator.cuda()
        self.label_dim = None
        self.time_step = None

        # config directories
        if not self.REALS:
            self.metrics_dir = os.path.join(self.evaluation_dir, "metrics")
            self.fid_dir = os.path.join(self.metrics_dir, "fid")
            self.is_dir = os.path.join(self.metrics_dir, "is")
            check_folder(self.fid_dir)
            check_folder(self.is_dir)

        # load real data
        print("Loading real dataset...")
        if self.domain == "image":
            real_dataset, self.label_dim = load_image_dataset(dataroot=self.dataroot, dataset_name=self.dataset,
                                                              dataset_mode="train", image_size=self.image_size)
            self.reals_loader = DataLoader(dataset=real_dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)
        elif self.domain == "time-series":
            real_dataset = load_ts_dataset(dataroot=self.dataroot, dataset_name=self.dataset, dataset_mode="test")
            self.label_dim = real_dataset[1].shape[1]
            self.time_step = real_dataset[0].shape[1]
            self.reals_loader = DataLoader(TensorDataset(real_dataset[0], real_dataset[1]), batch_size=self.batch_size,
                                           shuffle=True)

        # build classifier
        self.classifier = None
        if self.dataset is not "CIFAR10":
            self.classifierModel = Classifier(dataset_name=self.dataset, label_dim=self.label_dim,
                                              time_step=self.time_step)

    def calculate_scores(self):
        """Calculating evaluation metrics

        Return:
            is_mean: Mean value of IS score.
            is_std: std value of IS score.
            fid_score: Value of FID score.
        """

        # calculate IS score
        print("Evaluating based on IS...")
        self.is_mean, self.is_std = self.calculate_inception_score()

        # calculate FID score
        print("Evaluating based on FID...")
        self.fid_score = self.calculate_fretchet()

        return self.is_mean, self.is_std, self.fid_score

    def load_fake_data(self):
        """Loading fake datasets to evaluate

        Return:
            fakes_loader: Dataloader of fake dataset.
            remove: Determines if dataloader contains data with or without labels.
        """

        print("Loading fake dataset...")
        fakes_loader, remove = None, None
        if self.REALS:
            if self.domain == "image":
                fakes_dataset, _ = load_image_dataset(dataroot=self.dataroot, dataset_name=self.dataset,
                                                      dataset_mode="test", image_size=self.image_size)
                fakes_loader = DataLoader(dataset=fakes_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers)
            elif self.domain == "time-series":
                fakes_dataset = load_ts_dataset(self.dataroot, self.dataset, dataset_mode="train")
                fakes_loader = DataLoader(TensorDataset(fakes_dataset[0], fakes_dataset[1]), batch_size=self.batch_size,
                                          shuffle=True)
            remove = True

        else:
            fakes_dataset = []
            with torch.no_grad():
                for i in range(self.sample_size // self.batch_size):
                    fake_data = self.generate_fake_samples()
                    fakes_dataset.append(fake_data.data.cpu())
                fakes_dataset = torch.cat(fakes_dataset, 0)
            if self.domain == "image":
                fakes_loader = DataLoader(dataset=fakes_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers)
                remove = False
            elif self.domain == "time-series":
                fakes_loader = DataLoader(TensorDataset(fakes_dataset), batch_size=self.batch_size, shuffle=True)
                remove = True
        return fakes_loader, remove

    def generate_fake_samples(self):
        """
        Generating fake samples

        """

        fake_data = None
        self.generator.eval()
        if self.gan_model == "SamplerGAN":
            from samplerGAN import Sampler
            sampler = Sampler(self.batch_size, self.noise_dim, self.label_dim)
            y_fake_ = torch.tensor(np.random.randint(0, self.label_dim, self.batch_size)).type(torch.LongTensor)
            z_ = sampler.sample_noise(self.batch_size, y_fake_)
            if self.cuda:
                z_ = z_.cuda()
            if self.auto_regressive:
                fake_data = self.generate_samples_autoregressively(batch_size=self.batch_size, noise=z_)
            else:
                fake_data = self.generator(z_)
        elif self.gan_model == "RCGAN":
            z_ = torch.randn(self.batch_size, self.noise_dim * self.time_step)
            y_fake_ = torch.tensor(np.random.randint(0, self.label_dim, self.batch_size)).type(torch.LongTensor)
            if self.cuda:
                z_, y_fake_ = z_.cuda(), y_fake_.cuda()
            fake_data = self.generator(z_, y_fake_)
        elif self.gan_model == "CGAN":
            z_ = torch.randn(self.batch_size, self.noise_dim).view(-1, self.noise_dim, 1, 1)
            y_fake_ = torch.tensor(np.random.randint(0, self.label_dim, self.batch_size)).type(torch.LongTensor)
            y_fake_gen_ = torch.zeros((self.batch_size, self.label_dim)).\
                scatter_(1, y_fake_.type(torch.LongTensor).unsqueeze(1), 1)
            y_fake_gen_ = y_fake_gen_.view(-1, self.label_dim, 1, 1)
            if self.cuda:
                z_, y_fake_gen_ = z_.cuda(), y_fake_gen_.cuda()
            fake_data = self.generator(z_, y_fake_gen_)
        elif self.gan_model == "InfoGAN":
            z_ = torch.randn(self.batch_size, self.noise_dim)
            y_disc_ = torch.tensor(np.random.multinomial(1, self.label_dim * [float(1.0 / self.label_dim)],
                                                         size=[self.batch_size])).type(torch.FloatTensor)
            y_cont_ = torch.zeros((self.batch_size, self.code_dim))
            if self.cuda:
                z_, y_disc_, y_cont_ = z_.cuda(), y_disc_.cuda(), y_cont_.cuda()
            fake_data = self.generator(z_, y_disc_, y_cont_)
        elif self.gan_model == "ACGAN":
            z_ = torch.randn(self.batch_size, self.noise_dim)
            y_fake_ = torch.tensor(np.random.randint(0, self.label_dim, self.batch_size)).type(torch.LongTensor)
            y_fake_gen_ = torch.zeros((self.batch_size, self.label_dim)).\
                scatter_(1, y_fake_.type(torch.LongTensor).unsqueeze(1), 1)
            z_ = torch.cat([y_fake_gen_, z_], 1)
            if self.cuda:
                z_ = z_.cuda()
            fake_data = self.generator(z_)
        self.generator.train()
        return fake_data

    def generate_samples_autoregressively(self, batch_size, noise):
        """
        Generating fake samples auto-regressively (in SamplerGAN-RNN model)

        """

        # Generate first fake time step (1/ts)
        gen_input = torch.zeros(batch_size, 1)
        if self.cuda:
            gen_input = gen_input.cuda()
        gen_hidden = self.generator.init_hidden(noise)
        x_t, gen_out, gen_hidden = self.generator(gen_input, gen_hidden)
        gen_input = gen_out
        x_fake_ = x_t
        # Generate remaining fake time step
        for ts in range(self.time_step - 1):
            x_t, gen_out, gen_hidden = self.generator(gen_input, gen_hidden)
            x_fake_ = torch.cat([x_fake_, x_t], 1)
            gen_input = gen_out
        x_fake_ = x_fake_.view(batch_size, self.time_step, 1)
        return x_fake_

    def modify_classifier(self):
        """
        Modifying the classifier: removing the last linear layer of the classifier

        """

        self.classifier = torch.nn.Sequential(*list(self.classifier.children())[:-1])

    def get_activations(self, metric_name, dataloader, remove=True):

        """Acquiring data representation from the classifier

        Args:
             metric_name: Name of the evaluation metric: IS/FID.
             dataloader: Dataloader of samples to evaluate.
             remove: Determines if dataloader contains data with or without labels.
        """

        self.classifier.eval()

        predictions = []

        for data in dataloader:
            # Removing labels from real dataset
            batch = data[0] if remove else data
            if self.cuda:
                batch = batch.cuda()

            # Up-sampling
            if metric_name == "IS" and self.dataset == "CIFAR10":
                up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
                if self.cuda:
                    up = up.cuda()
                batch = up(batch)

            batch_size = batch.shape[0]

            predict = self.classifier(batch)
            if metric_name == "IS":
                predict = nn.functional.softmax(predict, dim=1)
            elif metric_name == "FID" and self.dataset == "CIFAR10":
                predict = predict[0]

                # If model output is not scalar, apply global spatial average pooling.
                # This happens if you choose a dimensionality not equal 2048.
                if predict.shape[2] != 1 or predict.shape[3] != 1:
                    predict = adaptive_avg_pool2d(predict, output_size=(1, 1))

            predictions.append(predict.cpu().detach().numpy().reshape(batch_size, -1))

        return np.concatenate(predictions)

    # ---------------------
    # Inception Score
    # ---------------------
    def calculate_inception_score(self):
        """
        Calculating IS

        """

        if self.dataset == "CIFAR10":
            self.classifier = inception_v3(pretrained=True, transform_input=False)
        else:
            self.classifier = self.classifierModel.load(metric="IS", domain=self.domain)

        if self.cuda:
            self.classifier.cuda()

        if self.REALS:
            data_loader = self.reals_loader
            remove = True
        else:
            data_loader, remove = self.load_fake_data()
        predictions = self.get_activations(metric_name="IS", dataloader=data_loader, remove=remove)

        # Compute the mean kl-divergence
        eps = 1E-16
        num_splits = 10
        num_samples = predictions.shape[0]
        num_samples_per_splits = num_samples // num_splits
        scores = []
        for i in range(num_splits):
            ix_start, ix_end = i * num_samples_per_splits, (i+1) * num_samples_per_splits
            subset = predictions[ix_start:ix_end]
            subset = subset.astype('float32')
            mean_subset = np.expand_dims(np.mean(subset, axis=0), axis=0)
            kl = subset * (np.log(subset + eps) - np.log(mean_subset + eps))
            sum_kl = np.sum(kl, axis=1)
            mean_kl = np.mean(sum_kl)
            is_score = np.exp(mean_kl)
            scores.append(is_score)
        is_mean, is_std = np.mean(scores), np.std(scores)

        return is_mean, is_std
        
    # ---------------------
    # Fretchet Inception Distance
    # ---------------------
    def calculate_activation_statistics(self, dataloader, remove=True):
        """
        Calculating mean and std values of the data representation distribution

        """

        act = self.get_activations(metric_name="FID", dataloader=dataloader, remove=remove)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Numpy implementation of the Frechet Distance

        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

        diff = mu1 - mu2
        cov_mean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(cov_mean).all():
            msg = 'FID calculation produces singular product; adding %s to diagonal of cov estimates' % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            cov_mean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(cov_mean):
            if not np.allclose(np.diagonal(cov_mean).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_mean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            cov_mean = cov_mean.real

        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_mean)
        return fid

    def calculate_fretchet(self):
        """
        Calculating FID

        """

        if self.dataset == "CIFAR10":
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dimensions]
            self.classifier = InceptionV3([block_idx])
        else:
            self.classifier = self.classifierModel.load(metric="FID", domain=self.domain)
            self.modify_classifier()

        if self.cuda:
            self.classifier.cuda()

        mu1, std1 = self.calculate_activation_statistics(dataloader=self.reals_loader, remove=True)
        fakes_loader, remove = self.load_fake_data()
        mu2, std2 = self.calculate_activation_statistics(dataloader=fakes_loader, remove=remove)
        fid_value = self.calculate_frechet_distance(mu1=mu1, sigma1=std1, mu2=mu2, sigma2=std2)

        return fid_value

    # ---------------------
    # Metrics Visualization
    # ---------------------
    def visualize_metrics(self, metric_name, metric_scores, num_epochs, epoch):
        """
        Metrics of IS and FID visualization

        """

        fig, ax = plt.subplots()
        ax.set_xlim(0, num_epochs)
        ax.set_ylim(0, np.max(metric_scores)*1.1)
        plt.xlabel('Epoch {0}'.format(epoch + 1))
        plt.ylabel('Metrics values')
        labels = [i for i in range(0, num_epochs, int(num_epochs/5))]
        ax.set_xticks([i*int(num_epochs/5)-1 for i in range(0, len(labels))])
        ax.set_xticklabels(labels)
        plt.plot(metric_scores, label=metric_name)
        plt.legend()
        plt.grid(True)
        plt.xlim(xmin=1)
        plt_name = epoch + 1
        if metric_name == "FID":
            plt.savefig(self.fid_dir + "/%d.png" % plt_name)
        if metric_name == "IS":
            plt.savefig(self.is_dir + "/%d.png" % plt_name)
        plt.close()
