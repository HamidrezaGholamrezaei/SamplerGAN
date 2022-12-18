"""
SamplerGAN model implementation

"""

import pickle
import time
import itertools

import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

from dataset import *
from metrics import Metrics
from models.samplerGAN import *
from utils import *


class Sampler(object):
    def __init__(self, batch_size, noise_dim, label_dim):
        """
        Sampler module implementation

        Args:
            batch_size (int): Size of the batch.
            noise_dim (int): Dimension of the noise vector.
            label_dim (int): Number of classes in the dataset.
        """

        super(Sampler, self).__init__()
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.label_dim = label_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set distributions parameters
        self.onehot = torch.tensor(np.diag(np.ones(self.label_dim)), device=self.device)
        self.distributions = []

        # Set noise distributions
        for i in range(self.label_dim):
            self.distributions.append(
                MultivariateNormal(self.onehot[i], self.onehot).sample((self.batch_size, self.noise_dim))
            )

    def sample_noise(self, size_, labels):
        """
        Noise vector sampling based on given labels

        Args:
            size_: Size of the noise vector.
            labels: Input label vector, of shape (size_, label_dim).
        Return:
            z_: Sampled noise vector, of shape (size_, noise_dim * label_dim).
        """

        distribution_idx = labels.detach().numpy()
        z_ = torch.zeros(size_, self.noise_dim, self.label_dim)
        for i in range(size_):
            z_[i, :] = self.distributions[int(distribution_idx[i])][i, :]
        z_ = z_.view(-1, self.noise_dim * self.label_dim)
        return z_


class SamplerGAN(object):
    generator = None
    discriminator = None
    optimizerG: optim.Adam
    optimizerD: optim.Adam
    BCE_loss: nn.BCELoss
    fixed_sampler: Sampler
    random_sampler: Sampler

    def __init__(self, args):
        # set parameters
        self.domain = args.domain
        self.dataroot = args.dataroot
        self.gan_model = args.gan_model
        self.architecture = args.architecture
        self.dataset = args.dataset
        self.result_dir = args.result_dir
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.noise_dim = args.noise_dim
        self.image_size = args.image_size
        self.num_epochs = args.num_epochs
        self.auto_regressive = True if self.domain == "time-series" and self.architecture == "RNN" else False
        self.num_gpu = args.num_gpu
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        # config directories
        self.result_dir = os.path.join(self.result_dir, self.dataset)
        self.result_dir = os.path.join(self.result_dir, self.gan_model)
        if self.domain == "time-series":
            self.result_dir = os.path.join(self.result_dir, self.architecture)
        self.losses_dir = os.path.join(self.result_dir, "losses")
        self.evaluation_dir = os.path.join(self.result_dir, "evaluations")
        check_folder(self.result_dir)
        check_folder(self.losses_dir)
        check_folder(self.evaluation_dir)
        if self.domain == "image":
            self.images_dir = os.path.join(self.result_dir, "images")
            check_folder(self.images_dir)

        # check gpu
        self.cuda = True if torch.cuda.is_available() else False
        print(" [*] Cuda: ", self.cuda)

        # set training parameters
        self.train_hist = {}

        # set evaluation parameters
        self.eval_hist = {}
        self.eval = {}

        # load dataset
        if self.domain == "image":
            self.train_dataset, self.label_dim = load_image_dataset(dataroot=self.dataroot, dataset_name=self.dataset,
                                                                    dataset_mode="train", image_size=self.image_size)
            self.num_samples = self.label_dim ** 2
            self.data_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
            data = self.data_loader.__iter__().__next__()[0]
            self.channel_dim = data.shape[1]
        elif self.domain == "time-series":
            self.train_dataset = load_ts_dataset(dataroot=self.dataroot, dataset_name=self.dataset, dataset_mode="train")
            self.test_dataset = load_ts_dataset(dataroot=self.dataroot, dataset_name=self.dataset, dataset_mode="test")
            self.label_dim = self.train_dataset[1].shape[1]
            self.time_step = self.train_dataset[0].shape[1]
            self.data_loader = DataLoader(TensorDataset(self.train_dataset[0], self.train_dataset[1]),
                                          batch_size=self.batch_size, shuffle=True)

        # fixed noise & labels for testing in the image domain
        if self.domain == "image":
            self.fixed_sampler = Sampler(self.batch_size, self.noise_dim, self.label_dim)
            distribution_idx = torch.zeros(0).type(torch.LongTensor)
            fixed_label = torch.tensor(range(self.label_dim)).type(torch.LongTensor)
            for i in range(self.label_dim):
                l_ = torch.stack([fixed_label[i] for _ in range(self.label_dim)], axis=0)
                distribution_idx = torch.cat([distribution_idx, l_], 0)
            self.fixed_noise = self.fixed_sampler.sample_noise(self.num_samples, distribution_idx)
            if self.cuda:
                self.fixed_noise = self.fixed_noise.cuda()

    def build_model(self):
        """
        SamplerGAN generator and discriminator initialization

        """

        if self.domain == "image":
            self.generator = ImageGenerator(num_gpu=self.num_gpu, input_dim=self.noise_dim, label_dim=self.label_dim,
                                            output_dim=self.channel_dim, image_size=self.image_size,
                                            dataset_name=self.dataset)
            self.discriminator = ImageDiscriminator(num_gpu=self.num_gpu, input_dim=self.channel_dim, output_dim=1,
                                                    label_dim=self.label_dim, image_size=self.image_size,
                                                    dataset_name=self.dataset)
        elif self.domain == "time-series":
            if self.architecture == "Linear":
                self.generator = TSGeneratorLinear(num_gpu=self.num_gpu, input_dim=self.noise_dim,
                                                   output_dim=self.time_step, label_dim=self.label_dim)
                self.discriminator = TSDiscriminatorLinear(num_gpu=self.num_gpu, input_dim=self.time_step, output_dim=1,
                                                           label_dim=self.label_dim)
            elif self.architecture == "RNN":
                self.generator = TSGeneratorRNN(input_dim=self.noise_dim, output_dim=1, label_dim=self.label_dim,
                                                hidden_dim=self.time_step)
                self.discriminator = TSDiscriminatorRNN(time_step=self.time_step, label_dim=self.label_dim,
                                                        hidden_dim=self.time_step, output_dim=1)
            elif self.architecture == "TCN":
                self.generator = TSGeneratorTCN(input_dim=self.noise_dim, output_dim=1, label_dim=self.label_dim,
                                                time_step=self.time_step, n_layers=1, n_channel=10, kernel_size=8,
                                                dropout=0)
                self.discriminator = TSDiscriminatorTCN(time_step=self.time_step, label_dim=self.label_dim, output_dim=1,
                                                        n_layers=1, n_channel=10, kernel_size=8, dropout=0)
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

    def print_model(self):
        print('---------- Networks architecture -------------')
        print_network(self.generator)
        print_network(self.discriminator)
        print('-----------------------------------------------')

    def generate_samples_autoregressively(self, batch_size, noise):
        """
        Generating fake samples auto-regressively (used only for SamplerGAN-RNN model)

        Args:
            batch_size: Size of the batch.
            noise: Input noise vector, of shape (batch_size, noise_dim * label_dim).
        Return:
            x_fake_: Generated time-series data, of shape (batch_size, time_step, 1).
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

    def process_label(self, labels, onehot=True):
        """
        Processing labels vector before passing to the discriminator network

        Args:
            labels: Input labels vector.
            onehot: Determines whether the input labels vector is onehot.
        Return:
            y_: Processed labels vector.
        """

        y_ = None
        if self.domain == "image":
            fill = torch.zeros([self.label_dim, self.label_dim, self.image_size, self.image_size])
            for i in range(self.label_dim):
                fill[i, i, :, :] = 1
            y_ = fill[labels]
        else:
            encoder = OneHotEncoder(categories=[np.arange(i) for i in [self.label_dim]], sparse=False)
            if self.architecture == "Linear":
                y_ = labels if onehot else torch.from_numpy(encoder.fit_transform(np.expand_dims(labels, axis=-1)))
            elif self.architecture in ["RNN", "TCN"]:
                y_ = torch.max(labels, 1)[1] if onehot else labels
        return y_

    def train_model(self):
        """
        SamplerGAN model training

        """

        # training process
        self.train_hist['d_avg_losses'] = []
        self.train_hist['g_avg_losses'] = []
        self.train_hist['per_epoch_times'] = []
        self.train_hist['total_time'] = []

        # evaluation process
        best_is_score, best_fid_score = -1, np.inf
        best_is_epoch, best_fid_epoch = -1, -1
        best_state_gen_is, best_state_gen_fid = None, None
        self.eval_hist['per_epoch_is'] = []
        self.eval_hist['per_epoch_fid'] = []
        IS_hist = {'IS_hist': []}

        print("Starting Training Loop...")
        start_time = time.time()

        for epoch in range(self.num_epochs):
            d_losses = []
            g_losses = []

            # epoch starts
            epoch_start_time = time.time()

            for i, (data, labels) in enumerate(self.data_loader):
                mini_batch = data.size()[0]
                y_zeros_ = torch.zeros(mini_batch, self.time_step) \
                    if self.domain == "time-series" and self.architecture in ["RNN", "TCN"] else torch.zeros(mini_batch)
                y_ones_ = torch.ones(mini_batch, self.time_step) \
                    if self.domain == "time-series" and self.architecture in ["RNN", "TCN"] else torch.ones(mini_batch)
                if self.cuda:
                    y_zeros_, y_ones_ = y_zeros_.cuda(), y_ones_.cuda()

                # ---------------------
                # Train Discriminator
                # ---------------------

                # real data
                x_real_ = data.view(mini_batch, self.time_step, 1) \
                    if self.domain == "time-series" and self.architecture in ["RNN", "TCN"] else data
                y_real_ = self.process_label(labels)
                if self.cuda:
                    x_real_, y_real_ = x_real_.cuda(), y_real_.cuda()
                d_real_ = self.discriminator(x_real_, y_real_).squeeze()
                d_real_loss = self.BCE_loss(d_real_, y_ones_)

                # fake data
                self.random_sampler = Sampler(mini_batch, self.noise_dim, self.label_dim)
                y_fake_ = torch.tensor(np.random.randint(0, self.label_dim, mini_batch)).type(torch.LongTensor)
                z_ = self.random_sampler.sample_noise(mini_batch, y_fake_)
                y_fake_dis_ = self.process_label(y_fake_, onehot=False)
                if self.cuda:
                    z_, y_fake_dis_ = z_.cuda(), y_fake_dis_.cuda()

                if self.domain == "time-series" and self.architecture == "RNN":
                    x_fake_ = self.generate_samples_autoregressively(batch_size=mini_batch, noise=z_)
                else:
                    x_fake_ = self.generator(z_)
                d_fake_ = self.discriminator(x_fake_, y_fake_dis_).squeeze()
                d_fake_loss = self.BCE_loss(d_fake_, y_zeros_)

                # backpropagation
                d_loss = (d_real_loss + d_fake_loss) / 2
                self.discriminator.zero_grad()
                d_loss.backward()
                self.optimizerD.step()

                # ---------------------
                # Train Generator
                # ---------------------

                if self.domain == "time-series" and self.architecture == "RNN":
                    x_fake_ = self.generate_samples_autoregressively(batch_size=mini_batch, noise=z_)
                else:
                    x_fake_ = self.generator(z_)
                d_fake_ = self.discriminator(x_fake_, y_fake_dis_).squeeze()

                # backpropagation
                g_loss = self.BCE_loss(d_fake_, y_ones_)
                self.generator.zero_grad()
                g_loss.backward()
                self.optimizerG.step()

                # logging
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                print('Epoch [%d/%d], Step [%d/%d], d_loss: %.4f, g_loss: %.4f' %
                      (epoch + 1, self.num_epochs, i + 1, len(self.data_loader), d_loss.item(), g_loss.item()))

            d_avg_loss = torch.mean(torch.FloatTensor(d_losses))
            g_avg_loss = torch.mean(torch.FloatTensor(g_losses))
            self.train_hist['d_avg_losses'].append(d_avg_loss)
            self.train_hist['g_avg_losses'].append(g_avg_loss)

            # epoch ends
            if self.domain == "image":
                self.visualize_results(epoch)
            self.visualize_losses(epoch)
            epoch_end_time = time.time()
            per_epoch_time = epoch_end_time - epoch_start_time
            self.train_hist['per_epoch_times'].append(per_epoch_time)

            # epoch evaluation
            if self.domain == "image":
                argsM = {'dataroot': self.dataroot, 'dataset': self.dataset, 'image_size': self.image_size,
                         'num_workers': self.num_workers, 'gan_model': self.gan_model, 'generator': self.generator,
                         'noise_dim': self.noise_dim, 'code_dim': 0, 'evaluation_dir': self.evaluation_dir}
                metrics = Metrics(argsM=argsM, domain=self.domain, batch_size=50, sample_size=50000)
            else:
                argsM = {'dataroot': self.dataroot, 'dataset': self.dataset, 'gan_model': self.gan_model,
                         'generator': self.generator, 'noise_dim': self.noise_dim, 'evaluation_dir': self.evaluation_dir}
                metrics = Metrics(argsM=argsM, domain=self.domain, batch_size=10, sample_size=1000,
                                  auto_regressive=self.auto_regressive)
            per_epoch_is_mean, per_epoch_is_std, per_epoch_fid = metrics.calculate_scores()
            per_epoch_is = [per_epoch_is_mean, per_epoch_is_std]
            self.eval_hist['per_epoch_is'].append(per_epoch_is)
            self.eval_hist['per_epoch_fid'].append(per_epoch_fid)
            IS_hist['IS_hist'].append(per_epoch_is_mean)
            metrics.visualize_metrics(metric_name="IS", metric_scores=IS_hist['IS_hist'],
                                      num_epochs=self.num_epochs, epoch=epoch)
            metrics.visualize_metrics(metric_name="FID", metric_scores=self.eval_hist['per_epoch_fid'],
                                      num_epochs=self.num_epochs, epoch=epoch)
            print('Epoch [%d/%d] finished, is_score: %.2f \u00B1 %.2f, fid_score: %.4f' %
                  (epoch + 1, self.num_epochs, per_epoch_is_mean, per_epoch_is_std, per_epoch_fid))

            if per_epoch_is_mean > best_is_score:
                best_is_score = per_epoch_is_mean
                best_is_epoch = epoch
                best_state_gen_is = self.generator.state_dict()
                print('New best generator based on IS found!')
            if per_epoch_fid < best_fid_score:
                best_fid_score = per_epoch_fid
                best_fid_epoch = epoch
                best_state_gen_fid = self.generator.state_dict()
                print('New best generator based on FID found!')

            # BESTS
            print('Best IS score: %.5f, Epoch: ' % best_is_score, best_is_epoch)
            print('Best FID score: %.4f, Epoch: ' % best_fid_score, best_fid_epoch)

        # end training
        end_time = time.time()
        total_time = end_time - start_time
        self.train_hist['total_time'].append(total_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" %
              (np.mean(self.train_hist['per_epoch_times']), self.num_epochs, self.train_hist['total_time'][0]))

        # save models
        self.save(best_state_gen_is=best_state_gen_is, best_state_gen_fid=best_state_gen_fid)

    def visualize_results(self, epoch):
        """
        SamplerGAN generated images visualization

        """

        self.generator.eval()
        gen_images = self.generator(self.fixed_noise)
        self.generator.train()

        if self.dataset == "MNIST":
            gen_images = denorm_image(gen_images)
            size_figure_grid = 10
            fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
            for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)
            for k in range(10 * 10):
                i = k // 10
                j = k % 10
                ax[i, j].cla()
                ax[i, j].imshow(gen_images[k, 0].cpu().data, cmap='gray')
            plt_label = 'Epoch {0}'.format(epoch + 1)
            fig.text(0.5, 0.04, plt_label, ha='center')
            plt_name = epoch + 1
            plt.savefig(self.images_dir + "/%d.png" % plt_name)
            plt.close()
        elif self.dataset == "CIFAR10":
            file_name = epoch + 1
            destination_dir = self.images_dir + "/%d.png" % file_name
            torchvision.utils.save_image(gen_images, destination_dir, nrow=10, normalize=True)

    def visualize_losses(self, epoch):
        """
        SamplerGAN training losses visualization

        """

        d_losses = self.train_hist['d_avg_losses']
        g_losses = self.train_hist['g_avg_losses']
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.num_epochs)
        ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses)) * 1.1)
        plt.xlabel('Epoch {0}'.format(epoch + 1))
        plt.ylabel('Loss values')
        plt.plot(d_losses, label='Discriminator')
        plt.plot(g_losses, label='Generator')
        plt.legend()
        plt_name = epoch + 1
        plt.savefig(self.losses_dir + "/%d.png" % plt_name)
        plt.close()

    def generate_animations(self):
        """
        Animations generating for SamplerGAN

        """

        if self.domain == "image":
            generate_animation(src_dir=self.images_dir, dest_dir=self.result_dir, gif_name="images",
                               num_epochs=self.num_epochs)
        generate_animation(src_dir=self.losses_dir, dest_dir=self.result_dir, gif_name="losses",
                           num_epochs=self.num_epochs)
        generate_animation(src_dir=self.evaluation_dir + "/metrics/fid", dest_dir=self.evaluation_dir, gif_name="fid",
                           num_epochs=self.num_epochs)
        generate_animation(src_dir=self.evaluation_dir + "/metrics/is", dest_dir=self.evaluation_dir, gif_name="is",
                           num_epochs=self.num_epochs)

    def save(self, best_state_gen_is, best_state_gen_fid):
        """
        Saving best SamplerGAN models based on IS and FID metrics

        """

        torch.save(best_state_gen_is, self.result_dir + "/best_generator_is.pth")
        torch.save(best_state_gen_fid, self.result_dir + "/best_generator_fid.pth")
        with open(self.result_dir + "/training_history.pkl", 'wb') as f:
            pickle.dump(self.train_hist, f)
        with open(self.result_dir + "/evaluation_history.pkl", 'wb') as f:
            pickle.dump(self.eval_hist, f)

    def load(self, metric_name):
        """
        Loading best SamplerGAN models based on IS and FID metrics

        """

        self.build_model()
        if metric_name == "IS":
            self.generator.load_state_dict(torch.load(self.result_dir + "/best_generator_is.pth",
                                                      map_location=torch.device('cpu')))
        elif metric_name == "FID":
            self.generator.load_state_dict(torch.load(self.result_dir + "/best_generator_fid.pth",
                                                      map_location=torch.device('cpu')))
        return self.generator
