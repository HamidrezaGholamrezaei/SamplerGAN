"""
RCGAN model implementation

"""

import pickle
import time

import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dataset import *
from metrics import Metrics
from models.rcGAN import *
from utils import *


class RCGAN(object):
    generator = None
    discriminator = None
    optimizerG: optim.Adam
    optimizerD: optim.Adam
    BCE_loss: nn.BCELoss

    def __init__(self, args):
        # set parameters
        self.dataroot = args.dataroot
        self.gan_model = args.gan_model
        self.architecture = args.architecture
        self.dataset = args.dataset
        self.result_dir = args.result_dir
        self.batch_size = args.batch_size
        self.noise_dim = args.noise_dim
        self.num_epochs = args.num_epochs
        self.num_gpu = args.num_gpu
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        # config directories
        self.result_dir = os.path.join(self.result_dir, self.dataset)
        self.result_dir = os.path.join(self.result_dir, self.gan_model)
        self.result_dir = os.path.join(self.result_dir, self.architecture)
        self.losses_dir = os.path.join(self.result_dir, "losses")
        self.evaluation_dir = os.path.join(self.result_dir, "evaluations")
        check_folder(self.result_dir)
        check_folder(self.losses_dir)
        check_folder(self.evaluation_dir)

        # check gpu
        self.cuda = True if torch.cuda.is_available() else False
        print(" [*] Cuda: ", self.cuda)

        # set training parameters
        self.train_hist = {}

        # set evaluation parameters
        self.eval_hist = {}
        self.eval = {}

        # load dataset
        self.train_dataset = load_ts_dataset(dataroot=self.dataroot, dataset_name=self.dataset, dataset_mode="train")
        self.test_dataset = load_ts_dataset(dataroot=self.dataroot, dataset_name=self.dataset, dataset_mode="test")
        self.label_dim = self.train_dataset[1].shape[1]
        self.time_step = self.train_dataset[0].shape[1]
        self.data_loader = DataLoader(TensorDataset(self.train_dataset[0], self.train_dataset[1]),
                                      batch_size=self.batch_size, shuffle=True)

    def build_model(self):
        """
        RCGAN generator and discriminator initialization

        """

        if self.architecture == "RNN":
            self.generator = TSGeneratorRNN(input_dim=self.noise_dim, output_dim=1, label_dim=self.label_dim,
                                            time_step=self.time_step, hidden_dim=256)
            self.discriminator = TSDiscriminatorRNN(time_step=self.time_step, label_dim=self.label_dim,
                                                    hidden_dim=256, output_dim=1)
        elif self.architecture == "TCN":
            self.generator = TSGeneratorTCN(input_dim=self.noise_dim, output_dim=1, label_dim=self.label_dim,
                                            time_step=self.time_step, n_layers=1, n_channel=10, kernel_size=8, dropout=0)
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

    def train_model(self):
        """
        RCGAN model training

        """

        # training parameters
        self.train_hist['d_avg_losses'] = []
        self.train_hist['g_avg_losses'] = []
        self.train_hist['per_epoch_times'] = []
        self.train_hist['total_time'] = []

        # evaluation parameters
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
                y_zeros_ = torch.zeros(mini_batch, self.time_step)
                y_ones_ = torch.ones(mini_batch, self.time_step)
                if self.cuda:
                    y_zeros_, y_ones_ = y_zeros_.cuda(), y_ones_.cuda()

                # ---------------------
                # Train Discriminator
                # ---------------------

                # real data
                x_real_ = data.view(mini_batch, self.time_step, 1)
                y_real_ = torch.max(labels, 1)[1]
                if self.cuda:
                    x_real_, y_real_ = x_real_.cuda(), y_real_.cuda()
                d_real_ = self.discriminator(x_real_, y_real_).squeeze()
                d_real_loss = self.BCE_loss(d_real_, y_ones_)

                # fake data
                z_ = torch.randn(mini_batch, self.noise_dim * self.time_step)
                y_fake_ = torch.tensor(np.random.randint(0, self.label_dim, mini_batch)).type(torch.LongTensor)
                if self.cuda:
                    z_, y_fake_ = z_.cuda(), y_fake_.cuda()
                x_fake_ = self.generator(z_, y_fake_)
                d_fake_ = self.discriminator(x_fake_, y_fake_).squeeze()
                d_fake_loss = self.BCE_loss(d_fake_, y_zeros_)

                # backpropagation
                d_loss = (d_real_loss + d_fake_loss) / 2
                self.discriminator.zero_grad()
                d_loss.backward()
                self.optimizerD.step()

                # ---------------------
                # Train Generator
                # ---------------------

                x_fake_ = self.generator(z_, y_fake_)
                d_fake_ = self.discriminator(x_fake_, y_fake_).squeeze()

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
            self.visualize_losses(epoch)
            epoch_end_time = time.time()
            per_epoch_time = epoch_end_time - epoch_start_time
            self.train_hist['per_epoch_times'].append(per_epoch_time)

            # epoch evaluation
            argsM = {'dataroot': self.dataroot, 'dataset': self.dataset, 'gan_model': self.gan_model,
                     'generator': self.generator, 'noise_dim': self.noise_dim, 'evaluation_dir': self.evaluation_dir}
            metrics = Metrics(argsM=argsM, domain="time-series", batch_size=10, sample_size=1000)
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

    def visualize_losses(self, epoch):
        """
        RCGAN training losses visualization

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
        Animations generating for RCGAN

        """

        generate_animation(src_dir=self.losses_dir, dest_dir=self.result_dir, gif_name="losses",
                           num_epochs=self.num_epochs)
        generate_animation(src_dir=self.evaluation_dir + "/metrics/fid", dest_dir=self.evaluation_dir, gif_name="fid",
                           num_epochs=self.num_epochs)
        generate_animation(src_dir=self.evaluation_dir + "/metrics/is", dest_dir=self.evaluation_dir, gif_name="is",
                           num_epochs=self.num_epochs)

    def save(self, best_state_gen_is, best_state_gen_fid):
        """
        Saving best RCGAN models based on IS and FID metrics

        """

        torch.save(best_state_gen_is, self.result_dir + "/best_generator_is.pth")
        torch.save(best_state_gen_fid, self.result_dir + "/best_generator_fid.pth")
        with open(self.result_dir + "/training_history.pkl", 'wb') as f:
            pickle.dump(self.train_hist, f)
        with open(self.result_dir + "/evaluation_history.pkl", 'wb') as f:
            pickle.dump(self.eval_hist, f)

    def load(self, metric_name):
        """
        Loading best RCGAN models based on IS and FID metrics

        """

        self.build_model()
        if metric_name == "IS":
            self.generator.load_state_dict(torch.load(self.result_dir + "/best_generator_is.pth",
                                                      map_location=torch.device('cpu')))
        elif metric_name == "FID":
            self.generator.load_state_dict(torch.load(self.result_dir + "/best_generator_fid.pth",
                                                      map_location=torch.device('cpu')))
        return self.generator
