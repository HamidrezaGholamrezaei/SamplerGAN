"""
InfoGAN model implementation

"""

import pickle
import time
import itertools

import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from dataset import *
from metrics import Metrics
from models.infoGAN import Generator, Discriminator, Q_Module
from utils import *


# The negative log likelihood of normal distribution
class NormalNLLLoss:
    def __call__(self, x, mu, sigma):
        logli = -0.5 * (sigma.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(sigma.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())
        return nll


class InfoGAN (object):
    generator: Generator
    discriminator: Discriminator
    Q_Module: Q_Module
    optimizerG: optim.Adam
    optimizerD: optim.Adam
    optimizerQ: optim.Adam
    optimizerInfo: optim.Adam
    BCE_loss: nn.BCELoss
    CE_loss = nn.CrossEntropyLoss
    NLL_loss = NormalNLLLoss

    def __init__(self, args):
        # set parameters
        self.dataroot = args.dataroot
        self.gan_model = args.gan_model
        self.dataset = args.dataset
        self.result_dir = args.result_dir
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.noise_dim = args.noise_dim
        self.code_dim = args.code_dim
        self.image_size = args.image_size
        self.num_epochs = args.num_epochs
        self.num_gpu = args.num_gpu
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        # config directories
        self.result_dir = os.path.join(self.result_dir, self.dataset)
        self.result_dir = os.path.join(self.result_dir, self.gan_model)
        self.images_dir = os.path.join(self.result_dir, "images")
        self.losses_dir = os.path.join(self.result_dir, "losses")
        self.evaluation_dir = os.path.join(self.result_dir, "evaluations")
        check_folder(self.result_dir)
        check_folder(self.images_dir)
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
        dataset, label_dim = load_image_dataset(self.dataroot, self.dataset, "train", self.image_size)
        self.label_dim = label_dim
        self.num_samples = label_dim ** 2
        self.data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        data = self.data_loader.__iter__().__next__()[0]
        self.channel_dim = data.shape[1]

        # fixed noise & labels for testing
        temp_noise = torch.randn(self.label_dim, self.noise_dim)
        self.fixed_noise = temp_noise
        fixed_l_ = torch.zeros(self.label_dim, 1)
        for i in range(self.label_dim - 1):
            self.fixed_noise = torch.cat([self.fixed_noise, temp_noise], 0)
            temp = torch.ones(self.label_dim, 1) + i
            fixed_l_ = torch.cat([fixed_l_, temp], 0)
        self.fixed_label = torch.zeros(self.num_samples, self.label_dim)
        self.fixed_label.scatter_(1, fixed_l_.type(torch.LongTensor), 1)
        if self.cuda:
            self.fixed_noise, self.fixed_label = self.fixed_noise.cuda(), self.fixed_label.cuda()
        if self.code_dim != 0:
            self.fixed_code = torch.zeros((self.num_samples, self.code_dim))
            if self.cuda:
                self.fixed_code = self.fixed_code.cuda()

    def build_model(self):
        """
        InfoGAN generator and discriminator initialization

        """

        self.generator = Generator(num_gpu=self.num_gpu, input_dim=self.noise_dim, output_dim=self.channel_dim,
                                   label_dim=self.label_dim, code_dim=self.code_dim, image_size=self.image_size,
                                   dataset_name=self.dataset)
        self.discriminator = Discriminator(num_gpu=self.num_gpu, input_dim=self.channel_dim, output_dim=1,
                                           label_dim=self.label_dim, code_dim=self.code_dim, image_size=self.image_size,
                                           dataset_name=self.dataset)
        self.Q_Module = Q_Module(label_dim=self.label_dim, code_dim=self.code_dim + 1)
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizerQ = optim.Adam(self.Q_Module.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.Q_Module.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
            self.NLL_loss = NormalNLLLoss()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()
            self.NLL_loss = NormalNLLLoss()

    def print_model(self):
        print('---------- Networks architecture -------------')
        print_network(self.generator)
        print_network(self.discriminator)
        print_network(self.Q_Module)
        print('-----------------------------------------------')

    def train_model(self):
        """
        InfoGAN model training

        """

        # training parameters
        self.train_hist['d_avg_losses'] = []
        self.train_hist['g_avg_losses'] = []
        self.train_hist['q_avg_losses'] = []
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
            q_losses = []

            # epoch starts
            epoch_start_time = time.time()

            for i, (images, labels) in enumerate(self.data_loader):
                mini_batch = images.size()[0]
                y_zeros_ = torch.zeros(mini_batch)
                y_ones_ = torch.ones(mini_batch)
                if self.cuda:
                    y_zeros_, y_ones_ = y_zeros_.cuda(), y_ones_.cuda()

                # ---------------------
                # Train Discriminator
                # ---------------------

                # real data
                x_ = images
                if self.cuda:
                    x_ = x_.cuda()
                d_real_, _ = self.discriminator(x_)
                d_real_loss = self.BCE_loss(d_real_.squeeze(), y_ones_)

                # fake data
                z_ = torch.randn(mini_batch, self.noise_dim)
                y_fake_ = torch.tensor(np.random.randint(0, self.label_dim, mini_batch)).type(torch.LongTensor)
                y_disc_ = torch.zeros((mini_batch, self.label_dim)) \
                    .scatter_(1, y_fake_.type(torch.LongTensor).unsqueeze(1), 1)
                if self.cuda:
                    z_, y_disc_, y_fake_ = z_.cuda(), y_disc_.cuda(), y_fake_.cuda()
                if self.code_dim != 0:
                    y_cont_ = torch.from_numpy(np.random.uniform(-1, 1, size=(mini_batch, self.code_dim))) \
                        .type(torch.FloatTensor)
                    if self.cuda:
                        y_cont_ = y_cont_.cuda()
                if self.code_dim != 0:
                    gen_image = self.generator(z_, y_disc_, y_cont_)
                elif self.code_dim == 0:
                    gen_image = self.generator(z_, y_disc_)
                d_fake_, _ = self.discriminator(gen_image.detach())
                d_fake_loss = self.BCE_loss(d_fake_.squeeze(), y_zeros_)

                # backpropagation
                d_loss = d_real_loss + d_fake_loss
                self.discriminator.zero_grad()
                d_loss.backward(retain_graph=True)
                self.optimizerD.step()

                # ---------------------
                # Train Generator
                # ---------------------

                if self.code_dim != 0:
                    gen_image = self.generator(z_, y_disc_, y_cont_)
                elif self.code_dim == 0:
                    gen_image = self.generator(z_, y_disc_)
                d_fake_, shared_output = self.discriminator(gen_image)
                gen_loss = self.BCE_loss(d_fake_.squeeze(), y_ones_)

                q_disc, q_mu, q_sigma = self.Q_Module(shared_output)
                # Calculating loss for discrete latent code
                disc_loss = self.CE_loss(q_disc, y_fake_)
                # Calculating loss for continuous latent code
                cont_loss = 0
                if self.code_dim != 0:
                    cont_loss = self.NLL_loss(y_cont_, q_mu, q_sigma) * 0.1

                # backpropagation
                g_loss = gen_loss + disc_loss + cont_loss
                self.generator.zero_grad()
                g_loss.backward(retain_graph=True)

                # ---------------------
                # Information Optimization
                # ---------------------

                q_loss = disc_loss + cont_loss
                self.discriminator.zero_grad()
                self.Q_Module.zero_grad()
                q_loss.backward()

                # ---------------------
                # Optimization
                # ---------------------

                self.optimizerD.step()
                self.optimizerG.step()
                self.optimizerQ.step()

                # logging
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                q_losses.append(q_loss.item())
                # info_losses.append(info_loss.item())
                print('Epoch [%d/%d], Step [%d/%d], d_loss: %.4f, g_loss: %.4f, q_loss: %.4f' %
                      (epoch+1, self.num_epochs, i+1, len(self.data_loader),
                       d_loss.item(), g_loss.item(), q_loss.item()))

            d_avg_loss = torch.mean(torch.FloatTensor(d_losses))
            g_avg_loss = torch.mean(torch.FloatTensor(g_losses))
            q_avg_loss = torch.mean(torch.FloatTensor(q_losses))
            self.train_hist['d_avg_losses'].append(d_avg_loss)
            self.train_hist['g_avg_losses'].append(g_avg_loss)
            self.train_hist['q_avg_losses'].append(q_avg_loss)

            # epoch ends
            self.visualize_results(epoch)
            self.visualize_losses(epoch)
            epoch_end_time = time.time()
            per_epoch_time = epoch_end_time - epoch_start_time
            self.train_hist['per_epoch_times'].append(per_epoch_time)

            # epoch evaluation
            argsM = {'dataroot': self.dataroot, 'dataset': self.dataset, 'image_size': self.image_size,
                     'num_workers': self.num_workers, 'gan_model': self.gan_model, 'generator': self.generator,
                     'noise_dim': self.noise_dim, 'code_dim': self.code_dim, 'evaluation_dir': self.evaluation_dir}
            metrics = Metrics(argsM=argsM, domain="image", batch_size=50, sample_size=50000)
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
        InfoGAN generated images visualization

        """

        self.generator.eval()
        if self.code_dim != 0:
            gen_images = self.generator(self.fixed_noise, self.fixed_label, self.fixed_code)
        elif self.code_dim == 0:
            gen_images = self.generator(self.fixed_noise, self.fixed_label)
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

        self.generator.train()

    def visualize_losses(self, epoch):
        """
        InfoGAN training losses visualization

        """

        d_losses = self.train_hist['d_avg_losses']
        g_losses = self.train_hist['g_avg_losses']
        q_losses = self.train_hist['q_avg_losses']
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.num_epochs)
        ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
        plt.xlabel('Epoch {0}'.format(epoch + 1))
        plt.ylabel('Loss values')
        plt.plot(d_losses, label='Discriminator')
        plt.plot(g_losses, label='Generator')
        plt.plot(q_losses, label='Q_Module')
        plt.legend()
        plt_name = epoch + 1
        plt.savefig(self.losses_dir + "/%d.png" % plt_name)
        plt.close()

    def generate_animations(self):
        """
        Animations generating for InfoGAN

        """

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
        Saving best InfoGAN models based on IS and FID metrics

        """

        torch.save(best_state_gen_is, self.result_dir + "/best_generator_is.pth")
        torch.save(best_state_gen_fid, self.result_dir + "/best_generator_fid.pth")
        with open(self.result_dir + "/training_history.pkl", 'wb') as f:
            pickle.dump(self.train_hist, f)
        with open(self.result_dir + "/evaluation_history.pkl", 'wb') as f:
            pickle.dump(self.eval_hist, f)

    def load(self, metric_name):
        """
        Loading best InfoGAN models based on IS and FID metrics

        """

        self.build_model()
        if metric_name == "IS":
            self.generator.load_state_dict(torch.load(self.result_dir + "/best_generator_is.pth",
                                                      map_location=torch.device('cpu')))
        elif metric_name == "FID":
            self.generator.load_state_dict(torch.load(self.result_dir + "/best_generator_fid.pth",
                                                      map_location=torch.device('cpu')))
        return self.generator
