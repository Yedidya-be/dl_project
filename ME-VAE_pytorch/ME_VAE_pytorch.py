from typing import List

import torch
from models import BaseVAE
from tensorflow import Tensor
from torch import nn
from torch.nn import functional as F


class BetaVAE():
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,args)
        """ initialize model with argument parameters and build
        """
        self.chIndex = args.chIndex
        self.data_dir = args.data_dir
        self.image_dir = args.image_dir
        self.save_dir = args.save_dir
        self.out1_dir = args.out1_dir
        self.input2_dir = args.input2_dir

        self.use_vaecb = args.use_vaecb
        self.do_vaecb_each = args.do_vaecb_each
        self.use_clr = args.use_clr
        self.earlystop = args.earlystop

        self.latent_dim = args.latent_dim
        self.nlayers = args.nlayers
        self.inter_dim = args.inter_dim
        self.kernel_size = args.kernel_size
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.nfilters = args.nfilters
        self.learn_rate = args.learn_rate

        self.epsilon_std = args.epsilon_std

        self.latent_samp = args.latent_samp
        self.num_save = args.num_save

        self.do_tsne = args.do_tsne
        self.verbose = args.verbose
        self.phase = args.phase
        self.steps_per_epoch = args.steps_per_epoch

        self.data_size = len(os.listdir(os.path.join(self.data_dir, 'train')))
        self.file_names = os.listdir(os.path.join(self.data_dir, 'train'))

        self.image_size = args.image_size
        self.nchannel = args.nchannel
        self.image_res = args.image_res
        self.show_channels = args.show_channels

        if self.steps_per_epoch == 0:
            self.steps_per_epoch = self.data_size // self.batch_size

        self.build_model()

        self.conv = nn.ModuleList()
        filters = self.nfilters
        for i in range(self.nfilters):
            self.conv.append(
                nn.Conv2d(in_channels=filters, out_channels=filters * 2, kernel_size=kernel_size, stride=2, padding=1))
            filters *= 2

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filters * int(self.image_size / 2 ** self.nlayers) * int(self.image_size / 2 ** self.nlayers), self.inter_dim)
        self.fc_mean = nn.Linear(self.inter_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.inter_dim, self.latent_dim)



        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def sampling(self, sample_args):
        """ sample latent layer from normal prior
        """

        z_mean, z_log_var = sample_args

        epsilon = torch.randn(z_mean.shape[0], self.latent_dim, device=z_mean.device) * self.epsilon_std
        
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon

        z = z.to(current_device)

        return z
    


    def encoder1(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result1 = self.encoder1(input)
        result1 = torch.flatten(result1, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu1 = self.fc_mu1(result1)
        log_var1 = self.fc_var1(result1)

        return [mu1, log_var1]

    def encoder2(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result2 = self.encoder2(input)
        result2 = torch.flatten(result2, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu2 = self.fc_mu2(result2)
        log_var2 = self.fc_var2(result2)

        return [mu2, log_var2]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2) # need to be change
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu1, log_var1 = self.encoder1(input)
        mu2, log_var2 = self.encoder2(input)
        z1 = self.reparameterize(mu1, log_var1)
        z2 = self.reparameterize(mu2, log_var2)
        z = z1 @z2
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}



    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
