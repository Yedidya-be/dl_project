import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import os
import tqdm
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VariationalEncoder(nn.Module):
    def __init__(self, image_size=128, nchannel=1, nfilters=16, kernel_size=3, nlayers=3, inter_dim=128, latent_dim=32):
        super(VariationalEncoder, self).__init__()

        self.kl = None
        self.image_size = image_size
        self.nchannel = nchannel
        self.nfilters = nfilters
        self.kernel_size = kernel_size
        self.nlayers = nlayers
        self.inter_dim = inter_dim

        self.latent_dim = latent_dim
        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.nchannel,
                                            out_channels=self.nfilters,
                                            kernel_size=kernel_size,
                                            stride=2,
                                            padding=1))
        self.conv.add_module('ReLu-0', nn.ReLU())
        filters = nfilters
        kernel_size = kernel_size
        for i in range(nlayers):
            self.conv.add_module(f'conv_{i}', nn.Conv2d(in_channels=filters,
                                                        out_channels=filters * 2,
                                                        kernel_size=kernel_size,
                                                        stride=2,
                                                        padding=1))
            self.conv.add_module(f'relu_{i}', nn.ReLU())
            filters *= 2

        self.conv.add_module('flat', nn.Flatten())
        self.fc1 = nn.Linear(8192, self.inter_dim)  # change this hardcode! (need to calc shapes)
        self.fc_mean = nn.Linear(inter_dim, latent_dim)  # change and add layer name
        self.fc_log_var = nn.Linear(inter_dim, latent_dim)  # change and add layer name

    def encode(self, x):
        x = self.conv(x)
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        self.kl = (log_var ** 2 + mean ** 2 - torch.log(log_var) - 1 / 2).sum()
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return z  # , mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim=32, image_size=(1, 128, 128), nlayers=3, filters=64):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim

        self.image_size = image_size
        self.nlayers = nlayers
        self.filters = filters

        # Calculate the number of units for the first dense layer
        num_units = int(np.prod(image_size[1:]))
        print(f'{num_units =}')
        # First dense layer
        self.fc = nn.Linear(latent_dim, num_units)
        self.fc_activation = nn.ReLU()

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(self.nlayers):
            in_channels = int(filters)
            out_channels = int(filters / 2)
            conv_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                            padding=1, output_padding=1)

            self.conv_layers.append(conv_layer)
            filters /= 2

        # Last layer
        self.output_layer = nn.ConvTranspose2d(in_channels=int(filters), out_channels=1, kernel_size=3,
                                               stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.fc_activation(x)
        print(x.shape)
        x = x.view(-1, 64, 16, 16)

        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            x = nn.ReLU()(x)

        x = self.output_layer(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dim=latent_dims)
        self.decoder = Decoder(latent_dim=latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, data, epochs=2):
    opt = torch.optim.Adam(autoencoder.parameters())
    loss_values = []
    for epoch in tqdm.tqdm(range(epochs)):
        for x in data:
            x = x.to(device)  # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
            loss_values.append(loss.item())
            print(loss)
    sns.lineplot(range(len(loss_values)), loss_values)
    plt.show()
    return autoencoder


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, image_size=128):
        self.image_size = image_size
        self.root_dir = root_dir
        self.file_list = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        data = np.load(file_path)[0, ...].astype(np.float32).reshape(self.image_size, self.image_size)
        data = data / 70000
        # print(data)
        if self.transform:
            data = self.transform(data)
        return data

dir_path = r'X:\dl4cv_project\single_cell_without_background_128\Count00000_Point0013_ChannelPHASE_60x-100x_PH3,DAPI,A488,A555,A647_Seq0013'

def to_tensor32(data):
    data = data.astype(np.float32)
    return torchvision.transforms.ToTensor()(data)

data = torch.utils.data.DataLoader(
    NumpyDataset(dir_path, transform=to_tensor32),
    batch_size=64,
    shuffle=True
)

vae = VariationalAutoencoder(32).to(device) # GPU
vae = train(vae, data)


# v = VariationalEncoder()
# print(summary(v, [(1, 128, 128)], 64))
#
# d = Decoder()
# print(summary(d, [(32,)]))
#
vae = VariationalAutoencoder(32)
print(summary(vae, [(1, 128, 128)], 64))
# vae(torch.rand([64, 1, 128, 128]))
