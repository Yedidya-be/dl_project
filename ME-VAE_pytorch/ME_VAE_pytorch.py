#from https://avandekleut.github.io/vae/

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import os
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torchsummary import summary
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalEncoder(nn.Module):
    def __init__(self, image_size=128, nchannel=1, nfilters=16, kernel_size=3, nlayers=3, inter_dim=128, latent_dim=32):
        super(VariationalEncoder, self).__init__()

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
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return z, mean, log_var


v = VariationalEncoder()
print(summary(v, (1, 128, 128), 64))

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, image_size=128, nchannel=1, nfilters=16, kernel_size=3, nlayers=3, inter_dim=128, latent_dim=32):
        super(Decoder, self).__init__()

        self.image_size = image_size
        self.nchannel = nchannel
        self.nfilters = nfilters
        self.kernel_size = kernel_size
        self.nlayers = nlayers
        self.inter_dim = inter_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(latent_dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, 8192)
        self.deconv = nn.Sequential()
        filters = self.nfilters * (nlayers+1)* 2
        print(filters)
        # self.deconv.add_module(f'upsample-0', nn.Upsample(scale_factor=2, mode='nearest'))
        for i in range(nlayers):
            self.deconv.add_module(f'deconv_{i}', nn.Conv2d(in_channels=filters,
                                                            out_channels=filters//2,
                                                            kernel_size=kernel_size,
                                                            stride=2,
                                                            padding=1))
            self.deconv.add_module(f'relu_{i}', nn.ReLU())
            filters = filters//2
        self.deconv.add_module(f'deconv_{nlayers}', nn.Conv2d(in_channels=filters,
                                                               out_channels=self.nchannel,
                                                               kernel_size=kernel_size,
                                                               stride=2,
                                                               padding=1))
        self.deconv.add_module(f'sigmoid_{nlayers}', nn.Sigmoid())

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, self.nfilters * 2**(self.nlayers-1), (self.image_size//2**(self.nlayers))**2)
        x = self.deconv(x)
        return x

v = Decoder()
print(summary(v, (32), 64))

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VAE(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



def train(autoencoder, data, epochs=2):
    opt = torch.optim.Adam(autoencoder.parameters())
    loss_values = []
    for epoch in tqdm.tqdm(range(epochs)):
        for x in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
            loss_values.append(loss.item())
    sns.lineplot(range(len(loss_values)), loss_values)
    plt.show()
    return autoencoder


latent_dims = 32

# data = torch.utils.data.DataLoader(
#         torchvision.datasets.MNIST('./data',
#                transform=torchvision.transforms.ToTensor(),
#                download=True),
#         batch_size=128,
#         shuffle=True)

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, image_size = 128):
        self.image_size = image_size
        self.root_dir = root_dir
        self.file_list = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        data = np.load(file_path)[0, ...].astype(np.float32).reshape(self.image_size, self.image_size)
        data = data/70000
        # print(data)
        if self.transform:
            data = self.transform(data)
        return data

#example of usage
dir_path = r'X:\dl4cv_project\single_cell_data_without_background\Count00000_Point0000_ChannelPHASE_60x-100x_PH3,DAPI,A488,A555,A647_Seq0000\channel_PH3\OutputImages\train'
dir_path = r'X:\yedidyab\dl_project\test_data\single_cell_data\fov_12_hyb_1'
dir_path = r'X:\dl4cv_project\single_cell_without_background_128\Count00000_Point0036_ChannelPHASE 60x-100x PH3,DAPI,A488,A555,A647_Seq0036'
# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor().to(torch.float32)])

def to_tensor32(data):
    data = data.astype(np.float32)
    return torchvision.transforms.ToTensor()(data)

data = torch.utils.data.DataLoader(
    NumpyDataset(dir_path, transform=to_tensor32),
    batch_size=64,
    shuffle=True
)

vae = VariationalAutoencoder(latent_dims).to(device) # GPU
vae = train(vae, data)

for i, x in enumerate(data):
    x_hat = vae(x.to(device))
    x_hat = x_hat.detach().numpy()
    x = x.detach().numpy()
    break

# Select 3 random indices
indices = np.random.randint(0, 64, size=3)

# Plot the selected images
for i, index in enumerate(indices):
    plt.subplot(3, 2, 2 * i + 1)
    plt.imshow(x[index, 0, :, :], cmap='gray')
    plt.title("Original Image")

    plt.subplot(3, 2, 2 * i + 2)
    plt.imshow(x_hat[index, 0, :, :], cmap='gray')
    plt.title("Reconstructed Image")

plt.show()

print(summary(vae, (1,100,100), 64))