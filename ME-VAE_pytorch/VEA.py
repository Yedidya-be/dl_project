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

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(10000, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x=self.linear1(x)
        x = F.relu(x)
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        print(z)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 10000)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 100, 100))


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
    return autoencoder

latent_dims = 32

# data = torch.utils.data.DataLoader(
#         torchvision.datasets.MNIST('./data',
#                transform=torchvision.transforms.ToTensor(),
#                download=True),
#         batch_size=128,
#         shuffle=True)

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.file_list = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        data = np.load(file_path).astype(np.float32).reshape(100,100)
        data = data/70000
        # print(data)
        if self.transform:
            data = self.transform(data)
        return data

#example of usage
dir_path = r'X:\dl4cv_project\single_cell_data_without_background\Count00000_Point0000_ChannelPHASE_60x-100x_PH3,DAPI,A488,A555,A647_Seq0000\channel_PH3\OutputImages\train'
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
