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



class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 7, 2, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU()
        )
        self.mu = nn.Linear(4608, latent_dims) #TO CHANGE!
        self.log_var = nn.Linear(4608, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.mu.weight)
        nn.init.kaiming_uniform_(self.log_var.weight)

    def forward(self, x):
        # print(summary(.shape)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        var = torch.exp(log_var)
        z = mu + var * self.N.sample(mu.shape)
        self.kl = 0.5 * torch.sum(var + mu ** 2 - 1 - log_var)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dims, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=3, padding=1, output_padding = 2),
        )

    def forward(self, z):
        z = z.view(-1, latent_dims, 1, 1)
        x = self.decoder(z)
        x = torch.sigmoid(x)
        return x




class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



def train(autoencoder, data, epochs=2):
    opt = torch.optim.Adam(autoencoder.parameters())
    loss_values = []
    for epoch in tqdm.tqdm(range(epochs)):
        for x in data:
            print(x.shape)
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


latent_dims = 256

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
        data = np.load(file_path)[0, ...].astype(np.float32).reshape(100,100)
        data = data/70000
        # print(data)
        if self.transform:
            data = self.transform(data)
        return data

#example of usage
dir_path = r'X:\dl4cv_project\single_cell_data_without_background\Count00000_Point0000_ChannelPHASE_60x-100x_PH3,DAPI,A488,A555,A647_Seq0000\channel_PH3\OutputImages\train'
dir_path = r'X:\yedidyab\dl_project\test_data\single_cell_data\fov_12_hyb_1'
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