# from https://avandekleut.github.io/vae/
import pandas as pd
import torch
import seaborn as sns
from torch.utils.data import DataLoader

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt;
import tqdm
import re

plt.rcParams['figure.dpi'] = 200
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(10000, 4096)
        self.linear2 = nn.Linear(4096, 2048)
        self.linear3 = nn.Linear(2048, 1024)
        self.linear4 = nn.Linear(1024, 512)
        self.linear5 = nn.Linear(512, latent_dims)
        self.linear6 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        mu = self.linear5(x)
        sigma = torch.exp(self.linear6(x))
        z = mu + sigma * self.N.sample(mu.shape)
        # print(z)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 2048)
        self.linear4 = nn.Linear(2048, 4096)
        self.linear5 = nn.Linear(4096, 10000)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = F.relu(self.linear4(z))
        z = torch.sigmoid(self.linear5(z))
        return z.reshape((-1, 1, 100, 100))



class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, train_dara, validation_data, epochs=3):
    opt = torch.optim.Adam(autoencoder.parameters())
    loss_func = nn.BCELoss()
    losses = []
    val_losses = []
    i = 0
    for epoch in tqdm.tqdm(range(epochs)):
        for x in train_dara:
            i += 1
            x = x.to(device)  # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            losses.append((i, loss))
            loss.backward()
            opt.step()

        # Validation
        with torch.no_grad():
            val_loss = 0
            for x in validation_data:
                x = x.to(device)
                x_hat = autoencoder(x)
                val_loss += ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            val_loss /= len(validation_data)
            val_losses.append((i, val_loss))
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
    # sns.scatterplot(losses)
    # sns.scatterplot(val_losses)
    # plt.show()
    return losses, autoencoder


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
        data = np.load(file_path)[0, ...].astype(np.float32).reshape(100, 100)
        data = data / 70000
        # print(data)
        if self.transform:
            data = self.transform(data)
        return data


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, df_path, transform=None):
        self.file_list = []
        self.df = pd.read_csv(df_path)
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.npy'):
                    image = re.search(r'fov_(\d+)_hyb_1', dirpath)
                    cell = re.search(r'label_(\d+)_bb', filename)
                    if image:
                        fov_number = 'fov_' + image.group(1) + '_' + cell.group(1)
                        self.file_list.append((os.path.join(dirpath, filename), fov_number))
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, s = self.file_list[idx]
        od = self.df.loc[(self.df.field_of_view == '_'.join(s.split('_')[:-1])) & (
                    self.df.cell_id == float(s.split('_')[-1])), 'sample_name'].item()
        data = np.load(file_path)[0, ...].astype(np.float32).reshape(100, 100)
        data = data / 70000
        if self.transform:
            data = self.transform(data)
        return data, od


dir_path = r'X:\yedidyab\dl_project\test_data\single_cell_data\fov_12_hyb_1'
dir_path = r'X:\dl4cv_project\single_cell_data_without_background\Count00000_Point0000_ChannelPHASE_60x-100x_PH3,DAPI,A488,A555,A647_Seq0000\channel_PH3\OutputImages\train'

def to_tensor32(data):
    data = data.astype(np.float32)
    return torchvision.transforms.ToTensor()(data)


data = torch.utils.data.DataLoader(
    NumpyDataset(dir_path, transform=to_tensor32),
    batch_size=64,
    shuffle=True
)

vae = VariationalAutoencoder(latent_dims).to(device)  # GPU
vae = train(vae, data)

# train_data, validation_data = torch.utils.data.random_split(data.dataset, [0.8, 0.2])
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# val_loader = DataLoader(validation_data, batch_size=64, shuffle=True)
# losses, vae = train(vae, train_loader, val_loader)

for i, x in enumerate(data):
    x_hat = vae(x.to(device))
    print(x_hat.shape)
    x = x.detach().numpy()
    x_hat = x_hat.detach().numpy()
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

def latant_z(autoencoder, data, num_batches=100):
    for i, x in enumerate(data):
        z = vae.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        break

def plot_loss(losses):
    sns.scatterplot(losses)

x = [i[0] for i in losses]
y = [i[1].item() for i in losses]
plt.scatter(x, y)
plt.show()
# def plot_latent(autoencoder, data, num_batches=100):
#     for i, (x, y) in enumerate(data):
#         z = autoencoder.encoder(x.to(device))
#         z = z.to('cpu').detach().numpy()
#         plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
#         if i > num_batches:
#             plt.colorbar()
#             plt.show()
#             break
#
# plot_latent(vae, data)
#
def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 100
    img = np.zeros((n * w, n * w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(0, 0).to('cpu').detach().numpy()
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])


# plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
