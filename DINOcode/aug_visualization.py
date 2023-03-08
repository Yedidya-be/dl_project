import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import random
import os
import utils

gn = utils.AddGaussianNoise(mean=0.5, std=0.01)
gb = transforms.GaussianBlur(kernel_size=3, sigma = (2., 40.))
rpd = utils.RandomPixelsDropOut(p=1.)
cc = transforms.CenterCrop(20)
norm = utils.Normalization()

def plot_aug(data_path, n_image = 5, channel = 0):
    listdir = os.listdir(data_path)
    random_files = random.sample(listdir, n_image)
    fig, axs = plt.subplots(5, n_image, figsize=(15, 10))
    # Plot the selected file pairs
    for i, image in enumerate(random_files):
        img = norm(torch.tensor(np.load(os.path.join(data_path, image))))

        axs[0, i].imshow(img[0, ...])
        axs[0, i].set_title('original')

        axs[1, i].imshow(gn(img)[0, ...])
        axs[1, i].set_title('AddGaussianNoise')

        axs[2, i].imshow(gb(img)[0,...])
        axs[2, i].set_title('GaussianBlur')

        axs[3, i].imshow(rpd(img)[0,...])
        axs[3, i].set_title('RandomPixelsDropOut')

        axs[4, i].imshow(cc(img)[0,...])
        axs[4, i].set_title('CenterCrop')


    # axs[0, 0].set_ylabel('input1')
    # axs[1, 0].set_ylabel('input1')
    # axs[2, 0].set_ylabel('out')
    # axs[3, 0].set_ylabel('recon')
    plt.tight_layout()
    plt.savefig(r"X:\dl4cv_project\try_out\plot.png", dpi=800)
    plt.show()

data_path = r'X:\dl4cv_project\rotate_single_cell\OutputImages\train'

plot_aug(data_path)
