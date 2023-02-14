import numpy as np
import matplotlib.pyplot as plt
import random
import os


def plot_pairs(rotation_path, shifted_path, out_path, channel=0):
    # Get the list of npy files in each path
    rotation_files = [f for f in os.listdir(rotation_path) if f.endswith('.npy')]
    shifted_files = [f for f in os.listdir(shifted_path) if f.endswith('.npy')]
    out_files = [f for f in os.listdir(shifted_path) if f.endswith('.npy')]

    # Randomly select 3 pairs of files
    random_files = random.sample(list(zip(rotation_files, shifted_files, out_files)), 3)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))

    # Plot the selected file pairs
    for i, (rot_file, shift_file, out_file) in enumerate(random_files):
        rot_data = np.load(os.path.join(rotation_path, rot_file))[channel, ...]
        shift_data = np.load(os.path.join(shifted_path, shift_file))[channel, ...]
        out_data = np.load(os.path.join(out_path, out_file))[channel, ...]

        axs[0, i].imshow(rot_data)
        axs[0, i].set_title(rot_file)
        axs[1, i].imshow(shift_data)
        axs[1, i].set_title(shift_file)
        axs[2, i].imshow(out_data)
        axs[2, i].set_title(out_file)
    axs[0, 0].set_ylabel('rot')
    axs[1, 0].set_ylabel('shift')
    axs[2, 0].set_ylabel('out')
    plt.tight_layout()
    plt.show()

plot_pairs(r'X:\dl4cv_project\data_for_einav2\all_channels\shifted\train',r'X:\dl4cv_project\data_for_einav2\all_channels\rotation\train',r'X:\dl4cv_project\data_for_einav2\all_channels\OutputImages\train', channel=0)
