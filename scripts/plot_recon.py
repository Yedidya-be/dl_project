import numpy as np
import matplotlib.pyplot as plt
import random
import os


def plot_recon(input1, input2, out, recon, channel=0):
    color = 'Blues' if channel == 2 else 'Reds' if channel == 3 else 'Greens' if channel == 4 else 'gray'



    input1_data = np.load(input1)[..., channel]
    input2_data = np.load(input2)[..., channel]
    out_data = np.load(out)[..., channel]
    recon_data = np.load(recon)[..., channel]

    # Randomly select 3 pairs of files
    random_idx = random.sample(range(0, 99), 3)

    fig, axs = plt.subplots(4, 3, figsize=(15, 10))

    # Plot the selected file pairs
    for i, idx in enumerate(random_idx):
        _input1_data = input1_data[idx, ...]
        _input2_data = input2_data[idx, ...]
        _out_data = out_data[idx, ...]
        _recon_data = recon_data[idx, ...]

        axs[0, i].imshow(_input1_data, cmap=color)
        axs[0, i].set_title(idx)
        axs[1, i].imshow(_input2_data, cmap=color)
        axs[1, i].set_title(idx)
        axs[2, i].imshow(_out_data, cmap=color)
        axs[2, i].set_title(idx)
        axs[3, i].imshow(_recon_data, cmap=color)
        axs[3, i].set_title(idx)
    axs[0, 0].set_ylabel('input1')
    axs[1, 0].set_ylabel('input1')
    axs[2, 0].set_ylabel('out')
    axs[3, 0].set_ylabel('recon')
    plt.tight_layout()
    plt.savefig(r"X:\dl4cv_project\try_out\plot.png")
    plt.show()

main_file = r'X:\dl4cv_project\rotate_single_cell\Outputs_32'
# main_file = r'X:\dl4cv_project\data_for_einav2\all_channels\Outputs_l32'
input1 = main_file + r'\input2_data.npy'
input2 = main_file + r'\input2_data.npy'
out = main_file + r'\out1_data.npy'
recon = main_file + r'\recon.npy'
plot_recon(input1, input2, out, recon, channel=0)
