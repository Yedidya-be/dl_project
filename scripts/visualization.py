import matplotlib.pyplot as plt
import napari
import numpy as np
import os
import random
import re

def show_napari(image_path):
    image = np.load(image_path)
    viewer = napari.Viewer()
    viewer.add_image(image[0, ...], name='phase proj')
    viewer.add_image(image[1, ...], name='masks')
    viewer.add_image(image[2, ...], name='dapi')
    viewer.add_image(image[3, ...], name='ribo')
    viewer.add_image(image[4, ...], name='wga')
    viewer.show(block=True)  # wait until viewer window closes

def show_plt_size_condition(list_of_path, channel):
    # while True:
    image_path = random.choice(result)
    image = np.load(image_path)
    re
    plt.imshow(image[channel, ...])
    plt.title(f'num mask pixels = {np.count_nonzero(image[1, ...])}')
    plt.show()



path = r'X:\yedidyab\dl_project\single_cell_data'
result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
          os.path.splitext(f)[1] == '.npy']


def plot_matrices():
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    titles = ["Phase", "Masks", "Dapi", "Ribo", "WGA"]
    for i in range(5):
        image_path = random.choice(result)
        print(image_path)
        matrix = np.load(image_path)
        for j in range(5):
            ax = axes[i, j]
            if j == 0:
                ax.imshow(matrix[j,...], cmap='gray')
            elif j == 1:
                ax.imshow(matrix[j,...], cmap='viridis')
            elif j == 2:
                ax.imshow(matrix[j,...], cmap='Blues')
            elif j == 3:
                ax.imshow(matrix[j,...], cmap='Greens')
            elif j == 4:
                ax.imshow(matrix[j,...], cmap='Reds')
        title = re.search(r'Point(\d+)_Channel', image_path).group(1)
        axes[i, 0].set_ylabel(title, rotation=45, size='large')

    for j in range(5):
        axes[0,j].set_title(titles[j])

    plt.savefig(r'X:\yedidyab\dl_project\results\figures\data.png')
    plt.show()
plot_matrices()