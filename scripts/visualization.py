import matplotlib.pyplot as plt
import napari
import numpy as np
import os
import random

def show_napari(image_path):
    image = np.load(image_path)
    viewer = napari.Viewer()
    viewer.add_image(image[0, ...], name='phase proj')
    viewer.add_image(image[1, ...], name='masks')
    viewer.add_image(image[2, ...], name='dapi')
    viewer.add_image(image[3, ...], name='ribo')
    viewer.add_image(image[4, ...], name='wga')
    viewer.show(block=True)  # wait until viewer window closes

def show_plt_size_condition(list_of_path):
    # while True:
    image_path = random.choice(result)
    image = np.load(image_path)
    print(image_path)
    plt.imshow(image[1, ...])
    plt.title(f'num mask pixels = {np.count_nonzero(image[1, ...])}')
    plt.show()



path = r'X:\yedidyab\dl_project\single_cell_data'
result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if
          os.path.splitext(f)[1] == '.npy']

show_cell(random.choice(result))