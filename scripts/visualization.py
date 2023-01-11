import napari
import numpy as np

def show_cell(image_path):
    image = np.load(image_path)
    viewer = napari.Viewer()
    viewer.add_image(image[0, ...], name='phase proj')
    viewer.add_image(image[1, ...], name='masks')
    viewer.add_image(image[2, ...], name='dapi')
    viewer.add_image(image[3, ...], name='ribo')
    viewer.add_image(image[4, ...], name='wga')
    viewer.show(block=True)  # wait until viewer window closes