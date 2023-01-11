import napari
import nd2
import utils
import numpy as np

def show_cell(image_path, index):
    image = np.load(image_path)
    viewer = napari.Viewer()
    viewer.add_image(image.phase_projection)
    viewer.add_image(image.dapi, name='dapi')
    viewer.add_image(image.ribo, name='dapi')
    viewer.add_image(image.dapi, name='dapi')
    viewer.add_labels(image.masks_outlines)
    viewer.show(block=True)  # wait until viewer window closes