import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
import nd2
import napari
import copy
from scipy.ndimage import gaussian_filter


def get_phase_projection(image, chunk_size=128, pixels_per_cell=150, is_show_proj=False):
    """"
    Written by Yedidya Ben-Eliyahu, 05.25.2022

    Purpose: Find the local best plane of focus in curved 3D phase images.

    Approach: Divide the image into windows (chunks) and find the optimal plate of focus for each.
    Generate a 2D projection using the intensity values fro
    m the choosen planes.

    input:
    - img_phase = multi-plane phase image
    - chunk_size = window pixel size for calculating median plane of focus; num must divid

    Output:
    - phase projection = 2D corrected image
    -
    """
    # find best plane of focus of cells
    img_phase = image[:, 0, :, :]
    img_dapi = image[:, 1, :, :]

    img_min_int_proj = np.min(img_phase, axis=0)  # standard min intensity projection
    thresh = threshold_otsu(img_min_int_proj)  # find the best threshold value
    img_thresh_bw = img_min_int_proj > thresh  # background = 1, cells = 0
    img_thresh_bw = np.invert(img_thresh_bw)

    if is_show_proj:
        plt.imshow(img_thresh_bw)
        plt.title('img_thresh_bw (Old TH)')
        plt.colorbar()
        plt.show()

    dapi_max_int_proj = np.max(img_dapi, axis=0)  # standard min intensity projection
    thresh_dapi = threshold_otsu(dapi_max_int_proj)  # find the best threshold value
    dapi_thresh_bw = dapi_max_int_proj > thresh_dapi  # background = 1, cells = 0

    if is_show_proj:
        plt.imshow(dapi_thresh_bw)
        plt.title('dapi_thresh_bw')
        plt.colorbar()
        plt.show()

    comb = dapi_thresh_bw.astype(int) + img_thresh_bw.astype(int)
    comb = comb < 2

    if is_show_proj:
        plt.imshow(comb)
        plt.title('comb')
        plt.colorbar()
        plt.show()

    img_min_int_zslice_mat = img_phase.argmin(axis=0).astype(
        float)  # For each pixel, find the z-plane with min intensity
    masked_layers = img_min_int_zslice_mat
    masked_layers[comb] = np.nan  # keep only the cell data, nan important as 0 vals mess with median calc downstream

    img_size = masked_layers.shape[0]  # find the num of chunks
    num_of_chunks = int(img_size / chunk_size)
    optimal_layer_mat = np.zeros(
        masked_layers.shape)  # median layer values in chunk per pixel -> used for projection downstream

    i = img_size
    windows = []
    while i > chunk_size - 1:
        windows.append(int(i))
        i = i / 2

    phase_projections = []
    opts = []
    op = copy.deepcopy(optimal_layer_mat)
    for window in windows:
        for i in range(0, img_size - 1, window):
            for j in range(0, img_size - 1, window):
                chunk_median_z_layer = np.nanmedian(
                    masked_layers[i:i + window, j:j + window])  # nan insensitive median of entire chunk
                if np.isnan(chunk_median_z_layer) or np.count_nonzero(
                        ~np.isnan(masked_layers[i:i + window, j:j + window])) < pixels_per_cell:
                    chunk_median_z_layer = np.median(op[i:i + window, j:j + window])
                optimal_layer_mat[i:i + window, j:j + window] = chunk_median_z_layer

        phase_projection = np.take_along_axis(img_phase, optimal_layer_mat.astype(int)[np.newaxis], axis=0)[0]

        # save projection per window size
        pp = copy.deepcopy(phase_projection)
        phase_projections.append(pp)
        op = copy.deepcopy(optimal_layer_mat)
        opts.append(op)

        if is_show_proj:
            plt.imshow(optimal_layer_mat)
            plt.colorbar()
            plt.show()

    inner_mat = np.zeros(
        masked_layers.shape)  # median layer values in chunk per pixel -> used for projection downstream
    width = 10

    opt1 = copy.deepcopy(opts[-1])
    ph1 = copy.deepcopy(phase_projection)

    for i in range(chunk_size, img_size - chunk_size + 1, chunk_size):
        for j in range(chunk_size, img_size - chunk_size + 1, chunk_size):
            i_box = opts[-1][i - chunk_size: i, j - width: j + width]
            j_box = opts[-1][i - width: i + width, j - chunk_size: j]
            i_lay = np.sort(np.unique(i_box).astype(int))
            j_lay = np.sort(np.unique(j_box).astype(int))
            if i_lay[-1] - i_lay[0] > .1:
                i_slice = img_phase[i_lay[0]:i_lay[-1]+1, i - chunk_size: i, j - width: j + width]
                men = np.mean(i_slice, axis=0)
                ph1[i - chunk_size: i, j - width: j + width] = men
                opt1[i - chunk_size: i, j - width: j + width] = 14

            if j_lay[-1] - j_lay[0] > .1:
                j_slice = img_phase[j_lay[0]:j_lay[-1]+1, i - width: i + width, j - chunk_size: j]
                men = np.mean(j_slice, axis=0)
                ph1[i - width: i + width, j - chunk_size: j] = men
                opt1[i - width: i + width, j - chunk_size: j] = 14

    if is_show_proj:
        plt.imshow(opt1)
        plt.colorbar()
        plt.title('test')
        plt.show()

        #
        viewer = napari.Viewer()
        viewer.add_image(phase_projection, name='old_projection')
        viewer.add_image(ph1, name='corected(mean)')
        viewer.add_image(opts[-1], name='optimal_layer')
        viewer.add_image(opt1, name='optimal_layer with borders')
        viewer.add_image(img_phase, name='img_phase')
        viewer.show(block=True)

    return phase_projection, optimal_layer_mat



