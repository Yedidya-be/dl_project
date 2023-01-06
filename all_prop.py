# from phase_projection import get_phase_projection as gt
# from segmentation import get_segmentation
import matplotlib.pyplot as plt
import napari
import nd2
import numpy as np
import pandas as pd
from esda.moran import Moran
from pp_new import get_phase_projection as gt
from segmentation import get_segmentation
from libpysal.weights import lat2W
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import rotate
# import matplotlib
# matplotlib.use('TkAgg')
from skimage import measure


class Img:

    def __init__(self, path,
                 phase_c = 0,
                 dapi_c = 1,
                 ribo_c = 2,
                 wga_c = None,
                 phase_proj=True
                 ):
        self.label = None
        self.noise_ribo = None
        self.noise_dapi = None
        self.noise_phase = None
        self.prop_df = None
        self.optimal_layer_mat = None
        self.masks = None
        self.pred = None
        self.phase_projection = None
        self.cy3 = None
        self.pixel_value = None
        self.path = path
        self.matrix = nd2.imread(self.path)
        self.dapi = np.max(self.matrix[:, dapi_c, :, :], axis=0)
        self.ribo = np.max(self.matrix[:, ribo_c, :, :], axis=0)
        if wga_c:
            self.wga = np.max(self.matrix[:, wga_c, :, :], axis=0)
        if phase_proj:
            self.phase_projection, self.optimal_layer_mat = gt(self.matrix, chunk_size=128 * 2, is_show_proj=False)

    def segment(self, model=r'C:\Users\yedidyab\Box\Zohar_Yedidya\111022_Psyringea_growth1\croped\models\100X_model'):
        '''

        :return: segmentation using trained model
        '''
        self.pred = get_segmentation(model=model, img=np.array([self.phase_projection, self.dapi]))
        self.masks = self.pred[0]

        # create df which contain cells properties
        self.prop_df = pd.DataFrame(index=np.unique(self.masks))
        self.prop_df['label'] = np.unique(self.masks)

        # add array of noise in order to randomise backgrounds
        self.noise_dapi = self.dapi[self.masks == 0]
        self.noise_ribo = self.ribo[self.masks == 0]
        self.noise_phase = self.phase_projection[self.masks == 0]

    def napari_show(self, show_mask=True):
        '''
        using napari to show all layey of the object
        :param show_mask:
        :return:
        '''
        viewer = napari.Viewer()
        phase_l = viewer.add_image(self.phase_projection, name='phase')
        if show_mask:
            masks_l = viewer.add_image(self.masks, name='masks')
        dapi_l = viewer.add_image(self.dapi, name='dapi')
        ribo_l = viewer.add_image(self.ribo, name='ribo')
        viewer.show(block=True)  # wait until viewer window closes

    def calc_dapi_ribo(self):
        '''
        calculate mean and median value of Dapi and ribo per cell
        '''
        dapi_max = list()
        ribo_max = list()
        dapi_med = list()
        ribo_med = list()
        dapi_20 = []
        ribo_20 = []
        dapi_sum = []
        ribo_sum = []
        dapi_cv = []
        ribo_cv = []
        cv = lambda x: np.std(x) / np.mean(x)

        for i in np.unique(self.masks)[1:]:
            dapi_mask = self.dapi[self.masks == i]
            ribo_mask = self.ribo[self.masks == i]
            dapi_max.append(np.max(dapi_mask))
            dapi_med.append(np.median(dapi_mask))
            dapi_20.append(np.percentile(a=dapi_mask, q=20))
            ribo_max.append(np.max(ribo_mask))
            ribo_med.append(np.median(ribo_mask))
            ribo_20.append(np.percentile(a=ribo_mask, q=20))
            dapi_sum.append(np.sum(dapi_mask))
            ribo_sum.append(np.sum(ribo_mask))
            dapi_cv.append(cv(dapi_mask))
            ribo_cv.append(cv(ribo_mask))



        self.prop_df['dapi_max'] = dapi_max
        self.prop_df['ribo_max'] = ribo_max        
        self.prop_df['dapi_sum'] = dapi_sum
        self.prop_df['ribo_sum'] = ribo_sum
        self.prop_df['dapi_med'] = dapi_med
        self.prop_df['ribo_med'] = ribo_med
        self.prop_df['dapi_20'] = dapi_20
        self.prop_df['ribo_20'] = ribo_20
        self.prop_df['dapi_fc'] = self.prop_df.dapi_20 / self.prop_df.dapi_max
        self.prop_df['ribo_fc'] = self.prop_df.ribo_20 / self.prop_df.ribo_max
        self.prop_df['dapi_cv'] = dapi_cv
        self.prop_df['ribo_cv'] = ribo_cv

    def add_regionprops_data(self):
        '''
        build properties df from label matrix, return df.
        '''

        props = measure.regionprops_table(self.masks, properties=['label',
                                                                  'area',
                                                                  'axis_major_length',
                                                                  'axis_minor_length',
                                                                  'centroid',
                                                                  'extent',
                                                                  'orientation',
                                                                  'eccentricity',
                                                                  'equivalent_diameter_area',
                                                                  'feret_diameter_max',
                                                                  'perimeter',
                                                                  'bbox',
                                                                  'solidity'])
        props_data = pd.DataFrame(props)
        self.prop_df = self.prop_df.merge(props_data, on='label')
        self.prop_df['orientation_d'] = np.rad2deg(self.prop_df.orientation)
        self.prop_df['minor_major'] = self.prop_df.axis_minor_length / self.prop_df.axis_major_length

    def plot_by_cell_idx(self, cell_idx, dapi=False, ribo=False, masks=False, wga=False, space=0):
        '''

        :param space: expend the mat by [pspace
        :param cell_idx: cell index
        :param dapi: show dapi?
        :param ribo: show ribo?
        :param masks: show mask?
        :return: matrix that cut with bbox
        '''

        min_r = self.prop_df['bbox-0'][cell_idx - 1] - space
        min_c = self.prop_df['bbox-1'][cell_idx - 1] - space
        max_r = self.prop_df['bbox-2'][cell_idx - 1] + space
        max_c = self.prop_df['bbox-3'][cell_idx - 1] + space
        # print(self.prop_df.label.loc[cell_idx - 1])
        # print(cell_idx)

        # select relevant pixels
        if dapi:
            mat = self.dapi[min_r: max_r, min_c: max_c]
        elif ribo:
            mat = self.ribo[min_r: max_r, min_c: max_c]
        elif masks:
            mat = self.masks[min_r: max_r, min_c: max_c]
        elif wga:
            mat = self.wga[min_r: max_r, min_c: max_c]
        else:
            mat = self.phase_projection[min_r: max_r, min_c: max_c]

        return mat

    def cut_cells(self, random_idx, dapi=False, ribo=False, masks=False, wga=False, space=0):
        '''
        :param ribo:
        :param dapi:
        :param self: Img object
        :param random_idx: some index
        :return: plot cell with idx, rotate to vertical ad cut unrelevant pixels.
        '''
        # take the cell box
        mat = self.plot_by_cell_idx(random_idx, dapi=dapi, ribo=ribo, masks=masks, wga=wga, space=0)
        # rotate by angel*(-1) to vertical
        ang = -1 * self.prop_df.orientation_d[random_idx - 1]
        rotated_mat = rotate(mat, angle=ang)
        # cut the rotated matrix by [(row_axis-minor_axis)/2 : (row_axis+minor_axis)/2]
        minor_l = self.prop_df.axis_minor_length[random_idx - 1]
        col = rotated_mat.shape[1]
        d = int((col - minor_l) / 2)
        t = rotated_mat[:, d:col - d]
        return t

    def plot_by_channel_idx(self, idx_array, n_rows=2, n_cols=3, dapi=False, ribo=False, masks=False, add_noise=True,
                            to_show=True):
        # crate l_mat & l_titles
        l_mat = []
        l_titel = []

        # loop over idx_array, cut the cells and add noise
        for i in idx_array:
            # cut the cell and rotate
            m_temp = self.cut_cells(i, dapi=dapi, ribo=ribo, masks=False)
            # we want to calculate morans I with the minimal noise pixel, so i save the matrix before expend
            for_moran = m_temp
            m_temp = extend_mat(m_temp)

            if add_noise:
                if dapi:
                    noise_arr = self.noise_dapi
                elif ribo:
                    noise_arr = self.noise_ribo
                else:
                    noise_arr = self.noise_phase
                m_temp = random_noise(noise_arr, m_temp)
                for_moran = random_noise(noise_arr, for_moran)

            l_mat.append(m_temp)
            l_titel.append(f'idx={i}, moran{"{:.2f}".format(moran(for_moran))}')

        plot_multiple_idx(l_mat, l_titel, n_rows, n_cols)



    def calc_moran(self, idx, extend=False, add_noise=True, sigma=1):
        m_phase = self.cut_cells(idx)
        m_dapi = self.cut_cells(idx, dapi=True)
        m_ribo = self.cut_cells(idx, ribo=True)

        if extend:
            m_phase = extend_mat(m_phase)
            m_dapi = extend_mat(m_dapi)
            m_ribo = extend_mat(m_ribo)

        if add_noise:
            m_phase = random_noise(self.noise_phase, m_phase, sigma=sigma)
            m_dapi = random_noise(self.noise_dapi, m_dapi, sigma=sigma)
            m_ribo = random_noise(self.noise_ribo, m_ribo, sigma=sigma)

        return [moran(mat) for mat in [m_phase, m_dapi, m_ribo]]

        # if to_show:
        #     plt.imshow(m_temp)
        #     plt.imshow()

    def add_label_layer(self, mix_masks=False, tow_labels=False):

        if mix_masks:
            # for better visualisation of the labels in napari, repalse the label value to random colors
            k = np.arange(np.max(self.masks))
            v = np.arange(np.max(self.masks))
            np.random.shuffle(v)

            out = np.zeros_like(self.masks)

            for key, val in zip(k, v):
                out[self.masks == key] = val

        # mark manualy
        viewer = napari.Viewer()
        phase = viewer.add_image(self.phase_projection, name='phase')
        if mix_masks:
            out = viewer.add_image(out, name='mix_masks')
        masks = viewer.add_image(self.masks, name='masks')
        label = viewer.add_labels(
            data=np.zeros([self.phase_projection.shape[0], self.phase_projection.shape[1]]).astype(int), name='label')
        if tow_labels:
            label2 = viewer.add_labels(
                data=np.zeros([self.phase_projection.shape[0], self.phase_projection.shape[1]]).astype(int),
                name='label2')

        viewer.show(block=True)  # wait until viewer window closes

        self.label = label.data
        if tow_labels:
            self.label2 = label2.data



def moran(Z):
    # Use your matrix here, instead of this random one
    # Create the matrix of weigthts
    w = lat2W(Z.shape[0], Z.shape[1])

    # Crate the pysal Moran object
    mi = Moran(Z, w)
    # Verify Moran's I results
    return mi.I
    # print(mi.p_norm)


def plot_multiple_idx(mat_list, titles_list, n_rows=2, n_columns=3):
    fig = plt.figure(figsize=(8, 8))
    for i, mat, title in zip(range(1, n_rows * n_columns + 1), mat_list, titles_list):
        # print(type(mat))
        fig.add_subplot(n_rows, n_columns, i)
        plt.imshow(mat)
        plt.title(title)
    plt.show()


def random_noise(noise_arr, to_fill_arr, sigma=3):
    rand_arr = np.random.choice(noise_arr, size=to_fill_arr.shape)
    blur = gaussian_filter(rand_arr, sigma=sigma)
    to_fill_arr[to_fill_arr == 0] = blur[to_fill_arr == 0]
    return to_fill_arr


def random_index(self, size=6):
    idx = np.random.choice(np.unique(self.masks), size=size)
    return idx


def extend_mat(mat, size=(40, 12)):
    z = np.zeros(size)
    m_row, m_col = mat.shape
    z_row, z_col = z.shape
    d_row = (z_row - m_row) / 2
    d_col = (z_col - m_col) / 2
    z[int(d_row):int(z_row - d_row), int(d_col):int(z_col - d_col)] = mat
    return z