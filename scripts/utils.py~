import copy

import numpy as np
import napari
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import distance

mpl.rcParams['figure.dpi'] = 300
from cellpose import io
from cellpose import models, plot
from skimage import morphology, measure
import seaborn as sns
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import math
import statistics
import pickle
import glob
import nd2
from phase_progection import get_phase_projection as gt
import os
from alignment import align
from reduce_high_signals import reduce


class Img:

    def __init__(self, path, temp_files_path = r'X:\yedidyab\dl_project\temp_files'):
        self.dev_list = None
        self.path = path
        self.name = self.path.split('.')[0].split('/')[-1]
        self.matrix = nd2.imread(self.path)
        self.phase_projection, self.optimal_layer_mat = gt(self.matrix, chunk_size=128 * 2, is_show_proj=False)
        self.dapi = np.max(self.matrix[:, 1, :, :], axis=0)
        self.ribo = np.max(self.matrix[:, 4, :, :], axis=0)
        self.wga = np.max(self.matrix[:, 3, :, :], axis=0)
        self.channels = [self.phase_projection, self.dapi, self.ribo, self.wga]
        self.size = self.dapi.shape[0]
        self.temp_files_path = temp_files_path

    def segment(self, model=r'C:\Users\yedidyab\Documents\zohar_data\111022_Psyringea_growth1\croped\models\100X_model',
                diameter=0, flow_threshold=0.4, cellprob_threshold=0, load_if_exist=True):
        '''

        :param model: segmentation model path
        :param diameter: cellpose param
        :param flow_threshold: cellpose param
        :param cellprob_threshold: cellpose param
        :param load_if_exist: load seg file from path/../segment/*.npy file
        :return: save the mask in path/../segment/, save masks and outline in the object.
        '''
        save_dir = f'{self.temp_files_path}/segment'
        save_path = f'{self.temp_files_path}/segment/{self.name}'
        if not os.path.exists(save_path + '_seg.npy') or load_if_exist == False:
            print(f'segment {self.name}')
            model = models.CellposeModel(gpu=True,
                                         pretrained_model=model)

            # use model diameter if user diameter is 0
            diameter = model.diam_labels if diameter == 0 else diameter
            img = np.array([self.phase_projection, self.dapi])
            self.pred = model.eval(img,
                                   channels=[1, 2],
                                   diameter=diameter,
                                   flow_threshold=flow_threshold,
                                   cellprob_threshold=cellprob_threshold
                                   )
            self.masks = self.pred[0]
            masks, flows, styles = self.pred

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            io.masks_flows_to_seg(img, masks, flows, diameter, save_path)

        loader = np.load(save_path + '_seg.npy', allow_pickle=True).item()
        self.masks, self.masks_outlines = loader['masks'], loader['outlines']

    def alighnment(self):
        self.channels = [align(self.phase_projection, channel) for channel in self.channels[1:]]

    def reduce_high_signals(self):
        self.channels = [reduce(channel) for channel in self.channels]

    def show_img(self):
        '''

        show phase, dapi and outline in napari
        '''
        viewer = napari.Viewer()
        viewer.add_image(self.phase_projection)
        viewer.add_image(self.dapi, name='dapi')
        viewer.add_labels(self.masks_outlines)
        viewer.show(block=True)  # wait until viewer window closes

    def build_all_props_df(self):
        '''
        build properties df from label matrix, return df.
        '''

        pros = measure.regionprops_table(self.masks, properties=['label',
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
                                                                 'solidity',
                                                                 'bbox'])
        props_data = pd.DataFrame(pros)
        props_data.index = props_data.label
        self.props_data = props_data

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

        self.props_data['dapi_max'] = dapi_max
        self.props_data['ribo_max'] = ribo_max
        self.props_data['dapi_sum'] = dapi_sum
        self.props_data['ribo_sum'] = ribo_sum
        self.props_data['dapi_med'] = dapi_med
        self.props_data['ribo_med'] = ribo_med
        self.props_data['dapi_20'] = dapi_20
        self.props_data['ribo_20'] = ribo_20
        self.props_data['dapi_fc'] = self.props_data.dapi_20 / self.props_data.dapi_max
        self.props_data['ribo_fc'] = self.props_data.ribo_20 / self.props_data.ribo_max
        self.props_data['dapi_cv'] = dapi_cv
        self.props_data['ribo_cv'] = ribo_cv

    def create_candidate_pairs(self, chunk_size=64):
        img_size = self.size
        all_comb = []
        # loop over the image with window_size = 64 (we miss the pairs that divide exactly in the border. we can add overlap after.
        for i in range(0, img_size - 1, chunk_size):
            for j in range(0, img_size - 1, chunk_size):
                # take all combination in window. if we use overlap we need to delete duplicates.
                cells_in = np.unique(self.masks[i:i + chunk_size, j:j + chunk_size])
                cells_in = cells_in[cells_in != 0]
                all_comb.extend(list(itertools.combinations(cells_in, 2)))
        self.all_comb = all_comb

    def predict_division(self,
                         load_if_exist=True,
                         rf_model=r'C:\Users\yedidyab\Documents\zohar_data\sampels\new_rf_label_again_after_search_942acc.pkl'):

        # Set the save directory and file path for the properties dataframe
        save_dir = f'{self.temp_files_path}/props_df'
        save_path = f'{self.temp_files_path}/props_df/{self.name}_df_pairs.csv'

        # If the properties dataframe does not exist or if load_if_exist is set to False,
        # create the properties dataframe and save it to the specified file path.
        if not os.path.exists(save_path) or load_if_exist == False:

            # If the masks attribute has not been set, call the segment method to generate masks.
            if not hasattr(self, 'masks'):
                self.segment()
            # If the props_data attribute has not been set, call the build_all_props_df method to generate the properties dataframe.
            if not hasattr(self, 'props_data'):
                self.build_all_props_df()

            # If the ribo_max column is not present in the properties dataframe, call the calc_dapi_ribo method to calculate it.
            if 'ribo_max' not in self.props_data.columns:
                self.calc_dapi_ribo()

            # Create the candidate pairs from the masks and properties data.
            self.create_candidate_pairs()
            # Generate a list of all the properties for each pair of objects.
            prop_all = create_pairs_all_props_list(self.props_data, self.all_comb, 'na', output_list=[])
            # Create a dataframe of all the properties for each pair of objects.
            self.prop_df = create_all_props_df(prop_all)
            # If the save directory does not exist, create it.
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # Save the properties dataframe to the specified file path.
            self.prop_df.to_csv(save_path)
            self.props_data.to_csv(f'{self.temp_files_path}/props_df/{self.name}_df_single.csv')

        # If the properties dataframe already exists and load_if_exist is set to True, load the dataframe from the
        # specified file path.
        else:
            self.prop_df = pd.read_csv(save_path)
            self.props_data = pd.read_csv(f'{self.temp_files_path}/props_df/{self.name}_df_single.csv')
        # Open the specified model file and load the model into the rf_model attribute.
        file = open(rf_model, "rb")
        self.rf_model = pickle.load(file)
        # Get the list of features used by the model.
        features = self.rf_model.feature_names_in_
        # Use the model to make predictions on the properties dataframe and store the predictions in a new column.
        self.prop_df['prediction'] = self.rf_model.predict(self.prop_df[features])
        # correct prediction
        th = 0.3
        self.prop_df.loc[abs(self.prop_df.orientation) < 1 - th, 'prediction'] = 0
        self.prop_df.loc[abs(self.prop_df.orientation) > 1 + th, 'prediction'] = 0
        th_or = 0.5
        self.prop_df.loc[self.prop_df.orientation_line < 0 - th, 'prediction'] = 0
        self.prop_df.loc[self.prop_df.orientation_line > 0 + th, 'prediction'] = 0
        # Add a new column to store the prediction probabilities.
        self.prop_df['prob'] = None
        # Use the model to calculate the prediction probabilities for each row in the properties dataframe and store them in the 'prob' column.
        self.prop_df['prob'] = self.rf_model.predict_proba(self.prop_df[features])

    def replace_values_in_mask(self):
        """
        This function takes in a 2D mask and a list of tuples containing the (value1, value2) that should be replaced in the mask.
        This function will iterate over all the elements of the mask and check if any of the elements match with the value2 of tuple.
        If yes, then it will replace that element with the value1 of that tuple.
        """
        for val1, val2 in self.prop_df[self.prop_df.prediction == 1].idx.str.split('&').apply(lambda x: [int(i) for i in x]):
            self.masks[self.masks == val2] = val1

    def extract_single_cell_images(self, output_size=100, corner_distance=100):
        save_dir = f'{self.temp_files_path}/../single_cell_data'
        pros = measure.regionprops_table(self.masks, properties=['label',
                                                                 'bbox'])
        props_data = pd.DataFrame(pros)
        props_data.index = props_data.label

        for cell in props_data.iterrows():
            label = cell[1]['label']
            min_row, min_col, max_row, max_col = cell[1]['bbox-0'], cell[1]['bbox-1'], cell[1]['bbox-2'], cell[1][
                'bbox-3']
            # Calculate the top-left and bottom-right coordinates of the bounding box
            top_left = (max(0, min_col - (output_size - (max_col - min_col)) // 2),
                        max(0, min_row - (output_size - (max_row - min_row)) // 2))
            bottom_right = (min(self.masks.shape[1], top_left[0] + output_size),
                            min(self.masks.shape[0], top_left[1] + output_size))

            #filter border cells
            if top_left[0] < corner_distance or top_left[1] < corner_distance or bottom_right[0] > self.masks.shape[
                1] - corner_distance or bottom_right[1] > self.masks.shape[0] - corner_distance:
                continue

            # Extract the bounding box from the image and mask
            bounding_box_img = self.phase_projection[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            bounding_box_mask = self.masks[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            bounding_box_dapi = self.dapi[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            bounding_box_ribo = self.ribo[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            bounding_box_wga = self.wga[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            bb_mask = np.zeros_like(bounding_box_mask)
            bb_mask[bounding_box_mask == label] = 1

            # Combine the image and mask into a single array
            bounding_box = np.array([bounding_box_img, bb_mask, bounding_box_dapi, bounding_box_ribo, bounding_box_wga])
            if not os.path.exists(f'{save_dir}/{self.name}'):
                os.makedirs(f'{save_dir}/{self.name}')
            np.save(file=f'{save_dir}/{self.name}/label_{label}_bb_{min_col}_{max_row}.npy', arr=bounding_box)


def plot_by_cells_idx(img, cell1, cell2, space=0):
    '''
    for QA
    '''
    bbox1_0 = img.props_data.loc[cell1 - 1]['bbox-0']
    bbox1_1 = img.props_data.loc[cell1 - 1]['bbox-1']
    bbox1_2 = img.props_data.loc[cell1 - 1]['bbox-2']
    bbox1_3 = img.props_data.loc[cell1 - 1]['bbox-3']
    bbox2_0 = img.props_data.loc[cell2 - 1]['bbox-0']
    bbox2_1 = img.props_data.loc[cell2 - 1]['bbox-1']
    bbox2_2 = img.props_data.loc[cell2 - 1]['bbox-2']
    bbox2_3 = img.props_data.loc[cell2 - 1]['bbox-3']

    max_r = max(bbox1_2, bbox2_2).astype(int)
    min_r = min(bbox1_0, bbox2_0).astype(int)
    max_c = max(bbox1_3, bbox2_3).astype(int)
    min_c = min(bbox1_1, bbox2_1).astype(int)
    print(max_r, max_c, min_r, min_c)
    mat = img.masks_outlines[min_r - space: max_r + space, min_c - space: max_c + space]
    plt.imshow(mat)
    return mat


def calc_major_pairs(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    min_t = (min(x1_min, x2_min), (min(y1_min, y2_min)))
    max_t = (max(x1_max, x2_max), (max(y1_max, y2_max)))
    dist = distance.euclidean(min_t, max_t)
    return dist

def is_overlap_1d(x1_max, x1_min, x2_max, x2_min):
    if x1_max >= x2_min and x2_max >= x1_min:
        return True


def is_overlap_2d(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    if (is_overlap_1d(x1_max, x1_min, x2_max, x2_min) and is_overlap_1d(y1_max, y1_min, y2_max, y2_min)):
        return True
    else:
        return False

def create_pairs_all_props_list(props_data, pairs_list, label, output_list):
    '''
    ######################### TEST ###################
    input:
    props_data - see func above
    pairs_list - list of pairs
    label - label for the pairs (pair/not_pair)
    output_list - list to append

    output:
    list of tuples with properies for each pair.
    '''
    prop_l = output_list
    for pair in pairs_list:
        idx_p1 = pair[0]
        idx_p2 = pair[1]
        # orientaion
        p1_orientation = props_data.loc[idx_p1].orientation
        p2_orientation = props_data.loc[idx_p2].orientation

        orientation = p1_orientation / p2_orientation

        # to avoid inf
        if orientation > 100 or orientation < -100:
            orientation = 100

        # axis_major_length/axis_minor_lengt
        p1_axis_major_length = props_data.loc[idx_p1].axis_major_length
        p2_axis_major_length = props_data.loc[idx_p2].axis_major_length

        p1_axis_minor_length = props_data.loc[idx_p1].axis_minor_length
        p2_axis_minor_length = props_data.loc[idx_p2].axis_minor_length

        p1_minor_magor = p1_axis_major_length / p1_axis_minor_length
        p2_minor_magor = p2_axis_major_length / p2_axis_minor_length

        minor_magor = p1_minor_magor / p2_minor_magor

        # extent
        p1_extent = props_data.loc[idx_p1].extent
        p2_extent = props_data.loc[idx_p2].extent

        extent = p1_extent / p2_extent

        # centroid-x
        p1_centroid_0 = props_data.loc[idx_p1]['centroid-0']
        p2_centroid_0 = props_data.loc[idx_p2]['centroid-0']

        # centroid-y
        p1_centroid_1 = props_data.loc[idx_p1]['centroid-1']
        p2_centroid_1 = props_data.loc[idx_p2]['centroid-1']

        distance = math.dist([p1_centroid_0, p1_centroid_1], [p2_centroid_0, p2_centroid_1])

        slope = (p1_centroid_1 - p2_centroid_1) / (p1_centroid_0 - p2_centroid_0)
        orientation_line = math.atan(slope) - p1_orientation
        # orientation_line = slope

        # eccentricity
        p1_eccentricity = props_data.loc[idx_p1].eccentricity
        p2_eccentricity = props_data.loc[idx_p2].eccentricity

        eccentricity = p1_eccentricity / p2_eccentricity

        # equivalent_diameter_area
        p1_equivalent_diameter_area = props_data.loc[idx_p1].equivalent_diameter_area
        p2_equivalent_diameter_area = props_data.loc[idx_p2].equivalent_diameter_area

        equivalent_diameter_area = p1_equivalent_diameter_area / p2_equivalent_diameter_area

        # feret_diameter_max
        p1_feret_diameter_max = props_data.loc[idx_p1].feret_diameter_max
        p2_feret_diameter_max = props_data.loc[idx_p2].feret_diameter_max

        feret_diameter_max = p1_feret_diameter_max / p2_feret_diameter_max

        # perimeter
        p1_perimeter = props_data.loc[idx_p1].perimeter
        p2_perimeter = props_data.loc[idx_p2].perimeter

        perimeter = p1_perimeter / p2_perimeter

        # solidity
        p1_solidity = props_data.loc[idx_p1].solidity
        p2_solidity = props_data.loc[idx_p2].solidity

        solidity = p1_solidity / p2_solidity

        # dapi_max
        p1_dapi_max = props_data.loc[idx_p1].dapi_max
        p2_dapi_max = props_data.loc[idx_p2].dapi_max

        dapi_max = p1_dapi_max / p2_dapi_max

        # ribo_max
        p1_ribo_max = props_data.loc[idx_p1].ribo_max
        p2_ribo_max = props_data.loc[idx_p2].ribo_max

        ribo_max = p1_ribo_max / p2_ribo_max
        # dapi_sum
        p1_dapi_sum = props_data.loc[idx_p1].dapi_sum
        p2_dapi_sum = props_data.loc[idx_p2].dapi_sum

        dapi_sum = p1_dapi_sum / p2_dapi_sum
        # ribo_sum
        p1_ribo_sum = props_data.loc[idx_p1].ribo_sum
        p2_ribo_sum = props_data.loc[idx_p2].ribo_sum

        ribo_sum = p1_ribo_sum / p2_ribo_sum
        # dapi_med
        p1_dapi_med = props_data.loc[idx_p1].dapi_med
        p2_dapi_med = props_data.loc[idx_p2].dapi_med

        dapi_med = p1_dapi_med / p2_dapi_med
        # ribo_med
        p1_ribo_med = props_data.loc[idx_p1].ribo_med
        p2_ribo_med = props_data.loc[idx_p2].ribo_med

        ribo_med = p1_ribo_med / p2_ribo_med
        # dapi_20
        p1_dapi_20 = props_data.loc[idx_p1].dapi_20
        p2_dapi_20 = props_data.loc[idx_p2].dapi_20

        dapi_20 = p1_dapi_20 / p2_dapi_20
        # ribo_20
        p1_ribo_20 = props_data.loc[idx_p1].ribo_20
        p2_ribo_20 = props_data.loc[idx_p2].ribo_20

        ribo_20 = p1_ribo_20 / p2_ribo_20
        # dapi_fc
        p1_dapi_fc = props_data.loc[idx_p1].dapi_fc
        p2_dapi_fc = props_data.loc[idx_p2].dapi_fc

        dapi_fc = p1_dapi_fc / p2_dapi_fc
        # ribo_fc
        p1_ribo_fc = props_data.loc[idx_p1].ribo_fc
        p2_ribo_fc = props_data.loc[idx_p2].ribo_fc

        ribo_fc = p1_ribo_fc / p2_ribo_fc
        # dapi_cv
        p1_dapi_cv = props_data.loc[idx_p1].dapi_cv
        p2_dapi_cv = props_data.loc[idx_p2].dapi_cv

        dapi_cv = p1_dapi_cv / p2_dapi_cv
        # ribo_cv
        p1_ribo_cv = props_data.loc[idx_p1].ribo_cv
        p2_ribo_cv = props_data.loc[idx_p2].ribo_cv

        ribo_cv = p1_ribo_cv / p2_ribo_cv

        # is_overlap
        p1_bbox = props_data[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].loc[idx_p1]
        p2_bbox = props_data[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].loc[idx_p2]

        is_overlap = is_overlap_2d(p1_bbox, p2_bbox)

        # calc_major_pairs
        major_pairs = calc_major_pairs(p1_bbox, p2_bbox)

        # major_axis and centroids variance
        major_dist_var = statistics.stdev([distance, p1_axis_major_length, p2_axis_major_length])

        prop_l.append((str(idx_p1) + '&' + str(idx_p2),
                       orientation,
                       minor_magor,
                       extent,
                       distance,
                       orientation_line,
                       eccentricity,
                       equivalent_diameter_area,
                       feret_diameter_max,
                       perimeter,
                       solidity,
                       is_overlap,
                       major_pairs,
                       major_dist_var,
                       dapi_max,
                       ribo_max,
                       dapi_sum,
                       ribo_sum,
                       dapi_med,
                       ribo_med,
                       dapi_20,
                       ribo_20,
                       dapi_fc,
                       ribo_fc,
                       dapi_cv,
                       ribo_cv,
                       label))
    return prop_l


def create_all_props_df(prop_l):
    '''
    convert list of tuples to df
    '''
    prop_df = pd.DataFrame(prop_l,
                           columns=['idx',
                                    'orientation',
                                    'minor_magor',
                                    'extent',
                                    'distance',
                                    'orientation_line',
                                    'eccentricity',
                                    'equivalent_diameter_area',
                                    'feret_diameter_max',
                                    'perimeter',
                                    'solidity',
                                    'is_overlap',
                                    'major_pairs',
                                    'major_dist_var',
                                    'dapi_max',
                                    'ribo_max',
                                    'dapi_sum',
                                    'ribo_sum',
                                    'dapi_med',
                                    'ribo_med',
                                    'dapi_20',
                                    'ribo_20',
                                    'dapi_fc',
                                    'ribo_fc',
                                    'dapi_cv',
                                    'ribo_cv',
                                    'value']
                           )
    return prop_df


def plot_confusion_mat(y_test, predicted):
    '''
    I use this finction to evaluate the model preformence
    '''

    results = confusion_matrix(y_test, predicted)
    strings = strings = np.asarray([['TN = ', 'FP = '],
                                    ['FN = ', 'TP = ']])

    labels = (np.asarray(["{0} {1:.3f}".format(string, value)
                          for string, value in zip(strings.flatten(),
                                                   results.flatten())])
              ).reshape(2, 2)

    fig, ax = plt.subplots()
    ax = sns.heatmap(results, annot=labels, fmt="", ax=ax,
                     xticklabels=['not in division', 'in devision'],
                     yticklabels=['not in division', 'in devision'])
    ax.set_xlabel('True labels', fontsize=10)
    ax.set_ylabel('predicted labels', fontsize=10)

    plt.show()


def random_forest(X_train, X_test, y_train, y_test, save_path=None, load_path=None):
    # build random forest model
    if load_path:
        file = open(load_path, "rb")
        rf = pickle.load(file)
    else:
        rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42, max_depth=6, min_samples_leaf=20)
        # fit
        rf.fit(X_train, y_train)
    print(rf.get_params(deep=True))

    predicted = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print(f'Mean accuracy score: {accuracy:.3}')
    plot_confusion_mat(y_test, predicted)
    if save_path:
        pickle.dump(rf, open(save_path, 'wb'))
    return rf


def load_img(filepath, model):
    print(filepath)
    temp = Img(filepath)
    temp.segment(model=model)
    temp.add_label_layer(load_if_exist=True, ask_for_labeling=False)
    temp.build_all_props_df()
    temp.calc_dapi_ribo()
    print(44)
    prop_pairs = create_pairs_all_props_list(temp.props_data, temp.dev_list, 'pairs', output_list=[])
    prop_all = create_pairs_all_props_list(temp.props_data, temp.not_dev_list, 'not_pairs', output_list=prop_pairs)
    prop_train = create_all_props_df(prop_all)
    prop_train['image'] = temp.name
    return prop_train


def create_train_df(train_path, model):
    all_df = []
    for filepath in glob.iglob(train_path):
        df = load_img(filepath, model)
        all_df.append(df)
    train_df = pd.concat(all_df)
    train_df.reset_index(inplace=True)
    return train_df


def train():
    # test_script
    model = r'C:\Users\yedidyab\Documents\zohar_data\111022_Psyringea_growth1\croped\models\100X_model'
    train_path = r'C:\Users\yedidyab\Documents\zohar_data\sampels/*.nd2'

    all_df = []
    for filepath in glob.iglob(train_path):
        print(1)
        df = load_img(filepath, model)
        all_df.append(df)
    train_df = pd.concat(all_df)
    train_df.reset_index(inplace=True)
    train_df.to_csv(r'C:\Users\yedidyab\Documents\zohar_data\sampels\train_df_with_line_or.csv')
    features = ['orientation',
                'minor_magor',
                'extent',
                'distance',
                'eccentricity',
                'equivalent_diameter_area',
                'feret_diameter_max',
                'perimeter',
                'solidity',
                'is_overlap',
                'major_dist_var',
                'dapi_max',
                'ribo_max',
                'dapi_sum',
                'ribo_sum',
                'dapi_med',
                'ribo_med',
                'dapi_20',
                'ribo_20',
                'dapi_fc',
                'ribo_fc',
                'dapi_cv',
                'ribo_cv'
                ]

    # split the data(80%) to train and test(20%)
    X_train, X_test, y_train, y_test = train_test_split(
        train_df[features],
        train_df.value,
        test_size=0.2,
        random_state=123456)

    rf_cls = random_forest(X_train, X_test, y_train, y_test,
                           save_path=r'C:\Users\yedidyab\Documents\zohar_data\sampels\rf_model.pkl')


def predict(path,
            seg_model=r'C:\Users\yedidyab\Documents\zohar_data\111022_Psyringea_growth1\croped\models\100X_model',
            rf_model=r'C:\Users\yedidyab\Documents\zohar_data\sampels\new_rf_label_again_942acc_all_featurs.pkl'):
    print(path)
    features = ['orientation',
                'minor_magor',
                'extent',
                'distance',
                'eccentricity',
                'equivalent_diameter_area',
                'feret_diameter_max',
                'perimeter',
                'solidity',
                'is_overlap',
                'major_dist_var',
                'dapi_max',
                'ribo_max',
                'dapi_sum',
                'ribo_sum',
                'dapi_med',
                'ribo_med',
                'dapi_20',
                'ribo_20',
                'dapi_fc',
                'ribo_fc',
                'dapi_cv',
                'ribo_cv'
                ]
    temp = Img(path)
    temp.segment(model=seg_model)
    temp.add_label_layer(load_if_exist=True, ask_for_labeling=False)
    temp.build_all_props_df()
    temp.calc_dapi_ribo()
    temp.create_candidate_pairs()
    prop_all = create_pairs_all_props_list(temp.props_data, temp.all_comb, 'na', output_list=[])
    prop_df = create_all_props_df(prop_all)
    file = open(rf_model, "rb")
    rf_cls = pickle.load(file)
    prop_df['prediction'] = rf_cls.predict(prop_df[features])
    prop_df['prob'] = None
    prop_df['prob'] = rf_cls.predict_proba(prop_df[features])
    return temp, prop_df


def show(test, prop_df):
    # run  - show(*predict(r'C:\Users\yedidyab\Documents\zohar_data\141122_Psyringea_growth3\B_010.nd2'))
    # def plot_prediction
    img = test.phase_projection
    masks = test.masks
    pred_df = prop_df

    dev = pred_df[pred_df.prediction == 'pairs'].idx.str.split('&').explode().unique().astype(int)
    no = pred_df[pred_df.prediction != 'pairs'].idx.str.split('&').explode().unique().astype(int)
    no = no[np.isin(no, dev, invert=True)]
    d = np.where(np.isin(masks, dev), 1, 0)
    n = np.where(np.isin(masks, no), 2, 0)
    pred_df.sort_values(by='prob', ignore_index=True, inplace=True, ascending=False)
    done = []
    prob = np.ones_like(masks, dtype=float)
    for n, (i, j) in enumerate(pred_df.idx.str.split('&')):
        pred = pred_df.prob[n]
        prob[masks == int(i)] = pred
        prob[masks == int(j)] = pred
    viewer = napari.Viewer(title=test.name)
    viewer.add_image(img, name='image')
    viewer.add_image(prob, name='prob')
    viewer.add_labels(test.masks_outlines, name='masks_outlines')

    viewer.add_labels(d, name='dev')


def predict_all_imgs(path,
                     seg_model=r'C:\Users\yedidyab\Documents\zohar_data\111022_Psyringea_growth1\croped\models\100X_model',
                     rf_model=r'C:\Users\yedidyab\Documents\zohar_data\sampels\rf_model.pkl'):
    print(path)
    file = open(rf_model, "rb")
    rf_cls = pickle.load(file)
    features = ['orientation',
                'minor_magor',
                'extent',
                'distance',
                'eccentricity',
                'equivalent_diameter_area',
                'feret_diameter_max',
                'perimeter',
                'solidity',
                'is_overlap',
                'major_dist_var',
                'dapi_max',
                'ribo_max',
                'dapi_sum',
                'ribo_sum',
                'dapi_med',
                'ribo_med',
                'dapi_20',
                'ribo_20',
                'dapi_fc',
                'ribo_fc',
                'dapi_cv',
                'ribo_cv'
                ]
    all_df = []
    for file in glob.iglob(path):
        name = file.split('\\')[-1].split('.')[0]
        print(name)
        print(1)
        temp = Img(file)
        temp.segment(model=seg_model)
        temp.add_label_layer(load_if_exist=True, ask_for_labeling=False)
        temp.build_all_props_df()
        # temp.calc_dapi_ribo()
        temp.create_candidate_pairs()
        prop_all = create_pairs_all_props_list(temp.props_data, temp.all_comb, 'na', output_list=[])
        prop_df = create_all_props_df(prop_all)
        prop_df['name'] = name
        prop_df['prediction'] = rf_cls.predict(prop_df[features])
        prop_df['prob'] = None
        prop_df['prob'] = rf_cls.predict_proba(prop_df[features])
        all_df.append(prop_df)
    df = pd.concat(all_df)
    df.to_csv(fr'{path}/../df.csv')
    return temp, prop_df

# dirs = [r'C:\Users\yedidyab\Documents\zohar_data\111022_Psyringea_growth1',r'C:\Users\yedidyab\Documents\zohar_data\121022_Psyringea_growth2',r'C:\Users\yedidyab\Documents\zohar_data\141122_Psyringea_growth3']
# for i in dirs:
#     path = i+r'\*.nd2'
#     predict_all_imgs(path)
