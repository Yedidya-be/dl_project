import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
import pandas as pd
from detect_division import Img
import glob
import tqdm
import math
import seaborn as sns


def extract_single_cell_data_from_pairs_df(df):
    """
    This function takes in a dataframe df and extracts relevant data for each cell in the dataframe.
    It does this by iterating through each row of the input dataframe and extracting the cell indices,
     major pair, prediction, and probability. These extracted data are then stored in a new list called extracted_data.

    After all rows have been processed, a new dataframe is created from the extracted_data list and named extracted_df.
    The extracted_df dataframe is then modified in the following ways:

    The rows are sorted in ascending order by the probability column.
    For each unique cell index, the row with the highest probability is dropped.
    The resulting dataframe is filtered to only include rows where the prediction column is equal to 1.
    Finally, the modified dataframe is returned by the function.

Parameters:
    df (pandas DataFrame): The input dataframe.

Returns:
    pandas DataFrame: A modified dataframe with relevant data for each cell.
"""
    # Initialize empty list to store data
    extracted_data = []
    # Iterate through rows of the input dataframe
    for index, major_pair, orientation_p,orientation_l, prediction, probability in zip(df.idx, df.major_pairs, df.orientation,df.orientation_line, df.prediction, df.prob):
        # Split the index into two cell indices
        cell_index_1, cell_index_2 = [int(j) for j in index.split('&')]

        # Append data for both cell indices
        extracted_data.append([cell_index_1, major_pair, orientation_p, orientation_l, prediction, probability, cell_index_2])
        extracted_data.append([cell_index_2, major_pair, orientation_p, orientation_l, prediction, probability, cell_index_1])

    # Create a new dataframe with the extracted data
    extracted_df = pd.DataFrame(extracted_data,
                                columns=['cell_idx', 'major_pairs', 'orientation_p', 'orientation_line', 'prediction', 'probability', 'pair_idx'])

    # Drop rows with the highest probability for each unique cell index
    extracted_df.sort_values(by='probability', inplace=True)
    extracted_df.drop_duplicates(subset='cell_idx', keep='first', inplace=True)

    # Select only rows with a prediction of 1
    extracted_df = extracted_df[extracted_df.prediction == 1]

    # Return the modified dataframe
    return extracted_df


def create_single_df(path = None,
                     rf_model = r'C:\Users\yedidyab\Documents\zohar_data\sampels\new_rf_label_again_942acc_all_featurs.pkl',
                     single_df = None, pairs_df = None, name=None):
    """
    Creates a dataframe for a single image.
    Parameters:
        path (str): The path to the image.
        rf_model: A trained random forest model.

    Returns:
        pandas DataFrame: A dataframe with data for each cell in the image.
    """

    # # Read image
    image = Img(path)
    #
    # Perform image segmentation
    image.segment()

    # Build dataframe with image properties
    image.build_all_props_df()

    # Calculate DAPI and ribosomal DNA content for each cell
    image.calc_dapi_ribo()

    # Use trained model to make predictions for each cell
    image.predict(rf_model)

    # Extract relevant data for each cell
    pairs_df = extract_single_cell_data_from_pairs_df(image.prop_df)

    # Create a dataframe with data for each cell
    single_df = image.props_data

    # Add the image name as a column
    single_df['name'] = name

    # Rename the 'label' column to 'cell_idx'
    single_df.rename(columns={'label': 'cell_idx'}, inplace=True)

    # Merge the pairs data with the single cell data
    single_df = single_df.merge(pairs_df, on='cell_idx', how='left')

    # Replace null values in the 'major_pairs' column with the 'axis_major_length' value
    single_df.major_pairs.fillna(single_df.axis_major_length, inplace=True)

    # Replace null values in the 'pred' column with 0
    single_df.prediction.fillna(0, inplace=True)

    # Add the image name as a column
    single_df['name'] = image.name

    # Return the modified dataframe
    return single_df


def make_bins_col(df, to_bin, n_bins):
    """
    This function takes a dataframe, a column to bin, and the number of bins as input.
    It creates a new column in the dataframe with the same name as the original column,
    but with "_bins" appended to it. The new column contains the binned values of the
    original column, which are converted to integers.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    to_bin (str): The name of the column in the dataframe to bin.
    n_bins (int): The number of bins to create.

    Returns:
    None
    """
    # Use pandas.cut to create bins for the values in the specified column
    # retbins=True returns the bin edges
    bins = pd.cut(np.array(df[to_bin]), n_bins, retbins=True, labels=range(1, n_bins + 1))[0]

    # Create the new column name by appending "_bins" to the original column name
    new_bin_name = to_bin + '_bins'

    # Add the new column to the dataframe with the binned values converted to integers
    df[new_bin_name] = bins.astype('int')


def read_image_from_df(cell_data, dir=r'C:\\Users\\yedidyab\\Documents\\zohar_data\\sampels\\', space=2):
    """
    This function reads an image from a file specified in a given dataframe, segments the image, and returns a
    subimage defined by bounding box coordinates contained in the dataframe.

    Parameters:
    cell_data (pandas.DataFrame): A dataframe containing the following columns:
        - name: the file name of the image to read
        - bbox-0: the minimum row index of the bounding box
        - bbox-1: the minimum column index of the bounding box
        - bbox-2: the maximum row index of the bounding box
        - bbox-3: the maximum column index of the bounding box
    dir (str): The directory where the image files are stored.
    space (int): The number of rows/columns to add to the bounding box on each side.

    Returns:
    numpy.ndarray: A subimage of the input image defined by the bounding box coordinates and the space parameter.
    """
    # Calculate the extended bounding box coordinates
    min_r = cell_data['bbox-0'] - space
    min_c = cell_data['bbox-1'] - space
    max_r = cell_data['bbox-2'] + space
    max_c = cell_data['bbox-3'] + space

    # Construct the file path to the image file
    img_path = dir + cell_data['name'] + r'.nd2'

    # Read the image file and segment it
    image = Img(img_path)
    image.segment()

    # Extract the subimage defined by the bounding box coordinates
    cell_img = image.phase_projection[min_r: max_r, min_c: max_c]

    return cell_img


main_dir = r'C:\\Users\\yedidyab\\Documents\\zohar_data\\sampels\\'
df = pd.read_csv(r'C:\\Users\\yedidyab\\Documents\\zohar_data\\sampels\\df_singel_with_pair_major_new.csv')

th=0.3
df.loc[abs(df.orientation_p) < 1-th, 'prediction'] = 0
df.loc[abs(df.orientation_p) > 1+th, 'prediction'] = 0
th_or=0.5
df.loc[df.orientation_line < 0-th, 'prediction'] = 0
df.loc[df.orientation_line > 0+th, 'prediction'] = 0

df_div = df[df.prediction == 1]
df_not_div = df[df.prediction != 1]

n_bins = 6

# bin the cells by major
make_bins_col(df_not_div, 'major_pairs', n_bins)

# set cells that in division to the last bin
df_div['major_pairs_bins'] = n_bins + 1

df = pd.concat([df_div, df_not_div])

# show grid by size
example_number = 7

# Group the data by the "major_pairs_bins" column and select 4 random rows from each group
grouped = df.groupby('major_pairs_bins').apply(lambda x: x.sample(example_number))

# Create a figure with subplots for each "major_pairs_bins" value
fig, axs = plt.subplots(example_number, len(grouped.index.levels[0]))

# Transpose the subplot array so that the "cat" values are columns
axs = axs.T

# Iterate over the groups and the selected rows
for (cat, group), ax in zip(grouped.iterrows(), axs.flatten()):
    # Get the data for the current group
    cell_data = group

    # Extract the bounding box coordinates from the data
    min_r = cell_data['bbox-0']
    min_c = cell_data['bbox-1']
    max_r = cell_data['bbox-2']
    max_c = cell_data['bbox-3']

    if cell_data['prediction'] == 1:
        # Get the pair index from cell_data
        pair_idx = cell_data['pair_idx']
        image_name = cell_data['name']
        # print(f'{pair_idx=}')
        # print(f'{image_name=}')
        pair_data = df.loc[(df.cell_idx == pair_idx) & (df.name == image_name)]
        # Update min_r, min_c, max_r, and max_c using the maximum of the corresponding values in cell_data and df
        probability = cell_data['probability']
        print(f'{probability=}')
        min_r = min(min_r, pair_data['bbox-0'].values[0])
        min_c = min(min_c, pair_data['bbox-1'].values[0])
        max_r = max(max_r, pair_data['bbox-2'].values[0])
        max_c = max(max_c, pair_data['bbox-3'].values[0])

    # Construct the image path using the 'name' column and the main directory
    img_path = main_dir + cell_data['name'] + r'.nd2'

    # Create an Img object and segment the image
    image = Img(img_path)
    image.segment()

    # Crop the image using the bounding box coordinates
    cell_img = image.phase_projection[min_r: max_r, min_c: max_c]

    # Display the image in the current subplot
    ax.imshow(cell_img)

    # Remove the image coordinates from the display
    # ax.set_xticks([])
    # ax.set_yticks([])

    # Set the title of the subplot to the "cat" value
    print(cell_data['orientation_p'])
    ax.set_title('cat = {}'.format(cell_data['orientation_p']))

# Show the figure
plt.show()

# craete df forom the begining
dfs = []
rf_model = r'C:\Users\yedidyab\Documents\zohar_data\sampels\new_rf_label_again_942acc_all_featurs.pkl'
train_path = r'C:\\Users\\yedidyab\\Documents\\zohar_data\\sampels\\*.nd2'
for filepath in tqdm.tqdm(glob.iglob(train_path)):
    df = create_single_df(filepath, rf_model)
    dfs.append(df)

df = pd.concat(dfs)
df.to_csv(r'C:\\Users\\yedidyab\\Documents\\zohar_data\\sampels\\df_singel_with_pair_major_new.csv')

di = {'A1': 3.3, 'B1': 5.7,
      'C1': 0.04, 'D1': 0.07,
      'A2': 3.7, 'B2': 2.64,
      'C2': 0.09, 'D2': 0.25,
      'A3': 0.5, 'B3': 1,
      'C3': 1.6, 'D3': 2}

df['OD'] = df.name.str[6] + df.name.str[-1]
df.replace({"OD": di}, inplace=True)
df['id_name'] = df.name.str[6] + df.name.str[-1]

features = ['area', 'dapi_sum', 'ribo_sum', 'dapi_cv', 'ribo_cv', 'major_pairs']
# normalization
df_norm = (df[features] - df[features].mean()) / df[features].std()
df_min_max = (df[features] - df[features].min()) / df[features].max()
lut = dict(zip(df.OD.unique(), sns.color_palette("hls", len(df.OD.unique()))))
row_colors = df.OD.map(lut)
sns.clustermap(df[features], row_colors=row_colors)
plt.show()

# dfs=[]
# i=0
# rf_model = r'C:\Users\yedidyab\Documents\zohar_data\sampels\new_rf_label_again_after_search_942acc.pkl'
# train_path = r'C:\Users\yedidyab\Documents\zohar_data\sampels\*.nd2'
# for filepath in tqdm.tqdm(glob.iglob(train_path)):
#     df = create_singel_df(filepath, rf_model)
#     dfs.append(df)
#     i+=1
#     if i==2:
#         break
# df=pd.concat(dfs)
