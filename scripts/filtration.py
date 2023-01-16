import os
import shutil
import pandas as pd
import glob

def filter_npy(path, single_cell_dir, th):
    """
    This function filters the npy files based on a threshold value for the "area" column,
    and moves the filtered files to another directory.
    """
    # read the concatenated csv file
    df = pd.read_csv(path)
    # create the output directory
    output_dir = os.path.dirname(path)
    filtered_dir = os.path.join(output_dir, '../filtered_npy')
    print(filtered_dir)
    if not os.path.exists(filtered_dir):
        os.makedirs(filtered_dir)
    # loop through the rows of the dataframe
    for i, row in df.iterrows():
        # check if the "area" value is below the threshold
        if row['area'] < th:
            # construct the path of the npy file
            npy_path = os.path.join(single_cell_dir, row['image'].split('_df_single')[0], 'label_' +str(row['label']) + '_*')
            print(npy_path)
            print(glob.glob(npy_path))
            to_move = glob.glob(npy_path)
            if len(to_move) == 1:
                npy_file = glob.glob(npy_path)[0]
                # move the npy file to the filtered directory
                shutil.move(npy_file, filtered_dir)

path = r'/home/labs/danielda/yedidyab/dl_project/temp_files/props_df/concatenated_single_csv.csv'
single_cell_dir = r'/home/labs/danielda/yedidyab/dl_project/single_cell_data'
th = 100
filter_npy(path, single_cell_dir, th)