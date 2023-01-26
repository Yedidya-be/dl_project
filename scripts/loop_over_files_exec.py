import glob
import subprocess
import os
import pandas as pd

path = r'/home/labs/danielda/yedidyab/dl_project/test_data/raw_data/*nd2'

single_cell_path = r'/home/labs/danielda/yedidyab/dl_project/test_data/single_cell_data_test/'
if not os.path.exists(single_cell_path):
    os.makedirs(single_cell_path)


def concatenate_csv(path):
    """
    This function takes all csv files in directory that have "single" in their name,
    and concatenates them together into a single csv file.
    """
    # use glob to match all CSV files with "single" in their name
    all_files = glob.glob(path + "/*single*.csv")

    # create an empty list to store the dataframes
    list_ = []

    # loop through the list of file names
    for file in all_files:
        # read the CSV file into a dataframe
        df = pd.read_csv(file)
        # add a column to the dataframe with the file name
        df['image'] = os.path.basename(file)
        # add the dataframe to the list
        list_.append(df)

    # concatenate all the dataframes in the list
    result = pd.concat(list_)

    # get the directory path of the first file in the list
    output_dir = os.path.dirname(all_files[0])
    # create the output file path
    output_file = os.path.join(output_dir, 'concatenated_single_csv.csv')

    # write the concatenated dataframe to a new CSV file
    result.to_csv(output_file, index=False)


for file in glob.iglob(path):
    print(file)

    if os.path.exists(single_cell_path + file.split('.')[0].split('/')[-1]):
        print(f'{file} already exist')
    elif 'hyb_12' in file:
        print('hyb_12 - skip')
    else:
        to_exec = 'bsub -gpu num=1 -q gpu-short -J test1 -eo /home/labs/danielda/yedidyab/dl_project/temp_files/wecax_out_err/errors_%J.txt -oo /home/labs/danielda/yedidyab/dl_project/temp_files/wecax_out_err/output_%J.txt -R rusage[mem=5000] python /home/labs/danielda/yedidyab/dl_project/scripts/run_one_test_image.py'.split()
        to_exec.append(file)
        subprocess.run(to_exec)

# save single_concatenated_csv
# path = r'/home/labs/danielda/dl4cv_project/temp_files/props_df/'
# concatenate_csv(path)
