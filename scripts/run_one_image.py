import utils
import sys
import os

temp_files_path = r'/home/labs/danielda/yedidyab/dl_project/temp_files/'
segmentation_model = r'/home/labs/danielda/yedidyab/dl_project/models/cellpose_100X_model'
detect_div_model = r'/home/labs/danielda/yedidyab/dl_project/models/rf_detect_div_model_942acc.pkl'

path = sys.argv[1]


def run_image(path):

    temp = utils.Img(path, temp_files_path=temp_files_path)
    temp.segment(model=segmentation_model)
    temp.alighnment()
    temp.reduce_high_signals()
    temp.predict_division(rf_model=detect_div_model)
    temp.replace_values_in_mask()
    temp.extract_single_cell_images()


run_image(path)