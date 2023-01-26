import utils
import sys
import os
import numpy as np

temp_files_path = r'/home/labs/danielda/yedidyab/dl_project/test_data/temp_files'
segmentation_model = r'/home/labs/danielda/yedidyab/dl_project/models/cellpose_100X_model'
detect_div_model = r'/home/labs/danielda/yedidyab/dl_project/models/rf_detect_div_model_942acc.pkl'

path1 = sys.argv[1]
path2 = path1.replace('hyb_1', 'hyb_12')


# import sys
# sys.path.append(r'C:\Users\yedidyab\Box\Yedidya_Ben_Eliyahu\dl_proj\dl_project\scripts')
# import utils
# path = r'X:\yedidyab\dl_project\raw_data\Count00000_Point0011_ChannelPHASE 60x-100x PH3,DAPI,A488,A555,A647_Seq0011.nd2'
# temp_files_path = r'X:/yedidyab/dl_project/temp_files/'
# segmentation_model = r'X:/yedidyab/dl_project/models/cellpose_100X_model'
# detect_div_model = r'X:/yedidyab/dl_project/models/rf_detect_div_model_942acc.pkl'
def run_image(path1, path2):

    temp = utils.Img(path1, temp_files_path=temp_files_path)
    temp2 = utils.Img(path2, temp_files_path=temp_files_path)
    temp.wga = temp2.wga
    temp.segment(model=segmentation_model)
    temp.alighnment()
    # temp.predict_division(rf_model=detect_div_model)
    # temp.replace_values_in_mask()
    temp.reduce_high_signals()
    temp.extract_single_cell_images()


run_image(path1, path2)