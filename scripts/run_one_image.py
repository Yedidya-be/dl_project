import utils
import sys
import os

temp_files_path = r'/home/labs/danielda/yedidyab/dl_project/temp_files'
segmentation_model = r'/home/labs/danielda/yedidyab/dl_project/models/cellpose_100X_model'
detect_div_model = r'/home/labs/danielda/yedidyab/dl_project/models/rf_detect_div_model_942acc.pkl'
single_cell_path = r'/home/labs/danielda/dl4cv_project/single_cell_without_background_128'
path = sys.argv[1]

# import sys
# sys.path.append(r'C:\Users\yedidyab\Box\Yedidya_Ben_Eliyahu\dl_proj\dl_project\scripts')
# import utils
# path = r'X:\yedidyab\dl_project\raw_data\Count00000_Point0011_ChannelPHASE 60x-100x PH3,DAPI,A488,A555,A647_Seq0011.nd2'
# temp_files_path = r'X:/yedidyab/dl_project/temp_files/'
# segmentation_model = r'X:/yedidyab/dl_project/models/cellpose_100X_model'
# detect_div_model = r'X:/yedidyab/dl_project/models/rf_detect_div_model_942acc.pkl'

def run_image(path):

    temp = utils.Img(path, temp_files_path=temp_files_path)
    temp.segment(model=segmentation_model)
    temp.alighnment()
    temp.reduce_high_signals()
    temp.predict_division(rf_model=detect_div_model)
    temp.replace_values_in_mask()
    temp.extract_single_cell_images(output_size=128, save_dir=single_cell_path)


run_image(path)