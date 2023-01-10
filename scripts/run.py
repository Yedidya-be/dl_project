import main

test = main.Img(r'Y:\Automation\automation_runs\20221227_160445_692\Count00000_Point0025_ChannelPHASE 60x-100x PH3,DAPI,A488,A555,A647_Seq0025.nd2')
test.segment()
test.alighnment()
test.reduce_high_signals()
test.predict_division()
test.replace_values_in_mask()
test.extract_single_cell_images()
