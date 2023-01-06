import numpy as np
from pp_new import get_phase_projection
from tifffile import imwrite
import glob
import nd2

directory = r'C:\Users\yedidyab\Box\Yedidya_Ben_Eliyahu\data\010822_yedidia_pao1_allighen/*.nd2'


def correct_z(dir, out_path, to_crop=False):
    for filepath in glob.iglob(dir):
        name = filepath.split('\\')[-1].split('.')[0]
        print(name)
        image = nd2.imread(filepath)
        phase, _ = get_phase_projection(image)
        dapi = np.max(image[:, 1, :, :], axis=0)
        to_save = np.array([phase, dapi])
        # print(3)

        if to_crop:
            phase_d = phase[500:1000, 500:1000]
            dapi_d = dapi[500:1000, 500:1000]
            imwrite(fr'{out_path}\{name}_partial.tif', np.array([phase_d, dapi_d]))
            print(f'save {out_path}\{name}_partial.tif')
        else:
            imwrite(fr'{out_path}\{filepath}_corrcted.tif', to_save)
            print(f'save {filepath}')


def test():
    correct_z(r'C:\Users\yedidyab\Documents\zohar_data\111022_Psyringea_growth1\*.nd2', r'C:\Users\yedidyab\Documents\zohar_data\111022_Psyringea_growth1\new_crop', to_crop=True)

# test()