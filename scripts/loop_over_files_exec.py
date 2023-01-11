import glob
import subprocess
import os

path = r'/home/labs/danielda/yedidyab/dl_project/raw_data/*nd2'
single_cell_path = r'/home/labs/danielda/yedidyab/dl_project/single_cell_data/'

for file in glob.iglob(path):

    if os.path.exists(single_cell_path + file.split('.')[0].split('/')[-1]):
        print(f'{file} already exist')
    else:
        to_exec = 'bsub -gpu num=1 -q gpu-short -J test1 -eo /home/labs/danielda/yedidyab/dl_project/temp_files/errors_%J.txt -oo /home/labs/danielda/yedidyab/dl_project/temp_files/output_%J.txt -R rusage[mem=5000] python /home/labs/danielda/yedidyab/dl_project/scripts/run_one_image.py'.split()
        to_exec.append(file)
        subprocess.run(to_exec)


