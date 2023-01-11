import glob

path = r'/home/labs/danielda/yedidyab/dl_project/raw_data/*nd2'
i=3
for file in glob.iglob(path):
    print(rf'python /home/labs/danielda/yedidyab/dl_project/scripts/run_one_file.py {file}')
    exec(rf'python /home/labs/danielda/yedidyab/dl_project/scripts/run_one_file.py {file}')
    i += 1
    if i > 3:
        break