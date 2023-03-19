import subprocess

latents = [64, 32, 16, 8, 4]

for l in latents:
    to_exec = f'bsub -q new-long -R "rusage[mem=200GB]" -eo /home/labs/danielda/yedidyab/dl_project/temp_files/wecax_out_err/errors_%J.txt -oo /home/labs/danielda/yedidyab/dl_project/temp_files/wecax_out_err/output_%J.txt python /home/labs/danielda/dl4cv_project/ME-VAE_Architecture/main.py --data_dir /home/labs/danielda/yedidyab/dl_project/test_data/single_cell_data/ --input2_dir /home/labs/danielda/yedidyab/dl_project/test_data/single_cell_data/ --out1_dir /home/labs/danielda/yedidyab/dl_project/test_data/single_cell_data/ --save_dir /home/labs/danielda/yedidyab/dl_project/results/vae/Outputs_l{l} --image_size 128 --latent_dim {l} --batch_size 10 --epochs 10 --nchannel 5 --verbose 1'
    subprocess.run(to_exec, shell = True)
