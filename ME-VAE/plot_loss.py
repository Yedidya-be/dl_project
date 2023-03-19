import matplotlib.pyplot as plt

log_path = r'X:\dl4cv_project\data_for_einav2\all_channels\Outputs_l32\training.log'
save_plot_path = r'C:\Users\yedidyab\Box\Yedidya_Ben_Eliyahu\dl_proj\dl_project\notebooks\vae_figs\32\plot_loss_VAE.png'


#log_path = r'X:\dl4cv_project\data_for_einav\Outputs\training.log'
#save_plot_path = r'C:\Users\yedidyab\Box\Yedidya_Ben_Eliyahu\dl_proj\dl_project\notebooks\MEVAE\plot_loss_MEVAE.png'

# Read the loss values from the text file
with open(log_path, 'r') as f:
    lines = f.readlines()

epochs = []
losses = []

for line in lines[1:]:
    epoch, loss = line.split()
    epochs.append(int(epoch))
    losses.append(float(loss))

# Create a line plot of the loss values
plt.plot(epochs, losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.show()
plt.savefig(save_plot_path)
