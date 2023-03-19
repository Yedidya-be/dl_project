import matplotlib.pyplot as plt
import json


log_path = r'X:\dl4cv_project\results\re18_300ep_1000dim\log.txt'
save_plot_path = r'X:\dl4cv_project\results\re18_300ep_1000dim\plot_loss.png'

# Open the file and read the contents
with open(log_path, 'r') as f:
    contents = f.read()

# Split the contents by newline character to get each line
lines = contents.split('\n')

# Extract the train_loss values from each line
train_losses = []
for line in lines:
    if line:
        train_loss = json.loads(line)['train_loss']
        train_losses.append(train_loss)

# Plot the train_loss values
plt.plot(train_losses)
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.show()
plt.savefig(save_plot_path)

