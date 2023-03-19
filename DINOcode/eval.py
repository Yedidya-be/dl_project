import utils
import torch
from torchvision import models as torchvision_models
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import os

pretrained_weights = r'/home/labs/danielda/dl4cv_project/results/re18_300ep_1000dim/checkpoint.pth'
filename = r'/home/labs/danielda/dl4cv_project/results/resnet50_100epc/latent_space_dino.csv'
bact_db = r'/home/labs/danielda/dl4cv_project/data_for_einav2/all_channels/OutputImages/train'

if os.name == 'nt':
    pretrained_weights = r'X:/dl4cv_project/results/re18_300ep_1000dim/checkpoint.pth'
    filename = r'X:/dl4cv_project/results/resnet50_100epc/latent_space_dino.csv'
    bact_db = r'X:/dl4cv_project/data_for_einav2/all_channels/OutputImages/train'

#pretrained_weights = r'/home/labs/danielda/dl4cv_project/results/resnet50_100epc/checkpoint.pth'
model_name = 'resnet18'
checkpoint_key = 'teacher'
patch_size = None

model = torchvision_models.__dict__[model_name]()

#model = torchvision_models.__dict__[model_name](num_classes=2048)

model.fc = nn.Identity()
model.cuda()
utils.load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size)
# model.conv1.weight.data = torch.rand(model.conv1.weight.shape).cuda()
# for layer in model.children():
#    if hasattr(layer, 'reset_parameters'):
#        layer.reset_parameters()

model.eval()
# model(torch.rand(1,3,128,128).cuda()).shape

batch_size=32
if os.path.isfile(filename):
    os.remove(filename)

# dataset = utils.BactDataBase('/home/labs/danielda/yedidyab/dl_project/test_data/single_cell_data/train')
dataset = utils.BactDataBase(bact_db)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

out = torch.zeros(0, 2048).cuda()
fovs = torch.zeros(0, batch_size).cuda()
labels_list = torch.zeros(0,batch_size).cuda()
for i, (data, labels) in enumerate(data_loader):
    data = torch.moveaxis(data, -1, 1).to(torch.float32).cuda()
    _out = model(data)
    _fov = labels['fov'].cuda()
    _labels = labels['label'].cuda()
    concat = torch.concatenate([_fov.reshape(batch_size, 1), _labels.reshape(batch_size, 1), _out], dim=1).detach().cpu().numpy()
    with open(filename, 'a') as f:
        # Append the new data to the existing file with the same dimensions
        np.savetxt(f, concat, delimiter=',', comments='')

    if i == 200:
        break

