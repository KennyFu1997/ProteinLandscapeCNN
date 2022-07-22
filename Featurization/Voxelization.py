#!/usr/bin/env python
# coding: utf-8

# # Take the orientated micro-envs and centers as input, output 3D grid representation, which will be used for 3D CNN later. #

# In[1]:


import numpy as np
import os
import math
import torch
import torch.utils.data
import torchvision.transforms as T
import time


# In[2]:


inp_path = "./IntermediateData/inputs"
lab_path = "./IntermediateData/labels"
out1 = "../NeuralNetwork/inputs"
out2 = "../NeuralNetwork/labels"
kernel_size = 7
train_scale = 0.9
blur = T.GaussianBlur(kernel_size)
aa_dict = {'SER': 0,'MET': 1,'PRO': 2,'ASP': 3,'LEU': 4,'THR': 5,'ASN': 6,'VAL': 7,'ALA': 8,'GLU': 9,'TYR': 10,'GLN': 11,'PHE': 12,'GLY': 13,'ILE': 14,'TRP': 15,'ARG': 16,'LYS': 17,'CYS': 18,'HIS': 19,'UNK': 20}
aa_dict_reversed = {v: k for k, v in aa_dict.items()}
lnp = np.linspace(0, 20, 20)

micros_tmp = list(sorted(os.listdir(inp_path)))
labels_tmp = list(sorted(os.listdir(lab_path)))

if '.DS_Store' in micros_tmp:
    micros_tmp.remove('.DS_Store')
if '.DS_Store' in labels_tmp:
    labels_tmp.remove('.DS_Store')
if '.ipynb_checkpoints' in micros_tmp:
    micros_tmp.remove('.ipynb_checkpoints')
if '.ipynb_checkpoints' in labels_tmp:
    labels_tmp.remove('.ipynb_checkpoints')
    
assert len(micros_tmp) == len(labels_tmp), "Data error!"
num_micros = len(micros_tmp)
idxs = np.array(range(num_micros))
np.random.shuffle(idxs)
micros_tmp = np.array(micros_tmp)[idxs]
labels_tmp = np.array(labels_tmp)[idxs]


# In[3]:


start_time = time.time()
micros = torch.zeros(num_micros, 4, 20, 20, 20)
labels = torch.zeros(num_micros)
for idx in range(num_micros):
    inp_name = micros_tmp[idx]
    micro_raw = np.load(os.path.join(inp_path, inp_name), allow_pickle=True)
    micro = torch.zeros(4, 20, 20, 20)

    for i, info in enumerate(micro_raw):
        coord = info[1:4].astype(float)
        coord_grid = []
        for j in range(3):
            pos, bins = np.histogram(coord[j], lnp)
            coord_grid.append(np.nonzero(pos)[0])
        if info[0] == 'N':
            micro[0, coord_grid[0], coord_grid[1], coord_grid[2]] += 1
        if info[0] == 'C':
            micro[1, coord_grid[0], coord_grid[1], coord_grid[2]] += 1
        if info[0] == 'O':
            micro[2, coord_grid[0], coord_grid[1], coord_grid[2]] += 1
        if info[0] == 'CA':
            micro[3, coord_grid[0], coord_grid[1], coord_grid[2]] += 1
    
    micro = blur(micro)
    micros[idx, :, :, :, :] = micro
    
    lab_name = labels_tmp[idx]
    label = np.load(os.path.join(lab_path, lab_name), allow_pickle=True)
    label = aa_dict[str(label)]
    labels[idx] = label
print("--- %s seconds ---" % (time.time() - start_time))
        


# In[5]:


torch.save(micros, os.path.join(out1, 'micros.pt'))
torch.save(labels, os.path.join(out2, 'labels.pt'))

