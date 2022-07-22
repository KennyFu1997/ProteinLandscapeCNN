import time
import math
import numpy as np
import itertools
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


global meta_data
meta_data = np.load("./IntermediateData/meta_data.npy", allow_pickle=True)
global OUTPATH1
OUTPATH1 = './IntermediateData/inputs'
global OUTPATH2
OUTPATH2 = './IntermediateData/labels'

# Check the mean range of a protein in x, y and z axis.
# This is to decide the size of box using for spatial sampling.
stat = []
for i in range(meta_data.shape[0]):
    xmin = np.min(meta_data[i][:, 1])
    xmax = np.max(meta_data[i][:, 1])
    ymin = np.min(meta_data[i][:, 2])
    ymax = np.max(meta_data[i][:, 2])
    zmin = np.min(meta_data[i][:, 3])
    zmax = np.max(meta_data[i][:, 3])
    stat.append([xmax-xmin, ymax-ymin, zmax-zmin])

# Compute box size, sample 1000 points
mean_range = np.mean(np.mean(np.array(stat), axis=0))
box_size = int(np.ceil(mean_range + 0.2*mean_range))
num_samples = 10

# Create helper histogram.
x = np.linspace(0, box_size, num_samples)
y = np.linspace(0, box_size, num_samples)
z = np.linspace(0, box_size, num_samples)
helper_linspacex, helper_linspacey, helper_linspacez = np.meshgrid(x, y, z)
helper_linspace = []
for i in range(num_samples):
    for j in range(num_samples):
        for k in range(num_samples):
            helper_linspace.append([helper_linspacex[i, j, k], helper_linspacey[i, j, k], helper_linspacez[i, j, k]])
helper_linspace = np.array(helper_linspace)

# Function to compute transformation.
def compute_transformation(CB, CA, C, N):
    
    xh = (C - CA).astype(float)
    xh /= np.linalg.norm(xh)
    yh = (N - CA).astype(float)
    yh /= np.linalg.norm(yh)
    zh = np.cross(xh, yh)
    zh /= np.linalg.norm(zh)

    T = np.zeros((4, 4))
    T[-1, -1] = 1
    T[:-1, 0] = xh
    T[:-1, 1] = yh
    T[:-1, 2] = zh
    T[:-1, 3] = CB
    
    return np.linalg.inv(T)

# 计算每条蛋白质内选中的aa的index
resids_all = []
#start_time = time.time()
for i, prot in enumerate(meta_data):
    atoms_coords = prot[:, 1:4].astype(float)
    n_atoms = atoms_coords.shape[0]
    # 所有原子坐标的最小值作为原点
    origin = np.min(atoms_coords, axis=0)
    # 所有的取样坐标
    sample_pos = helper_linspace + origin.reshape(1, -1)
    
    # 然后向量化计算最近的原子
    samples_tmp = np.repeat(sample_pos, n_atoms, axis=0)
    coords_tmp = np.repeat(atoms_coords.flatten().reshape(1, -1), np.power(num_samples, 3), axis=0).reshape(-1, 3)
    tmp = coords_tmp - samples_tmp
    # 这里有坑，注意python是行主导
    dis_tmp = np.linalg.norm(tmp, axis=1).reshape(np.power(num_samples, 3), -1)
    center_ids = np.argmin(dis_tmp, axis=1)
    # shape: (1000,)
    # value in each position means the index of chosen atom
    # Then we use these indices to exrtact atoms from meta_data and check their residue id.
    # After we have the corresponding residue ids, we use them to find CB/CA/N/C, which 
    # are used to compute transformation for each center residue
    
    # residue ids
    # 包含的是在一条蛋白质内选中的aa的集合
    resids = np.unique(prot[center_ids, -1])
    resids_all.append(resids) 

#print("--- %s seconds ---" % (time.time() - start_time))

cnt = 0
for i, resids in enumerate(resids_all):
    
    for resid in resids:

        mask = meta_data[i][:, -1] == resid
        micro_center = meta_data[i][mask, :]
        
        # and the center point
        # 这里有多个CB的话，只取第一个CB
        # (为什么会有多个CB？)
        
        CB = np.array(micro_center[micro_center[:, 0] == 'CB', 1:4]).flatten()
        '''
        if CB.shape[0] > 3:
            CB = CB[:3]
            print(micro_center)
        '''
        center = CB.copy()
        dis_tmp = np.linalg.norm((meta_data[i][:, 1:4] - center.reshape(1, -1)).astype(float), axis=1)
        micro_tmp = meta_data[i][dis_tmp <= 10, :]
        
        # T
        CA = np.array(micro_center[micro_center[:, 0] == 'CA', 1:4]).flatten()
        N = np.array(micro_center[micro_center[:, 0] == 'N', 1:4]).flatten()
        C = np.array(micro_center[micro_center[:, 0] == 'C', 1:4]).flatten()
        T = compute_transformation(CB, CA, C, N)
        
        # affine transformation
        coords_tmp = micro_tmp[:, 1:4].astype(float)
        coords_tmp = np.hstack(( coords_tmp, np.ones((coords_tmp.shape[0], 1)) )).T
        coords = (T @ coords_tmp)[:-1, :].T
        micro_tmp[:, 1:4] = coords
        
        # save as input
        np.save( os.path.join(OUTPATH1, 'envs')+f'{cnt}.npy' , micro_tmp.astype(object) )
        np.save( os.path.join(OUTPATH2, 'labs')+f'{cnt}.npy' , micro_center[0, -2] )

        cnt += 1
