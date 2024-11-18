# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 22:14:28 2024

@author: P51
"""

import torch
import torch.nn as nn
from torch.nn import MSELoss
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from spatial_filters_ultils import *
from function_utils import *
from scipy.io import loadmat, savemat
device = torch.device("cuda:0")

# # array signal parameters
fc = 1e9     # carrier frequency
c = 3e8      # light speed
M = 10        # array sensor number
N = 200       # snapshot number
K = 2
wavelength = c / fc  # signal wavelength
d = 0.5 * wavelength  # inter-sensor distance
# # spatial filtern (SF) training parameters
SF_NUM = 6       # number of spatial filters
doa_min = -60      # minimal DOA (degree)
doa_max = 60       # maximal DOA (degree)
grid_sf = 1        # DOA step (degree) for generating different scenarios
GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
SF_SCOPE = (doa_max - doa_min) / SF_NUM   # spatial scope of each filter
SNR_sf = 10
NUM_REPEAT_SF = 1    # number of repeated sampling with random noise


# # autoencoder parameters
input_size = M * M * 2 # the dimention of autoencoder's input layer 
batch_size_sf = 32
num_epoch_sf = 1000
learning_rate_sf = 0.001

# # training set parameters of classified NN
K = 2     # signal number
NUM_REPEAT_MUSIC = 50000    # number of repeated sampling with random noise
# # DNN parameters
grid_music = 0.1    # inter-grid angle in spatial spectrum
output_size_music = int((doa_max - doa_min + 0.5 * grid_music) / grid_music)   # spectrum grids
angle_space = np.linspace(doa_min, doa_max-0.1, num=output_size_music)
batch_size_music = 2048
learning_rate_music = 0.001
num_epoch_music = 50
# 构造协方差矩阵
doa_grid = np.linspace(doa_min, doa_max-0.1, num=output_size_music)
Amatrix = ula_array(M,d,wavelength,doa_grid).to(device)

# # array imperfection parameters
mc_flag = False
ap_flag = False
pos_flag = False
# Rho = np.linspace(0, 10,num=11)
# for rho in Rho:
rho = 0
model_name = 'spatialfilter'
# mutual coupling matrix
if mc_flag == True:
    model_name = model_name + '_mc'
    mc_para = rho * 0.3 * np.exp(1j * 60 / 180 * np.pi)
    MC_coef = mc_para ** np.array(np.arange(M))
    MC_mtx = la.toeplitz(MC_coef)
else:
    model_name = model_name
    MC_mtx = np.identity(M)
# amplitude & phase error
if ap_flag == True:
    model_name = model_name + '_ap'
    amp_coef = rho * np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2, -0.2])
    phase_coef = rho * np.array([0.0, -30, -30, -30, -30, -30, 30, 30, 30, 30])
    AP_coef = [(1+amp_coef[idx])*np.exp(1j*phase_coef[idx]/180*np.pi) for idx in range(M)]
    AP_mtx = np.diag(AP_coef)
else:
    model_name = model_name
    AP_mtx = np.identity(M)
# sensor position error
if pos_flag == True:
    model_name = model_name + '_pos'
    pos_para_ = rho * np.array([0.0, -1, -1, -1, -1, -1, 1, 1, 1, 1]) * 0.2 * d
    pos_para = np.expand_dims(pos_para_, axis=-1)
else:
    model_name = model_name
    pos_para = np.zeros([M, 1])
model_name = model_name + '_' + str(rho) + '.pt'

"training MUSIC-NET"
data_train_music = generate_training_data_music(M, N, K, d, wavelength,doa_min, doa_max, NUM_REPEAT_MUSIC,MC_mtx, AP_mtx, pos_para)
Rdata = data_train_music['cov_matrix']
Rdata = np.array(Rdata)
Rdata_real = Rdata.real
Rdata_imag = Rdata.imag

Rdata_real_ = torch.tensor(np.expand_dims(Rdata_real, 1),dtype=(torch.float32))
Rdata_imag_ = torch.tensor(np.expand_dims(Rdata_imag, 1),dtype=(torch.float32))
data_music_in = np.concatenate([Rdata_real_,Rdata_imag_],1)

Rdata_real = torch.tensor(Rdata_real,dtype=(torch.float32))
Rdata_imag = torch.tensor(Rdata_imag,dtype=(torch.float32))
Rdata_ = Rdata_real + 1j*Rdata_imag
Rdata_ = np.array(Rdata_)
music_model = nn.Sequential(nn.Conv2d(2, M, kernel_size=3,padding=1),
                            nn.ReLU(),
                            nn.Conv2d(M, 2*M, kernel_size=3,padding=1),
                            nn.ReLU(),
                            nn.Conv2d(2*M, 4*M, kernel_size=3,padding=1),
                            nn.ReLU(),
                            nn.Conv2d(4*M, 2*M, kernel_size=3,padding=1),
                            nn.ReLU(),
                            nn.Flatten(),
                            nn.Linear(2*M*M*M, 1000),
                            nn.ReLU(),
                            nn.Linear(1000, 2000),
                            nn.ReLU(),
                            nn.Linear(2000, output_size_music),
                            nn.ReLU())
music_model = music_model.to(device)
optimizer = torch.optim.Adam(music_model.parameters(), lr=learning_rate_music)
for epoch in range(num_epoch_music):
    music_model.train()
    data_batches, Rdata_batches = generate_music_batches(data_music_in,Rdata_,batch_size_music)
    for batch_idx in range(len(data_batches)):
        optimizer.zero_grad()
        data_batch = data_batches[batch_idx]
        data_batch_cuda = torch.tensor(data_batch).to(device)
        Rdata_batch = Rdata_batches[batch_idx]
        # Rdata_batch = np.array(Rdata_batch)
        Rdata_batch_cuda = torch.tensor(Rdata_batch).to(device)
        music_output = music_model(data_batch_cuda)
        loss_music = music_loss_fn(Rdata_batch_cuda,music_output,Amatrix,M,K)
        'loss backward'
        loss_music.backward()
        optimizer.step()
    print('music_Epoch: {}, loss_music: {:g}'.format(epoch, loss_music))
torch.save(music_model, 'music_net/music_net.pkl')

"test MUSIC-NET"
test_DOA = np.array([-45.7,5.1,20.3])
test_K = len(test_DOA)
test_SNR = np.ones(test_K)*(-3)
test_cov = generate_testing_data_music(M, N, test_K, d, wavelength,test_DOA,test_SNR, MC_mtx, AP_mtx, pos_para)
Rdata_real_ = torch.tensor(np.expand_dims(test_cov.real, 0),dtype=(torch.float32))
Rdata_imag_ = torch.tensor(np.expand_dims(test_cov.imag, 0),dtype=(torch.float32))
data_music_test = np.concatenate([Rdata_real_,Rdata_imag_],0)
data_music_test = np.expand_dims(data_music_test,0)
data_music_in_cuda = torch.tensor(data_music_test,dtype=(torch.float32)).to(device)
music_test = music_model(data_music_in_cuda)
testResult = np.squeeze(music_test.cpu().detach().numpy())
testResult = testResult/np.max(testResult)
Pmusic_out = ula_music_algrithm(test_cov,M,test_K,angle_space,d,wavelength)
plt.figure()
plt.plot(angle_space,Pmusic_out,'k')
plt.plot(angle_space,testResult,'r')
plt.show()
# fig4a_plot = {}
# fig4a_plot['x_axis'] = angle_space
# fig4a_plot['music_spec'] = Pmusic_out
# fig4a_plot['music_net'] = testResult
# fig4a_plot['true_doa'] = test_DOA
# fig4a_plot['true_doa_power'] = np.ones(test_K)
# savemat('figure_data/fig_4a.mat', fig4a_plot)

factor = 1
n_feature = np.where(testResult >=0.01)
n_feature = np.array(n_feature).T
n_sequnse = np.linspace(0, len(n_feature)-1,len(n_feature))
n_sequnse = np.expand_dims(n_sequnse,1)
data_kmeans = np.concatenate((n_sequnse,n_feature),axis=1)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=test_K)
kmeans.fit(data_kmeans)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.figure()
for ki in range(test_K):
    plot_data = data_kmeans[labels==ki,:]
    plot_data[:,1] = plot_data[:,1]/10 + doa_min
    plt.scatter(plot_data[:,0],plot_data[:,1],s=100,marker = 'o')
centroids[:,1] = centroids[:,1]/10 + doa_min
centroids = np.take_along_axis(centroids, np.argsort(centroids,axis=0), axis=0)
plt.scatter(centroids[:,0],centroids[:,1],s=50,c='r',marker = '*')
plt.scatter(centroids[:,0],test_DOA,s=80,edgecolor='black', facecolors='none')
plt.show()
centroids[:,1]

# fig4b_plot = {}
# fig4b_plot['doa_cluster_ind'] = centroids[:,0]
# fig4b_plot['doa_cluster_est'] = centroids[:,1]
# fig4b_plot['true_doa'] = test_DOA
# for ki in range(test_K):
#     plot_data = data_kmeans[labels==ki,:]
#     plot_data[:,1] = plot_data[:,1]/10 + doa_min
#     fig4b_plot['cluster_'+str(ki)+'indx'] = plot_data[:,0]
#     fig4b_plot['cluster_'+str(ki)+'val'] = plot_data[:,1]
# savemat('figure_data/fig_4b.mat', fig4b_plot)