import torch
import torch.nn as nn
from torch.nn import MSELoss
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from module_utils import *
from function_utils import *
from scipy.io import loadmat, savemat
device = torch.device("cuda:0")

# # array signal parameters
fc = 1e9     # carrier frequency
c = 3e8      # light speed
M = 10        # array sensor number
N = 400       # snapshot number
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
batch_size_sf = 64
num_epoch_sf = 1000
learning_rate_sf = 0.001

# # file path of neural network parameters

# # array imperfection parameters
mc_flag = True
ap_flag = True
pos_flag = True
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
"training SF-NET"
sf_model = SF_NET(input_size=input_size, SF_NUM=SF_NUM, test_sf_flag=False, model_path_sf=model_name)
optimizer = torch.optim.Adam(sf_model.parameters(), lr=learning_rate_sf)
data_train_sf = generate_training_data_sf(M, N, d, wavelength, SNR_sf, 
                                              doa_min,NUM_REPEAT_SF, 
                                              grid_sf, GRID_NUM_SF,
                                              input_size, SF_NUM, 
                                              SF_SCOPE, MC_mtx, AP_mtx, 
                                              pos_para)
for epoch in range(num_epoch_sf):
    sf_model.train()
    [data_batches, label_batches] = generate_sf_batches(data_train_sf,batch_size_sf)
    data_batches =torch.tensor(data_batches,dtype=(torch.float32)).to(device)
    label_batches =torch.tensor(label_batches,dtype=(torch.float32)).to(device)
    for batch_idx in range(len(data_batches)):
        data_batch = data_batches[batch_idx]
        label_batch = label_batches[batch_idx]
        label_batch = label_batch.T
        optimizer.zero_grad()
        out_sf = sf_model(data_batch)
        loss_sf_fn = MSELoss(reduce=True, size_average=True)
        loss_sf_mse = loss_sf_fn(out_sf,label_batch.T)
        # loss_sf_toeplitz = toeplize_loss(label_batch,out_sf.T,M)
        # loss_sf_toeplitz = torch.tensor(loss_sf_toeplitz,requires_grad= True)
        # loss_sf = loss_sf_mse + loss_sf_toeplitz/(loss_sf_toeplitz/loss_sf_mse)
        loss_sf = loss_sf_mse
        loss_sf.backward()
        optimizer.step()
        # print('sf_Epoch: {}, Batch: {}, loss: {:g},loss_mse: {:g},loss_toep: {:g}'.format(epoch, batch_idx, loss_sf,loss_sf_mse,loss_sf_toeplitz))
        print('sf_Epoch: {}, Batch: {}, loss: {:g}'.format(epoch, batch_idx, loss_sf))
""" spatial filter gain and phase responsing """
data_in_sf = data_train_sf['input']
data_in_sf =torch.tensor(data_in_sf,dtype=(torch.float32)).to(device)
data_out_sf = sf_model(data_in_sf)

sp_gain_scan = np.zeros([SF_NUM,GRID_NUM_SF])
sp_phase_scan = np.zeros([SF_NUM,GRID_NUM_SF])
for grid_num in range(GRID_NUM_SF):
    r_in = data_in_sf[grid_num,:]
    r_in_complex = convert_real_to_complex(r_in)
    for sf_idx in range(SF_NUM):
        u_p = data_out_sf[grid_num,sf_idx * input_size : (sf_idx + 1) *input_size]
        u_p_complex = convert_real_to_complex(u_p)
        corr_i_ = torch.matmul(torch.conj(r_in_complex), u_p_complex)
        corr_i = torch.abs(corr_i_)
        norm2_r = torch.linalg.norm(r_in_complex)
        norm2_u = torch.linalg.norm(u_p_complex)
        sp_gain_scan[sf_idx,grid_num] = corr_i
        sp_phase_scan[sf_idx,grid_num] = corr_i/(norm2_r*norm2_u)

angle_space = np.linspace(doa_min, doa_max-1, num=GRID_NUM_SF)
plt.figure()
for sf_idx in range(SF_NUM):
    sp_gain = sp_gain_scan[sf_idx,:]
    plt.plot(angle_space,sp_gain)
plt.show()
# fig2a_plot = {}
# fig2a_plot['x_axis'] = angle_space
# fig2a_plot['y_axis'] = sp_gain_scan
# savemat('figure_data/fig_2a.mat', fig2a_plot)
plt.figure()
for sf_idx in range(SF_NUM):
    sp_phase = sp_phase_scan[sf_idx,:]
    plt.plot(angle_space,sp_phase)
plt.show()
# fig2b_plot = {}
# fig2b_plot['x_axis'] = angle_space
# fig2b_plot['y_axis'] = sp_phase_scan
# savemat('figure_data/fig_2b.mat', fig2b_plot)
""" spatial filter DOA responsing """ 
# # test data parameters
test_DOA = np.array([15,25])
test_K = len(test_DOA)
test_SNR = np.ones(test_K)*(10)
test_cov_vector,test_cov,test_cov_per = generate_sf_test_vector(M, N, d, wavelength, test_DOA, test_SNR, MC_mtx, AP_mtx, pos_para)
test_cov_vector = torch.unsqueeze(torch.tensor(test_cov_vector), 0)
test_cov_vector_ =torch.tensor(test_cov_vector,dtype=(torch.float32)).to(device)
vector_out_sf = sf_model(test_cov_vector_)
vector_out_sf = torch.squeeze(vector_out_sf)
sp_gain_scan = np.zeros([SF_NUM,GRID_NUM_SF])
sp_phase_scan = np.zeros([SF_NUM,GRID_NUM_SF])
for grid_num in range(GRID_NUM_SF):
    r_in = data_in_sf[grid_num,:]
    r_in_complex = convert_real_to_complex(r_in)
    for sf_idx in range(SF_NUM):
        u_p = vector_out_sf[sf_idx * input_size : (sf_idx + 1) *input_size]
        u_p_complex = convert_real_to_complex(u_p)
        
        corr_i_ = torch.matmul(torch.conj(r_in_complex), u_p_complex)
        corr_i = torch.abs(corr_i_)
        norm2_r = torch.linalg.norm(r_in_complex)
        norm2_u = torch.linalg.norm(u_p_complex)
        sp_gain_scan[sf_idx,grid_num] = corr_i
        sp_phase_scan[sf_idx,grid_num] = corr_i/(norm2_r*norm2_u)

plt.figure()
for sf_idx in range(SF_NUM):
    sp_gain_test = sp_gain_scan[sf_idx,:]
    plt.plot(angle_space,sp_gain_test)
plt.show()

# fig2c_plot = {}
# fig2c_plot['x_axis'] = angle_space
# fig2c_plot['y_axis'] = sp_gain_scan 
# savemat('figure_data/fig_2c.mat', fig2c_plot)

test_DOA = np.array([5,15])
test_K = len(test_DOA)
test_SNR = np.ones(test_K)*(10)
test_cov_vector,test_cov,test_cov_per = generate_sf_test_vector(M, N, d, wavelength, test_DOA, test_SNR, MC_mtx, AP_mtx, pos_para)
test_cov_vector = torch.unsqueeze(torch.tensor(test_cov_vector), 0)
test_cov_vector_ =torch.tensor(test_cov_vector,dtype=(torch.float32)).to(device)
vector_out_sf = sf_model(test_cov_vector_)
vector_out_sf = torch.squeeze(vector_out_sf)
sp_gain_scan = np.zeros([SF_NUM,GRID_NUM_SF])
sp_phase_scan = np.zeros([SF_NUM,GRID_NUM_SF])
for grid_num in range(GRID_NUM_SF):
    r_in = data_in_sf[grid_num,:]
    r_in_complex = convert_real_to_complex(r_in)
    for sf_idx in range(SF_NUM):
        u_p = vector_out_sf[sf_idx * input_size : (sf_idx + 1) *input_size]
        u_p_complex = convert_real_to_complex(u_p)
        
        corr_i_ = torch.matmul(torch.conj(r_in_complex), u_p_complex)
        corr_i = torch.abs(corr_i_)
        norm2_r = torch.linalg.norm(r_in_complex)
        norm2_u = torch.linalg.norm(u_p_complex)
        sp_gain_scan[sf_idx,grid_num] = corr_i
        sp_phase_scan[sf_idx,grid_num] = corr_i/(norm2_r*norm2_u)

plt.figure()
for sf_idx in range(SF_NUM):
    sp_gain_test = sp_gain_scan[sf_idx,:]
    plt.plot(angle_space,sp_gain_test)
plt.show()

# fig2d_plot = {}
# fig2d_plot['x_axis'] = angle_space
# fig2d_plot['y_axis'] = sp_gain_scan 
# savemat('figure_data/fig_2d.mat', fig2d_plot)