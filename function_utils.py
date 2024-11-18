# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:46:47 2024

@author: P51
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
"sf_model train data generation"
def generate_training_data_sf(M, N, d, wavelength, SNR, doa_min, NUM_REPEAT_SF, grid, GRID_NUM, output_size, SF_NUM, SF_SCOPE, MC_mtx, AP_mtx, pos_para):
    data_train_sf = {}
    data_train_sf['input'] = []
    data_train_sf['label'] = []
    for doa_idx in range(GRID_NUM):
        DOA = doa_min + grid * doa_idx
        for rep_idx in range(NUM_REPEAT_SF):
            # DOA = np.random.uniform(doa_min,doa_min+GRID_NUM-1,1)
            SNR_sf = np.random.uniform(SNR,SNR,1)
            add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
            array_signal = 0
            signal_i = 10 ** (SNR_sf / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
            # phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
            
            array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
            phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA / 180 * np.pi)
            a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
            a_i = np.matmul(AP_mtx, a_i)
            a_i = np.matmul(MC_mtx, a_i)
            array_signal_i = np.matmul(a_i, signal_i)
            array_signal += array_signal_i
            array_output = array_signal + 1 * add_noise
            array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
            cov_vector = np.reshape(array_covariance,[-1])
            cov_vector = 1 / np.linalg.norm(cov_vector) * cov_vector
            cov_vector = np.concatenate([cov_vector.real, cov_vector.imag])
            data_train_sf['input'].append(cov_vector)
            
            # construct multi-task autoencoder target
            array_signal_per = 0
            array_geom_per = np.expand_dims(np.array(np.arange(M)), axis=-1) * d
            phase_shift_array_per = 2 * np.pi * array_geom_per / wavelength * np.sin(DOA / 180 * np.pi)
            a_i_per = np.cos(phase_shift_array_per) + 1j * np.sin(phase_shift_array_per)
            array_signal_i_per = np.matmul(a_i_per, signal_i)
            array_signal_per += array_signal_i_per
            array_output_per = array_signal_per + 1 * add_noise
            array_covariance_per = 1 / N * (np.matmul(array_output_per, np.matrix.getH(array_output_per)))
            cov_vector_per = np.reshape(array_covariance_per,[-1])
            cov_vector_per = 1 / np.linalg.norm(cov_vector_per) * cov_vector_per
            cov_vector_per = np.concatenate([cov_vector_per.real, cov_vector_per.imag])
            scope_label = int((DOA - doa_min) / SF_SCOPE)
            target_curr_pre = np.zeros([output_size * scope_label, 1])
            target_curr_post = np.zeros([output_size * (SF_NUM - scope_label - 1), 1])
            target_curr = np.expand_dims(cov_vector_per, axis=-1)
            target = np.concatenate([target_curr_pre, target_curr, target_curr_post], axis=0)
            data_train_sf['label'].append(np.squeeze(target))

    return data_train_sf

def generate_sf_batches(data_train, batch_size):
    data_ = data_train['input']
    label_ = data_train['label']
    data_len = len(label_)
    # shuffle data
    shuffle_seq = np.random.permutation(range(data_len))
    data = [data_[idx] for idx in shuffle_seq]
    label = [label_[idx] for idx in shuffle_seq]
    # generate batches
    num_batch = int(data_len / batch_size)
    data_batches = []
    label_batches = []
    for batch_idx in range(num_batch):
        batch_start = batch_idx * batch_size
        batch_end = np.min([(batch_idx + 1) * batch_size, data_len])
        data_batch = data[batch_start : batch_end]
        label_batch = label[batch_start: batch_end]
        data_batches.append(data_batch)
        label_batches.append(label_batch)
    return data_batches, label_batches


def toeplize_loss(R_label,R_pred,M):
    R_label = R_label.cpu().detach().numpy()
    R_pred = R_pred.cpu().detach().numpy()
    [len_in,batch_size] = R_pred.shape
    R_label_binaray = np.where(np.abs(R_label) > 0,1,0)
    R_pred = R_pred*R_label_binaray
    len_r = M*M*2
    
    R_pred = np.reshape(R_pred,[-1,len_r,batch_size])
    R_in = np.sum(R_pred,0)
    R_in_real = R_in[:int(len_r/2),:]
    R_in_imag = R_in[int(len_r/2):,:]
    # R_complex = R_in_real + 1j* R_in_imag
    cov_matrix_real = np.reshape(R_in_real,[M,M,batch_size])
    cov_matrix_imag = np.reshape(R_in_imag,[M,M,batch_size])
    loss_toeplitz = 0
    loss_curr = 0
    for row_idx in range(M-1):
        if (row_idx == 0):
            diag_real = np.diagonal(cov_matrix_real,row_idx).T
            diag_imag = np.diagonal(cov_matrix_imag,row_idx).T
            loss_curr = np.sum((diag_real - np.mean(diag_real,0))**2)/len(diag_real)/batch_size 
            + np.sum((diag_imag - np.mean(diag_imag,0))**2)/len(diag_imag)/batch_size
        elif(row_idx == M-1):
            diag_real1 = np.diagonal(cov_matrix_real,row_idx)
            diag_real2 = np.diagonal(cov_matrix_real,-row_idx)
            diag_imag1 = np.diagonal(cov_matrix_imag,row_idx)
            diag_imag2 = -np.diagonal(cov_matrix_imag,-row_idx)
            loss_mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)
            loss_curr = loss_mse_fn(diag_real1,diag_real2) + loss_mse_fn(diag_imag1,diag_imag2)
        else:
            diag_real1 = np.diagonal(cov_matrix_real,row_idx).T
            diag_real2 = np.diagonal(cov_matrix_real,-row_idx).T
            diag_imag1 = np.diagonal(cov_matrix_imag,row_idx).T
            diag_imag2 = -np.diagonal(cov_matrix_imag,-row_idx).T
            diag_real = np.concatenate([diag_real1,diag_real2])
            diag_imag = np.concatenate([diag_imag1,diag_imag2])
            loss_mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)
            loss_curr = np.sum((diag_real1 - diag_real2)**2)/len(diag_real1)/batch_size 
            + np.sum((diag_imag1 - diag_imag2)**2)/len(diag_imag1)/batch_size 
            + np.sum((diag_real - np.mean(diag_real,0))**2)/len(diag_real)/batch_size
            + np.sum((diag_imag - np.mean(diag_imag,0))**2)/len(diag_imag)/batch_size
        loss_toeplitz += loss_curr
    return loss_toeplitz

def convert_real_to_complex(real_vector):
    vector_len = len(real_vector)
    vector_real = real_vector[:int((vector_len + 1) / 2)]
    vector_imag = real_vector[int((vector_len + 1) / 2):]
    complex_vector = vector_real + 1j * vector_imag
    return complex_vector

def generate_sf_test_vector(M, N, d, wavelength, DOA, SNR, MC_mtx, AP_mtx, pos_para):
    K = len(DOA)
    add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
    array_signal = 0
    array_signal_per = 0
    for ki in range(K):
        signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        # signal_i_per = (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
        array_geom_per = np.expand_dims(np.array(np.arange(M)), axis=-1) * d
        phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
        phase_shift_per = 2 * np.pi * array_geom_per / wavelength * np.sin(DOA[ki] / 180 * np.pi)
        a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
        a_i = np.matmul(AP_mtx, a_i)
        a_i = np.matmul(MC_mtx, a_i)
        a_i_per = np.cos(phase_shift_per) + 1j * np.sin(phase_shift_per)
        array_signal_i = np.matmul(a_i,  signal_i)
        array_signal += array_signal_i
        array_signal_i_per = np.matmul(a_i_per, signal_i)
        array_signal_per += array_signal_i_per
    array_output = array_signal + add_noise
    array_output_per = array_signal_per + add_noise
    array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
    array_covariance_per = 1 / N * (np.matmul(array_output_per, np.matrix.getH(array_output_per)))
    cov_vector = np.reshape(array_covariance,[-1])
    cov_vector = 1 / np.linalg.norm(cov_vector) * cov_vector
    cov_vector = np.concatenate([cov_vector.real, cov_vector.imag])
    return cov_vector, array_covariance, array_covariance_per

"""" MUSIC-NET train data generating """
def generate_training_data_music(M, N, K, d, wavelength, doa_min, doa_max, NUM_REPEAT, MC_mtx, AP_mtx, pos_para):
    data_train_music = {}
    data_train_music['cov_matrix'] = []
    data_train_music['cov_matrix_vec'] = []
    # for delta_idx in range(len(doa_delta)):
    #     delta_curr = doa_delta[delta_idx]  # inter-signal direction differences
    #     delta_cum_seq_ = [delta_curr]  # doa differences w.r.t first signal
    #     delta_cum_seq = np.concatenate([[0], delta_cum_seq_])  # the first signal included
    #     delta_sum = np.sum(delta_curr)  # direction difference between first and last signals
    #     NUM_STEP = int((doa_max - doa_min - delta_sum) / step)  # number of scanning steps

    #     for step_idx in range(NUM_STEP):
    #         doa_first = doa_min + step * step_idx
    #         DOA = delta_cum_seq + doa_first

    for rep_idx in range(NUM_REPEAT):
        DOA = np.random.uniform(doa_min,doa_max-0.1,K)
        SNR = np.random.uniform(0,10,K)
        add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
        array_signal = 0
        for ki in range(K):
            signal_i =  10 ** (SNR[ki] / 20) *(np.random.randn(1, N) + 1j * np.random.randn(1, N))
            array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
            phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
            a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
            a_i = np.matmul(AP_mtx, a_i)
            a_i = np.matmul(MC_mtx, a_i)
            array_signal_i = np.matmul(a_i, signal_i)
            array_signal += array_signal_i
        array_output = array_signal + 1 * add_noise
        array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
        # array_covariance = 1 / np.linalg.norm(array_covariance) * array_covariance
        array_covariance_vec = array_covariance.reshape(-1,1)
        array_covariance_vec = np.concatenate([array_covariance_vec.real, array_covariance_vec.imag])
        # cov_vector_nf_ext = np.concatenate([cov_vector_nf_.real, cov_vector_nf_.imag])
        data_train_music['cov_matrix'].append(array_covariance) 
        data_train_music['cov_matrix_vec'].append(array_covariance_vec)
    return data_train_music

def generate_testing_data_music(M, N, d, wavelength, test_DOA, test_SNR, test_K, MC_mtx, AP_mtx, pos_para):
    add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
    array_signal = 0 
    for ki in range(test_K):
        # print(ki)
        signal_i =  10 ** (test_SNR[ki] / 20) *(np.random.randn(1, N) + 1j * np.random.randn(1, N))
        array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
        phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(test_DOA[ki] / 180 * np.pi)
        a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
        a_i = np.matmul(AP_mtx, a_i)
        a_i = np.matmul(MC_mtx, a_i)
        array_signal_i = np.matmul(a_i, signal_i)
        array_signal += array_signal_i
    array_output = array_signal + 1 * add_noise
    array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
    array_covariance = 1 / np.linalg.norm(array_covariance) * array_covariance
    array_covariance_vec = array_covariance.reshape(-1,1)
    array_covariance_vec = np.concatenate([array_covariance_vec.real, array_covariance_vec.imag])
    array_covariance_vec = 1 / np.linalg.norm(array_covariance_vec) * array_covariance_vec
    # cov_vector_nf_ext = np.concatenate([cov_vector_nf_.real, cov_vector_nf_.imag])
    # data_train_music['cov_matrix'].append(array_covariance) 
    # data_train_music['cov_matrix_vec'].append(array_covariance_vec)
    return array_covariance_vec, array_covariance

def generate_music_batches(data_train, Rdata,batch_size):
    # data_ = data_train['cov_matrix_vec']
    # label_ = data_train['cov_matrix']
    data_len = len(data_train)

    # shuffle data
    shuffle_seq = np.random.permutation(range(data_len))
    data = [data_train[idx] for idx in shuffle_seq]
    Rdata = [Rdata[idx] for idx in shuffle_seq]

    # generate batches
    num_batch = int(data_len / batch_size)
    data_batches = []
    Rdata_batches = []
    for batch_idx in range(num_batch):
        batch_start = batch_idx * batch_size
        batch_end = np.min([(batch_idx + 1) * batch_size, data_len])
        data_batch = data[batch_start : batch_end]
        Rdata_batch = Rdata[batch_start: batch_end]
        data_batches.append(data_batch)
        Rdata_batches.append(Rdata_batch)
    return data_batches, Rdata_batches

# def generate_music_batches_clean(data_train,batch_size):
#     # data_ = data_train['cov_matrix_vec']
#     # label_ = data_train['cov_matrix']
#     data_len = len(data_train)

#     # shuffle data
#     shuffle_seq = np.random.permutation(range(data_len))
#     data = [data_train[idx] for idx in shuffle_seq]
#     # Rdata = [Rdata[idx] for idx in shuffle_seq]

#     # generate batches
#     num_batch = int(data_len / batch_size)
#     data_batches = []
#     # Rdata_batches = []
#     for batch_idx in range(num_batch):
#         batch_start = batch_idx * batch_size
#         batch_end = np.min([(batch_idx + 1) * batch_size, data_len])
#         data_batch = data[batch_start : batch_end]
#         # Rdata_batch = Rdata[batch_start: batch_end]
#         data_batches.append(data_batch)
#         # Rdata_batches.append(Rdata_batch)
#     return data_batches
# def ula_array(M,d,wavelength,DOA,pos_para,AP_mtx,MC_mtx):
#     K = len(DOA)
#     A = []
#     for k in range(K):
#         a_k = ula_steer_vector(M,d,wavelength,DOA[k],pos_para,AP_mtx,MC_mtx)
#         A.append(a_k)
#     return np.array(A)

# def ula_steer_vector(M,d,wavelength,DOA,pos_para,AP_mtx,MC_mtx):
#     array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
#     phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA / 180 * np.pi)
#     a_k = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
#     a_k = np.matmul(AP_mtx, a_k)
#     a_k = np.matmul(MC_mtx, a_k)
#     return a_k
def ula_array(M,d,wavelength,DOA):
    K = len(DOA)
    A = []
    for k in range(K):
        a_k = ula_steer_vector(M,d,wavelength,DOA[k])
        A.append(a_k)
    A = np.array(A)
    A = np.squeeze(A)
    A = torch.tensor(A,dtype=torch.complex64)
    return A.T

def ula_steer_vector(M,d,wavelength,DOA):
    array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d
    phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA / 180 * np.pi)
    a_k = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
    # a_k = np.matmul(AP_mtx, a_k)
    # a_k = np.matmul(MC_mtx, a_k)
    return a_k


def music_loss_fn(Rdata,netOut,Amatrix,M,K):
    trainNum = len(Rdata)
    lamada, Us = torch.linalg.eig(Rdata)
    Un = Us[:,:,K:]
    Gn = torch.matmul(Un, torch.conj(Un.transpose(1,2)))
    AmatrixH = torch.conj(Amatrix.T)
    temp = torch.matmul(AmatrixH,Gn)
    temp1 = torch.transpose(temp,1,2).reshape(trainNum,-1)
    Amatrix1 = Amatrix.reshape(-1)
    temp2 = torch.mul(temp1,Amatrix1).reshape(trainNum,M,-1)
    Pmusic = torch.sum(temp2,1)
    Pmusic = 1/abs(Pmusic)
    Pmusic = Pmusic.cpu().numpy()
    Pmax = 1/np.max(Pmusic,1)
    Pmax = torch.from_numpy(Pmax)
    Pmax = Pmax.unsqueeze(1)
    Pmusic = torch.from_numpy(Pmusic)
    Pmusic = torch.mul(Pmax,Pmusic).to('cuda:0')
    loss = F.mse_loss(netOut.reshape(1,-1),Pmusic.reshape(1,-1))
    return loss

def ula_music_algrithm(R,M,K,angle_space,d,wavelength):
    M = len(R)
    Amatrix = ula_array(M,d,wavelength,angle_space).cpu().detach().numpy()
    lamada, Us = np.linalg.eig(R)
    Un = Us[:,K:]
    Gn = np.matmul(Un, np.conj(Un.T))
    AmatrixH = np.conj(Amatrix.T)
    temp = np.matmul(AmatrixH,Gn)
    temp1 = temp.T.reshape(-1)
    Amatrix1 = Amatrix.reshape(-1)
    temp2 = np.multiply(temp1,Amatrix1).reshape(M,-1)
    Pmusic = np.sum(temp2,0)
    Pmusic = 1/abs(Pmusic)
    Pmax = 1/np.max(Pmusic)
    Pmusic = Pmusic * Pmax
    return Pmusic

def threhold_filter(data_sf_out_np,SF_NUM,M,threhold,input_size):
    # filtering and threhold mean
    cov_matrix = []
    for sample_indx in range(len(data_sf_out_np)):
        cov_temp = []
        data_sf_out_temp = data_sf_out_np[sample_indx,:]
        for sf_indx in range(SF_NUM):
            cov_ = data_sf_out_temp[input_size*sf_indx:input_size*(sf_indx+1)]
            cov_real = cov_[:M*M].reshape(M,M)
            cov_imag = cov_[M*M:].reshape(M,M)
            cov = cov_real + 1j*cov_imag
            # print(np.linalg.norm(cov))
            
            if (np.linalg.norm(cov)>threhold):
                cov_temp.append(cov)
        cov_temp = np.array(cov_temp)
        cov_temp_ = np.sum(cov_temp,0)
        cov_matrix.append(cov_temp_)
    Rdata = np.array(cov_matrix)
    return Rdata*1000

# def generate_training_data_music_clean(M, N, K, d, wavelength, doa_min, doa_max, NUM_REPEAT):
#     data_train_music = []
#     # data_train_music['cov_matrix'] = []
#     for rep_idx in range(NUM_REPEAT):
#         DOA = np.random.uniform(doa_min,doa_max-0.1,K)
#         # if(np.abs(DOA[1]-DOA[0])<10):
#         #     DOA = np.random.uniform(doa_min,doa_max,K)
#         SNR = np.random.uniform(0,10,K)
#         add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
#         array_signal = 0
#         for ki in range(K):
#             signal_i =  10 ** (SNR[ki] / 20) *(np.random.randn(1, N) + 1j * np.random.randn(1, N))
#             array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1)
#             phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
#             a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
#             array_signal_i = np.matmul(a_i, signal_i)
#             array_signal += array_signal_i
#         array_output = array_signal + add_noise
#         array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
#         array_covariance = 1 / np.linalg.norm(array_covariance) * array_covariance
#         data_train_music.append(array_covariance) 
#     return data_train_music

# def generate_music_test_cov(M, N, d, wavelength, DOA, SNR):
#     K = len(DOA)
#     add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
#     array_signal = 0
#     for ki in range(K):
#         signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
#         array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d
#         phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
#         a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
#         array_signal_i = np.matmul(a_i,  signal_i)
#         array_signal += array_signal_i
#     array_output = array_signal + add_noise
#     array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
#     array_covariance = 1 / np.linalg.norm(array_covariance) * array_covariance
#     return array_covariance


# def generate_testing_data_music(M, N, K, d, wavelength,DOA,SNR, MC_mtx, AP_mtx, pos_para):
#     add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
#     array_signal = 0
#     for ki in range(K):
#         signal_i =  10 ** (SNR[ki] / 20) *(np.random.randn(1, N) + 1j * np.random.randn(1, N))
#         array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
#         phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
#         a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
#         a_i = np.matmul(AP_mtx, a_i)
#         a_i = np.matmul(MC_mtx, a_i)
#         array_signal_i = np.matmul(a_i, signal_i)
#         array_signal += array_signal_i
#         array_output = array_signal + 1 * add_noise
#         array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
#         # array_covariance = 1 / np.linalg.norm(array_covariance) * array_covariance
#     return array_covariance
def peaks_search(spec,test_DOA,doa_min):
    peaks_indx = find_peaks(spec)
    doa_est = peaks_indx[0]/10 + doa_min
    doa_net_est = []
    for k in range(len(test_DOA)):
        err = []
        for i in range(len(doa_est)):
            err.append(doa_est[i]-test_DOA[k])
        doa_net_est.append(doa_est[np.argmin(np.abs(err))])
    doa_net_est = np.array(doa_net_est)
    return doa_net_est