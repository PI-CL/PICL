from tkinter import N
import torch
import operator
import numpy as np
import os, sys
from functools import reduce
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import yaml
import random

def data_repeat(data, num_obs, num_state, gap):
    
    data_run = data

    num = data_run.shape[1]
    data_run = data_run

    u_sum = torch.zeros((data_run.shape[2],num,num_state,num_state), dtype=torch.float64)
    v_sum = torch.zeros((data_run.shape[2],num,num_state,num_state), dtype=torch.float64)
    h_sum = torch.zeros((data_run.shape[2],num,num_state,num_state), dtype=torch.float64)

    for uvh in range(data_run.shape[0]):  
        for t in range(data_run.shape[2]): 

            u_m = torch.zeros((num,num_obs,num_obs), dtype=torch.float64)
            
            for i in range(u_m.shape[1]):
                u_m[:,i] = data_run[uvh,:,t,i*u_m.shape[1]:i*u_m.shape[1] + u_m.shape[2]]

            u_s = torch.zeros((num,num_state,num_state), dtype=torch.float64)

            for i in range(u_s.shape[1]):
                if i%gap == 0:
                    for z in range(u_s.shape[2]):
                        if z%gap == 0:
                            u_s[:,i,z] = u_m[:, int(i/gap), int(z/gap)]
            
            if uvh == 0:
                u_sum[t] = u_s
            elif uvh == 1:
                v_sum[t] = u_s
            elif uvh == 2:
                h_sum[t] = u_s

    sum = torch.stack((u_sum, v_sum, h_sum), 0)

    return sum

def data_import(data_path, full_data_path, traj_str_tr, traj_end_tr, traj_str_ts, traj_end_ts, trsta, trend, tssta, tsend,
                num_obs, num_state, gap):
    
    data_run = torch.load(data_path)   # uvp, traj, t, ox*oy
    full_data = torch.load(full_data_path)  # uvp, traj, t, x, y

    data_train = data_run[:,traj_str_tr:traj_end_tr]  # uvp, traj, t, ox*oy
    full_train = full_data[:,traj_str_tr:traj_end_tr]  # uvp, traj, t, x, y
    data_test = data_run[:,traj_str_ts:traj_end_ts]
    full_test = full_data[:,traj_str_ts:traj_end_ts]

    data_train = data_train.permute(1,0,2,3)  # traj, uvp, t, ox*oy
    data_train = data_train[:,:,trsta:trend]
    data_test = data_test.permute(1,0,2,3)  # traj, uvp, t, ox*oy
    data_test = data_test[:,:,tssta:tsend]

    full_train = full_train.permute(1,0,2,3,4)  # traj, uvp, t, ox*oy
    full_train = full_train[:,:,trsta:trend]
    full_test = full_test.permute(1,0,2,3,4)  # traj, uvp, t, ox*oy
    full_test = full_test[:,:,tssta:tsend]

    data_run_train, data_run_test, full_train, full_test, logs = normal(data_train, data_test, full_train, full_test, logs)  # traj, uvp, t, ox*oy

    data_run_train = data_run_train.permute(1,0,2,3)  # uvp, traj, t, ox*oy
    data_run_test = data_run_test.permute(1,0,2,3)

    data_train_re = data_repeat(data_run_train, num_obs, num_state, gap)  # uvp, t, traj, x, y
    data_test_re = data_repeat(data_run_test, num_obs, num_state, gap)

    data_train_re = data_train_re.permute(2,0,1,3,4)  # traj, uvp, t, x, y
    data_test_re = data_test_re.permute(2,0,1,3,4)

    data_run = torch.cat((data_train_re, full_train.to(data_train_re.device)), axis=1)  # traj, uvp*2, t, x, y
    data_test = torch.cat((data_test_re, full_test.to(data_test_re.device)), axis=1)  # traj, uvp*2, t, x, y
    
    return data_run, data_test

def data_divide(data_run, data_test, recall_size, N0, unlabel_size):
    
    input_data =data_run[:, :, :-1, :]  # traj, uvp*2, t, x, y
    output_data = data_run[:, :, 1:, :]
    
    input_data_test = data_test[:, :, :-1, :]
    output_data_test = data_test[:, :, 1:, :]

    data_in = input_data.permute(0,2,1,3,4)  # traj, t, uvp*2, x, y
    labels_out = output_data.permute(0,2,1,3,4)

    data_in_ensmble = np.empty((data_in.shape[0],data_in.shape[1]-recall_size+1,recall_size*6,data_in.shape[-2],data_in.shape[-1]))
    labels_out_ensmble = np.empty((labels_out.shape[0],labels_out.shape[1]-recall_size+1,recall_size*6,labels_out.shape[-2],labels_out.shape[-1]))
    
    data_in_test = input_data_test.permute(0,2,1,3,4)  # # traj, t, uvp*2, x, y
    labels_out_test = output_data_test.permute(0,2,1,3,4)
    data_in_test_ensmble = np.empty((data_in_test.shape[0],data_in_test.shape[1]-recall_size+1,recall_size*6,data_in_test.shape[-2],data_in_test.shape[-1]))
    labels_out_test_ensmble = np.empty((labels_out_test.shape[0],labels_out_test.shape[1]-recall_size+1,recall_size*6,labels_out_test.shape[-2],labels_out_test.shape[-1]))

    for i in range(data_in_ensmble.shape[1]):
        data_in_ensmble[:,i] = data_in[:,i:i+recall_size].reshape(-1,recall_size*6,data_in.shape[-2],data_in.shape[-1])
        labels_out_ensmble[:,i] = labels_out[:,i:i+recall_size].reshape(-1,recall_size*6,labels_out.shape[-2],labels_out.shape[-1])
        
    for i in range(data_in_test_ensmble.shape[1]):
        data_in_test_ensmble[:,i] = data_in_test[:,i:i+recall_size].reshape(-1,recall_size*6,data_in_test.shape[-2],data_in_test.shape[-1])
        labels_out_test_ensmble[:,i] = labels_out_test[:,i:i+recall_size].reshape(-1,recall_size*6,labels_out_test.shape[-2],labels_out_test.shape[-1])
    
    train_data = torch.tensor(data_in_ensmble, dtype=torch.float64)  # traj, t_en, uvp*2*n, x, y
    train_labels = torch.tensor(labels_out_ensmble, dtype=torch.float64)
      
    unlabel_data = train_data[-int(N0*unlabel_size):]
    unlabel_lab = train_labels[-int(N0*unlabel_size):]
    train_data = train_data[:-int(N0*unlabel_size)]
    train_labels = train_labels[:-int(N0*unlabel_size)]

    unlabel_data = unlabel_data.view(-1, unlabel_data.shape[2], unlabel_data.shape[3], unlabel_data.shape[4])
    unlabel_lab = unlabel_lab.view(-1, unlabel_lab.shape[2], unlabel_lab.shape[3], unlabel_lab.shape[4])
    train_data = train_data.view(-1, train_data.shape[2], train_data.shape[3], train_data.shape[4])
    train_labels = train_labels.view(-1, train_labels.shape[2], train_labels.shape[3], train_labels.shape[4])
    
    test_data = torch.tensor(data_in_test_ensmble, dtype=torch.float64)
    test_labels = torch.tensor(labels_out_test_ensmble, dtype=torch.float64)
    test_data = test_data.view(-1, test_data.shape[2], test_data.shape[3], test_data.shape[4])
    test_labels = test_labels.view(-1, test_labels.shape[2], test_labels.shape[3], test_labels.shape[4])
    
    return unlabel_data, unlabel_lab, train_data, train_labels, test_data, test_labels

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def ensemble(input_data, output_data, recall_size):
    data_in = input_data.permute(1,0,2,3)
    labels_out = output_data.permute(1,0,2,3)
    data_in_ensmble = np.empty((data_in.shape[0]-recall_size,recall_size*6,data_in.shape[-2],data_in.shape[-1]))
    labels_out_ensmble = np.empty((labels_out.shape[0]-recall_size,recall_size*6,labels_out.shape[-2],labels_out.shape[-1]))
    for i in range(data_in_ensmble.shape[0]):
        data_in_ensmble[i] = data_in[i:i+recall_size].reshape(-1,data_in.shape[-2],data_in.shape[-1])  # (nt, 3*recall, 41, 41)
        labels_out_ensmble[i] = labels_out[i:i+recall_size].reshape(-1,labels_out.shape[-2],labels_out.shape[-1])  # (nt, 3*recall, 41, 41)
    return data_in_ensmble, labels_out_ensmble

def size(data_run):
    N0, nt, nx, ny = 1, data_run.shape[1], data_run.shape[2],data_run.shape[3]
    shape = [nx, ny]
    return shape, N0, nt, nx, ny

def data_divide(data_run):
    input_data =torch.tensor(data_run[:, :, :-1, :])
    output_data = torch.tensor(data_run[:, :, 1:, :])
    return input_data, output_data

def normal(data_run_train, data_run_test, full_run_train, full_run_test, logs):

    u_mean, u_std = data_run_train[:,0,:,:].mean(), data_run_train[:,0,:,:].std()
    v_mean, v_std = data_run_train[:,1,:,:].mean(), data_run_train[:,1,:,:].std()
    h_mean, h_std = data_run_train[:,2,:,:].mean(), data_run_train[:,2,:,:].std()
    logs['mean'].append(u_mean)
    logs['mean'].append(v_mean)
    logs['mean'].append(h_mean)
    logs['std'].append(u_std)
    logs['std'].append(v_std)
    logs['std'].append(h_std)

    data_run_train[:,0,:,:] = (data_run_train[:,0,:,:] - u_mean) / u_std
    data_run_train[:,1,:,:] = (data_run_train[:,1,:,:] - v_mean) / v_std
    data_run_train[:,2,:,:] = (data_run_train[:,2,:,:] - h_mean) / h_std
    full_run_train[:,0,:,:,:] = (full_run_train[:,0,:,:,:] - u_mean) / u_std
    full_run_train[:,1,:,:,:] = (full_run_train[:,1,:,:,:] - v_mean) / v_std
    full_run_train[:,2,:,:,:] = (full_run_train[:,2,:,:,:] - h_mean) / h_std

    data_run_test[:,0,:,:] = (data_run_test[:,0,:,:] - u_mean) / u_std
    data_run_test[:,1,:,:] = (data_run_test[:,1,:,:] - v_mean) / v_std
    data_run_test[:,2,:,:] = (data_run_test[:,2,:,:] - h_mean) / h_std
    full_run_test[:,0,:,:,:] = (full_run_test[:,0,:,:,:] - u_mean) / u_std
    full_run_test[:,1,:,:,:] = (full_run_test[:,1,:,:,:] - v_mean) / v_std
    full_run_test[:,2,:,:,:] = (full_run_test[:,2,:,:,:] - h_mean) / h_std


    return data_run_train, data_run_test, full_run_train, full_run_test, logs

def load_config(file):
    with open(file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    return config

def rel_error(x, _x):

    if len(x.shape)==1:
        x = x.reshape(1, -1)
        _x = _x.reshape(1, -1)
    else:
        B = x.size(0)
        x, _x = x.reshape(B, -1), _x.reshape(B, -1)

    return torch.norm(x - _x, 2, dim=1)/torch.norm(_x, 2, dim=1)

def abs_error(x, _x):

    if len(x.shape)==1:
        x = x.reshape(1, -1)
        _x = _x.reshape(1, -1)
    else:
        B = x.size(0)
        x, _x = x.reshape(B, -1), _x.reshape(B, -1)
    
    return torch.norm(x - _x, 2, dim=1) 

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

def masker(sample_gap, data):
    out_data = data
    weight_loss = torch.tensor(np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2])))
    for i in range(out_data.shape[1]):
        for z in range(out_data.shape[2]):
            if z%sample_gap == 0 and i%sample_gap == 0:
                weight_loss[:,i,z] = 0
            else:
                weight_loss[:,i,z] = out_data[:,i,z]

    return weight_loss

def fdmd2D(u, device, Lx, Ly):
    bs = u.shape[0]
    nx = u.shape[1]
    ny = u.shape[2]
    dimu = u.shape[-1]
    dx = Lx / nx
    dy = Ly / ny

    ux = torch.zeros(bs, nx, ny, dimu).to(device)
    uy = torch.zeros(bs, nx, ny, dimu).to(device)
    for i in range(nx-1):
        ux[:, i] = (u[:, i+1] - u[:, i]) / dx
    ux[:, -1] = ux[:, -2]
    for j in range(ny-1):
        uy[:, :, j] = (u[:, :, j+1] - u[:, :, j]) / dy
    uy[:, :, -1] = uy[:, :, -2]

    return ux, uy

def fftd2D(u, device):
    nx = u.shape[-3]
    ny = u.shape[-2]
    dimu = u.shape[-1]
    u_h = torch.fft.fft2(u, dim=[1, 2]).reshape(-1, nx, ny, dimu)

    k_x = torch.arange(-nx//2, nx//2) * 2 * torch.pi / 2.2
    k_y = torch.arange(-ny//2, ny//2) * 2 * torch.pi / 0.41
    k_x = torch.fft.fftshift(k_x)
    k_y = torch.fft.fftshift(k_y)
    k_x = k_x.reshape(nx, 1).repeat(1, ny).reshape(1,nx,ny,1).to(device)
    k_y = k_y.reshape(1, ny).repeat(nx, 1).reshape(1,nx,ny,1).to(device)
    lap = -(k_x ** 2 + k_y ** 2)

    ux_h = 1j * k_x * u_h
    uy_h = 1j * k_y * u_h
    ulap_h = lap * u_h

    ux = torch.fft.ifft2(ux_h, dim=[1, 2])
    uy = torch.fft.ifft2(uy_h, dim=[1, 2])
    u_lap = torch.fft.ifft2(ulap_h, dim=[1, 2])

    ux = torch.real(ux).reshape(-1, nx, ny, dimu)
    uy = torch.real(uy).reshape(-1, nx, ny, dimu)
    u_lap = torch.real(u_lap).reshape(-1, nx, ny, dimu)

    return ux, uy, u_lap

def calMean(data_list):
    ans = []
    for data in data_list:
        length = data.shape[0]
        if (length % 10 != 0):
            data = data[:-(length % 10)]
        data = data.reshape(length // 10, 10, -1).mean(1)
        ans.append(data)
    return ans

def calVar(data_list):
    ans = []
    for data in data_list:
        length = data.shape[0]
        if (length % 10 != 0):
            data = data[:-(length % 10)]
        data_min = data.reshape(length // 10, 10, -1).min(1)
        data_max = data.reshape(length // 10, 10, -1).max(1)
        ans.append([data_min.values, data_max.values])
    return ans

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PredLog():
    def __init__(self, length):
        self.length = length
        self.loss1 = AverageMeter()
        self.loss2 = AverageMeter()
        self.loss3 = AverageMeter()
        self.loss4 = AverageMeter()
        self.loss5 = AverageMeter()
        self.loss6 = AverageMeter()
        self.loss7 = AverageMeter()
        self.loss8 = AverageMeter()
        self.loss9 = AverageMeter()
    
    def update(self, loss_list):
        for i in range(len(loss_list)):
            exec(f'self.loss{i+1}.update(loss_list[{i}], self.length)')

    def save_log(self, logs):
        logs['test_pred_loss'].append(self.loss1.avg)
        logs['test_input_en_loss'].append(self.loss2.avg)
        logs['test_output_en_loss'].append(self.loss3.avg)
        logs['test_physics_loss'].append(self.loss4.avg)
        logs['test_phys_encode_loss'].append(self.loss5.avg)
        logs['test_hidden_loss'].append(self.loss6.avg)
        logs['test_error_pred'].append(self.loss7.avg)
        logs['test_error_input'].append(self.loss8.avg)
        logs['test_error_output'].append(self.loss9.avg)
    
    def save_train(self, logs):
        logs['train_pred_loss'].append(self.loss1.avg)
        logs['train_input_en_loss'].append(self.loss2.avg)
        logs['train_output_en_loss'].append(self.loss3.avg)
        logs['train_physics_loss'].append(self.loss4.avg)
        logs['train_phys_encode_loss'].append(self.loss5.avg)
        logs['train_hidden_loss'].append(self.loss6.avg)
        logs['train_error_pred'].append(self.loss7.avg)
        logs['train_error_input'].append(self.loss8.avg)
        logs['train_error_output'].append(self.loss9.avg)

    def save_phys_retrain(self, logs):
        logs['retrain_phys_loss'].append(self.loss1.avg)

    def save_data_retrain(self, logs):
        logs['retrain_data_loss'].append(self.loss1.avg)