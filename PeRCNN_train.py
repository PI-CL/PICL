import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as scio
import time
import os
from torch.utils.data import Dataset, DataLoader
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_default_dtype(torch.float32)

torch.manual_seed(66)
np.random.seed(66)

lap_2d_op = [[[[    0,   0, -1/12,   0,     0],
               [    0,   0,   4/3,   0,     0],
               [-1/12, 4/3,   - 5, 4/3, -1/12],
               [    0,   0,   4/3,   0,     0],
               [    0,   0, -1/12,   0,     0]]]]

class upscaler(nn.Module):
    ''' Upscaler to convert low-res to high-res initial state '''

    def __init__(self):
        super(upscaler, self).__init__()
        self.layers = []
        self.layers.append(
            nn.ConvTranspose2d(3, 8, kernel_size=6, padding=4 // 2, stride=2, output_padding=1, bias=True))
        self.layers.append(torch.nn.Sigmoid())
        self.layers.append(
            nn.ConvTranspose2d(8, 8, kernel_size=7, padding=4 // 2, stride=2, output_padding=1, bias=True))
        self.layers.append(nn.Conv2d(8, 3, 1, 1, padding=0, bias=True))
        self.convnet = torch.nn.Sequential(*self.layers)

    def forward(self, h):
        return self.convnet(h)

class RCNNCell(nn.Module):
    ''' Recurrent convolutional neural network Cell '''
   
    def __init__(self, input_channels, hidden_channels, input_kernel_size):

        super(RCNNCell, self).__init__()

        # the initial parameters
        self.input_channels = input_channels  # no use, always 2
        self.hidden_channels = hidden_channels
        self.input_kernel_size = 5
        self.input_stride = 1

        self.dx = 0.01
        self.dt = 0.5
        self.mu_up = 0.021  # set to the mu of NSWE
        # Design the laplace_u term # [-1, 1]
        np.random.seed(1234)
        self.CA = torch.nn.Parameter(torch.tensor((np.random.rand()-0.5)*2, dtype=torch.float32), requires_grad=True)
        self.CB = torch.nn.Parameter(torch.tensor((np.random.rand()-0.5)*2, dtype=torch.float32), requires_grad=True)

        # padding_mode='replicate' not working for the test
        self.W_laplace = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=0, bias=False)
        self.W_laplace.weight.data = 1/self.dx**2*torch.tensor(lap_2d_op, dtype=torch.float32)
        self.W_laplace.weight.requires_grad = False

        # Nonlinear term for u (up to 3rd order)
        self.Wh1_u = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh2_u = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh3_u = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh4_u = nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1,
                               stride=1, padding=0, bias=True)
        # Nonlinear term for v (up to 3rd order)
        self.Wh1_v = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh2_v = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh3_v = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh4_v = nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1,
                               stride=1, padding=0, bias=True)
        # Nonlinear term for h (up to 3rd order)
        self.Wh1_h = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh2_h = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh3_h = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=1,
                               stride=self.input_stride, padding=0, bias=True, )
        self.Wh4_h = nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=1,
                               stride=1, padding=0, bias=True)

        # initialize filter's wweight and bias
        self.filter_list = [self.Wh1_u, self.Wh2_u, self.Wh3_u, self.Wh4_u, self.Wh1_v, self.Wh2_v, self.Wh3_v, self.Wh4_v, self.Wh1_h, self.Wh2_h, self.Wh3_h, self.Wh4_h]
        self.init_filter(self.filter_list, c=0.02)

    def init_filter(self, filter_list, c):
        '''
        :param filter_list: list of filter for initialization
        :param c: constant multiplied on Xavier initialization
        '''
        for filter in filter_list:
            # Xavier initialization and then scale
            torch.nn.init.xavier_uniform_(filter.weight)
            filter.weight.data = c*filter.weight.data
            # filter.weight.data.uniform_(-c * np.sqrt(1 / (5 * 5 * 16)), c * np.sqrt(1 / (5 * 5 * 16)))
            if filter.bias is not None:
                filter.bias.data.fill_(0.0)

    def forward(self, h):

        # periodic padding, can also be achieved using 'circular' padding (NSWE is also the perodic boundary condition)
        h_pad = torch.cat((    h[:, :, :, -2:],     h,     h[:, :, :, 0:2]), dim=3)
        h_pad = torch.cat((h_pad[:, :, -2:, :], h_pad, h_pad[:, :, 0:2, :]), dim=2)
        
        u_pad = h_pad[:, 0:1, ...]  
        v_pad = h_pad[:, 1:2, ...]
        p_pad = h_pad[:, 1:2, ...]
        u_prev = h[:, 0:1, ...]   
        v_prev = h[:, 1:2, ...]
        p_prev = h[:, 1:2, ...]

        u_res = self.mu_up*torch.sigmoid(self.CA)*self.W_laplace(u_pad) + self.Wh4_u( self.Wh1_u(h)*self.Wh2_u(h)*self.Wh3_u(h) )
        v_res = self.mu_up*torch.sigmoid(self.CB)*self.W_laplace(v_pad) + self.Wh4_v( self.Wh1_v(h)*self.Wh2_v(h)*self.Wh3_v(h) )
        p_res = self.Wh4_h( self.Wh1_h(h)*self.Wh2_h(h)*self.Wh3_h(h) )
        u_next = u_prev + u_res * self.dt
        v_next = v_prev + v_res * self.dt
        p_next = p_prev + p_res * self.dt
        ch = torch.cat((u_next, v_next, p_next), dim=1)
        
        return ch, ch

    def init_hidden_tensor(self, prev_state):
        return prev_state.cuda()


class RCNN(nn.Module):

    ''' Recurrent convolutional neural network layer '''

    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                       step=1, effective_step=[1]):

        super(RCNN, self).__init__()
        
        # input channels of layer includes input_channels and hidden_channels of cells 
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = 1
        self.input_kernel_size = input_kernel_size
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.init_state = []

        # Upconv as initial state generator
        self.UpconvBlock = upscaler().cuda()

        name = 'crnn_cell'
        cell = RCNNCell(
            input_channels = self.input_channels,
            hidden_channels = self.hidden_channels,
            input_kernel_size = self.input_kernel_size)

        setattr(self, name, cell)
        self._all_layers.append(cell)


    def forward(self, init_state_low):

        self.init_state = self.UpconvBlock(init_state_low)
        internal_state = []
        outputs = [self.init_state]
        second_last_state = []
        
        for step in range(self.step):
            name = 'crnn_cell'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_state
                internal_state = h

            # forward
            h = internal_state
            # hidden state + output
            h, o = getattr(self, name)(h) 
            
            internal_state = h

            if step == (self.step - 2):
                #  last output is a dummy for central FD
                second_last_state = internal_state.clone()

            # after many layers output the result save at time step t
            # if step in self.effective_step:
            
            outputs.append(o)

        return outputs, second_last_state


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=5, name=''):
        '''
        :param DerFilter: constructed derivative filter, e.g. Laplace filter
        :param deno: resolution of the filter, used to divide the output, e.g. c*dt, c*dx or c*dx^2
        :param kernel_size:
        :param name: optional name for the operator
        '''
        super(Conv2dDerivative, self).__init__()
        self.deno = deno  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.deno


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, deno, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.deno = deno  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.deno


class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dt = (1.0/2), dx = (1.0/100)):

        '''
        Construct the derivatives, X = Width, Y = Height      

        '''

        self.dt = dt
        self.dx = dx
       
        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter = lap_2d_op,
            deno = (dx**2),
            kernel_size = 5,
            name = 'laplace_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 1, 0]]],
            deno = (dt*1),
            kernel_size = 3,
            name = 'partial_t').cuda()

    def get_phy_Loss(self, output):  # Staggered version
        '''
        Calculate the physics loss

        Args:
        -----
        output: tensor, dim:
            shape: [time, channel, height, width]

        Returns:
        --------
        f_u: float
            physics loss of u

        f_v: float
            physics loss of v
        '''

        # spatial derivatives
        laplace_u = self.laplace(output[0:-2, 0:1, :, :])  # 201x1x128x128
        laplace_v = self.laplace(output[0:-2, 1:2, :, :])  # 201x1x128x128

        # temporal derivatives - u
        u = output[:, 0:1, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx*leny, 1, lent)
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent-2)
        u_t = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        # temporal derivatives - v
        v = output[:, 1:2, 2:-2, 2:-2]
        v_conv1d = v.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        v_conv1d = v_conv1d.reshape(lenx*leny, 1, lent)
        v_t = self.dt(v_conv1d)  # lent-2 due to no-padding
        v_t = v_t.reshape(leny, lenx, 1, lent-2)
        v_t = v_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u = output[0:-2, 0:1, 2:-2, 2:-2]  # [step, c, height(Y), width(X)]
        v = output[0:-2, 1:2, 2:-2, 2:-2]  # [step, c, height(Y), width(X)]

        # make sure the dimensions consistent
        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        # gray scott eqn
        Du = 2e-5
        Dv = Du/4
        f = 1/25
        k = 3/50

        f_u = (Du*laplace_u - u*(v**2) + f*(1-u) - u_t)/1
        f_v = (Dv*laplace_v + u*(v**2) - (f+k)*v - v_t)/1

        return f_u, f_v

def get_ic_loss(model, init_state_low):
    Upconv = model.UpconvBlock
    # init_state_low = model.init_state_low
    init_state_bicubic = F.interpolate(init_state_low, (32, 32), mode='bicubic')
    mse_loss = nn.MSELoss()
    init_state_pred = Upconv(init_state_low)
    loss_ic = mse_loss(init_state_pred, init_state_bicubic)
    return loss_ic

def loss_gen(output, loss_func):
    '''calculate the phycis loss'''
    
    # Padding x axis due to periodic boundary condition
    output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)
    output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2)

    # get physics loss
    mse_loss = nn.MSELoss()
    f_u, f_v = loss_func.get_phy_Loss(output)
    loss = mse_loss(f_u, torch.zeros_like(f_u).cuda()) + \
           mse_loss(f_v, torch.zeros_like(f_v).cuda())
    return loss

def pretrain_upscaler(Upconv, train_dataloader, epoch=4000):
    '''
    :param Upconv: upscalar model
    :param init_state_low: low resolution measurement
    :return:
    '''
    optimizer = optim.Adam(Upconv.parameters(), lr = 0.02)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.99)
    mse_loss = nn.MSELoss()
    for epoch in range(epoch):
        
        for x_train, y_train in train_dataloader:
            optimizer.zero_grad()
            init_state_bicubic = F.interpolate(x_train, (32, 32), mode='bicubic')
            
            init_state_pred = Upconv(x_train)
            loss = mse_loss(init_state_pred, init_state_bicubic)
            loss.backward(retain_graph=True)
            print('[%d] loss: %.9f' % ((epoch+1), loss.item()))
            optimizer.step()
            scheduler.step()
            
            
def test(model, test_dataloader, n_iters, time_batch_size, learning_rate, dt, dx, cont=True):
    # define some parameters
    test_loss_list = []
    batch_loss, phy_loss, ic_loss, data_loss, val_loss = [0]*5

    with torch.no_grad():
        for x_test, y_test in test_dataloader:
            
            # One single batch
            output, second_last_state = model(x_test)  # output is a list
            output = torch.cat(tuple(output), dim=0)  
            mse_loss = nn.MSELoss()
            pred, gt = output[-32:, :, ::5, ::5], y_test.cuda()
            idx = int(pred.shape[0]*0.9)
            pred_tra, pred_val = pred, pred     # prediction
            gt_tra,   gt_val   =   gt,   gt     # ground truth
            loss_data  = rel_error(pred_tra, gt_tra).mean()         # data loss
            loss_valid = mse_loss(pred_val, gt_val)
            loss_ic   = get_ic_loss(model, x_test)

            data_loss, val_loss  = loss_data.item(), loss_valid.item()

            print('test_data_loss: %.7f' % (data_loss))
            test_loss_list.append(data_loss)

    return test_loss_list

def rel_error(x, _x):
    # calculate relative data in inference
    if len(x.shape)==1:
        x = x.reshape(1, -1)
        _x = _x.reshape(1, -1)
    else:
        B = x.size(0)
        x, _x = x.reshape(B, -1), _x.reshape(B, -1)
    return torch.norm(x - _x, 2, dim=1)/torch.norm(_x, 2, dim=1)

def train(model, train_dataloader, n_iters, time_batch_size, learning_rate, dt, dx, cont=True):
    # define some parameters
    train_loss_list = []
    # model
    if cont:
        model, optimizer, scheduler = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.985)

    for epoch in range(n_iters):
        for x_train, y_train in train_dataloader:

            optimizer.zero_grad()
            num_time_batch = 1
            batch_loss, phy_loss, ic_loss, data_loss, val_loss = [0]*5
            # One single batch
            output, second_last_state = model(x_train)  # output is a list
            output = torch.cat(tuple(output), dim=0) 
            
            mse_loss = nn.MSELoss()

            pred, gt = output[-32:, :, ::5, ::5], y_train.cuda()

            pred_tra, pred_val = pred, pred     # prediction
            gt_tra,   gt_val   =   gt,   gt     # ground truth
            loss_data  = mse_loss(pred_tra, gt_tra)         # data loss
            loss_valid = mse_loss(pred_val, gt_val)
            loss_ic   = get_ic_loss(model, x_train)

            loss = 40*loss_data + 0.25*loss_ic
            loss.backward(retain_graph=True)
            batch_loss += loss.item()
            ic_loss, data_loss, val_loss  = loss_ic.item(), loss_data.item(), loss_valid.item()
            optimizer.step()
            scheduler.step()
            # print loss in each epoch
            print('[%d/%d %d%%] loss: %.7f, ic_loss: %.7f, data_loss: %.7f, val_loss: %.7f' % ((epoch+1), n_iters, ((epoch+1)/n_iters*100.0),
              batch_loss, ic_loss, data_loss, val_loss))
            train_loss_list.append(batch_loss)

    return train_loss_list

def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./model/checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.98)
    return model, optimizer, scheduler


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def postProcess_2x3(output, truth, low_res, xmin, xmax, ymin, ymax, num, fig_save_path):
    ''' num: Number of time step
    '''
    x = np.linspace(-0.5, 0.5, 101)
    y = np.linspace(-0.5, 0.5, 101)
    x_star, y_star = np.meshgrid(x, y)
    u_low_res, v_low_res = low_res[num, 0, ...], low_res[num, 1, ...]
    u_low_res, v_low_res = np.kron(u_low_res.detach().cpu().numpy(), np.ones((4, 4))), \
                           np.kron(v_low_res.detach().cpu().numpy(), np.ones((4, 4)))
    u_low_res, v_low_res = np.concatenate((u_low_res, u_low_res[:, 0:1]), axis=1), \
                           np.concatenate((v_low_res, v_low_res[:, 0:1]), axis=1)
    u_low_res, v_low_res = np.concatenate((u_low_res, u_low_res[0:1, :]), axis=0), \
                           np.concatenate((v_low_res, v_low_res[0:1, :]), axis=0)
    u_star, v_star = truth[num, 0, ...], truth[num, 1, ...]
    u_pred, v_pred = output[num, 0, :, :].detach().cpu().numpy(), \
                     output[num, 1, :, :].detach().cpu().numpy()
    #
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    #
    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('u (PeCRNN)')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('u (Ref.)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 2].scatter(x_star, y_star, c=u_low_res, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[0, 2].axis('square')
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_title('u (Meas.)')
    fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('v (PeCRNN)')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('v (Ref.)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 2].scatter(x_star, y_star, c=v_low_res, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[1, 2].axis('square')
    ax[1, 2].set_xlim([xmin, xmax])
    ax[1, 2].set_ylim([ymin, ymax])
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].set_title('v (Meas.)')
    fig.colorbar(cf, ax=ax[1, 2], fraction=0.046, pad=0.04)
    #
    plt.savefig(fig_save_path + 'uv_comparison_'+str(num).zfill(3)+'.png')
    plt.close('all')

def postProcess_2x2(output, truth, xmin, xmax, ymin, ymax, num, fig_save_path):
    ''' num: Number of time step
    '''
    x = np.linspace(-0.5, 0.5, 101)
    y = np.linspace(-0.5, 0.5, 101)
    x_star, y_star = np.meshgrid(x, y)
    u_star, v_star = truth[num, 0, ...], truth[num, 1, ...]
    u_pred, v_pred = output[num, 0, :, :].detach().cpu().numpy(), \
                     output[num, 1, :, :].detach().cpu().numpy()
    #
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    #
    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('u (PeCRNN)')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=np.abs(u_star-u_pred), alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1*0.2)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('u (Error)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('v (PeCRNN)')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=np.abs(v_star-v_pred), alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1*0.2)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('v (Error)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    plt.savefig(fig_save_path + 'error_'+str(num).zfill(3)+'.png')
    plt.close('all')

def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)

def add_noise(truth, pec=0.05):
    from torch.distributions import normal
    assert truth.shape[1]==2
    torch.manual_seed(66)
    uv = [truth[:,0:1,:,:], truth[:,1:2,:,:]]
    uv_noi = []
    for truth in uv:
        n_distr = normal.Normal(0.0, 1.0)
        R = n_distr.sample(truth.shape)
        std_R = torch.std(R)          # std of samples
        std_T = torch.std(truth)
        noise = R*std_T/std_R*pec
        uv_noi.append(truth+noise)
    return torch.cat(uv_noi, dim=1)

def normal(data_run_train, data_run_test, logs):

    u_mean, u_std = data_run_train[:,0,:,:].mean(), data_run_train[:,0,:,:].std()
    v_mean, v_std = data_run_train[:,1,:,:].mean(), data_run_train[:,1,:,:].std()
    h_mean, h_std = data_run_train[:,2,:,:].mean(), data_run_train[:,2,:,:].std()

    data_run_train[:,0,:,:] = (data_run_train[:,0,:,:] - u_mean) / u_std
    data_run_train[:,1,:,:] = (data_run_train[:,1,:,:] - v_mean) / v_std
    data_run_train[:,2,:,:] = (data_run_train[:,2,:,:] - h_mean) / h_std

    data_run_test[:,0,:,:] = (data_run_test[:,0,:,:] - u_mean) / u_std
    data_run_test[:,1,:,:] = (data_run_test[:,1,:,:] - v_mean) / v_std
    data_run_test[:,2,:,:] = (data_run_test[:,2,:,:] - h_mean) / h_std

    return data_run_train, data_run_test, logs

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

class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor

        self.target_tensor = target_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')

    parser.add_argument('-dp', '--data_path', default='PICL', type=str, help='data path name')
    parser.add_argument('-lf', '--logs_fname', default='nswe', type=str, help='logs file name')
    parser.add_argument('-dc', '--dict', default='nswe', type=str, help='dict name')
    parser.add_argument('-dr', '--data_rate', default=1, type=float, help='logs file name')
    
    parser.add_argument('-L', '--L', default=4, type=int, help='the number of layers')
    parser.add_argument('-m', '--modes', default=12, type=int, help='the number of modes of Fourier layer')
    parser.add_argument('-w', '--width', default=32, type=int, help='the number of width of FNO layer')
    parser.add_argument('--drop', default=0.1, type=float, help='Dropout of unet')
    parser.add_argument('-ch', '--channel', default=32, type=int, help='channels of unet')
    
    parser.add_argument('--phys_gap', default=40, type=int, help = 'Number of gap of Phys')
    parser.add_argument('--phys_epochs', default=10, type=int, help = 'Number of Phys Epochs')
    parser.add_argument('--phys_steps', default=2, type=int, help = 'Number of Phys Steps')
    parser.add_argument('--phys_scale', default=0.1, type=float, help = 'Number of Phys Scale')
    parser.add_argument('--phys_random_select', default=False, type=bool, help = 'Whether random select')

    parser.add_argument('--batch_size', default=32, type=int, help = 'batch size')
    parser.add_argument('--epochs', default=10, type=int, help = 'Number of Epochs')
    parser.add_argument('--re_epochs', default=1, type=int, help = 'Number of Phys Retrain Epochs')
    parser.add_argument('--tune_epochs', default=10, type=int, help = 'Number of Data Retrain Epochs')
    parser.add_argument('--loop', default=2, type=int, help = 'Number of Total Loop')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--step_size', default=100, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler factor')
    parser.add_argument('--gpu', default=0, type=int, help='device number')

    parser.add_argument('-tg', '--tg', default=1, type=int, help = 'time gap')
    parser.add_argument('-Ng', '--Ng', default=1, type=int, help = 'N gap')
    parser.add_argument('-l1', '--lambda1', default=1, type=float, help='weight of data losses1')
    parser.add_argument('-l2', '--lambda2', default=0, type=float, help='weight of encode losses1')
    parser.add_argument('-l3', '--lambda3', default=0, type=float, help='weight of encode losses2')
    parser.add_argument('-l4', '--lambda4', default=100, type=float, help='weight of physics losses')
    parser.add_argument('-l5', '--lambda5', default=0, type=float, help='weight of physics losses of encoder')
    parser.add_argument('-l6', '--lambda6', default=0, type=float, help='weight of hidden loss')
    parser.add_argument('-fc', '--f_channels', default=0, type=int, help='channels of f encode')
    
    parser.add_argument('--re', default=0.1, type=float, help='re number')
    parser.add_argument('--lx', default=2.05, type=float, help='Lx')
    parser.add_argument('--ly', default=2.05, type=float, help='Ly')
    parser.add_argument('--en_dim', default=3, type=int, help='the input dimension of first layer of encoder')
    parser.add_argument('--gap_sample', default=5, type=int, help='sample gap from origin uvp in space')
    parser.add_argument('--obs_num', default=7, type=int, help='width and length of observation')
    parser.add_argument('--state_num', default=32, type=int, help='width and length of of state')
    parser.add_argument('--recall_size', default=1, type=int, help='recall history observation')

    parser.add_argument('--with_next', default=1, type=int, help='with ut+1 or not')
    parser.add_argument('--start_phys', default=500, type=int, help='with ut+1 or not')
    parser.add_argument('--seed', default=333, type=int, help='set random seed')
    parser.add_argument('--traj_str_tr', default=0, type=int, help='select the train set start')
    parser.add_argument('--traj_end_tr', default=450, type=int, help='select the train set end')
    parser.add_argument('--traj_str_ts', default=450, type=int, help='select the test set start')
    parser.add_argument('--traj_end_ts', default=500, type=int, help='select the test set end')

    parser.add_argument('-trsta', '--train_str', default=0, type=int, help='start position train')
    parser.add_argument('-trend', '--train_end', default=14, type=int, help='end position train')
    parser.add_argument('-tssta', '--test_str', default=0, type=int, help='start position test')
    parser.add_argument('-tsend', '--test_end', default=14, type=int, help='end position test')

    parser.add_argument('--unlabel_size', default=1/3, type=float, help='select the unlabel set scale')

    parser.add_argument('--ups', default=1, type=int, help='how to upsample in unet')
    parser.add_argument('--full', default=False, type=int, help='if the trans phys loss backward to encoder')

    return parser.parse_args(argv)


if __name__ == '__main__':
    
    args = get_args()
    
    data_path = 'data/nswe_coarse'  # the low-resolution NSWE data
    
    logs = dict()
    logs['args'] = args
    
    data_run = torch.load(data_path)   # uvp, traj, t, ox*oy

    num_obs = args.obs_num
    num_state = args.state_num
    gap = args.gap_sample
    trsta = args.train_str
    trend = args.train_end
    tssta = args.test_str
    tsend = args.test_end
    traj_str_tr = args.traj_str_tr
    traj_end_tr = args.traj_end_tr
    traj_str_ts = args.traj_str_ts
    traj_end_ts = args.traj_end_ts
    unlabel_size = args.unlabel_size
    
    ################# the same data preprocessing to our framework #################
    data_train = data_run[:,traj_str_tr:traj_end_tr]  # uvp, traj, t, ox*oy
    data_test = data_run[:,traj_str_ts:traj_end_ts]
    data_train = data_train.permute(1,0,2,3)  # traj, uvp, t, ox*oy
    data_train = data_train[:,:,trsta:trend]
    data_test = data_test.permute(1,0,2,3)  # traj, uvp, t, ox*oy
    data_test = data_test[:,:,tssta:tsend]
    data_run_train, data_run_test, logs = normal(data_train, data_test, logs)  # traj, uvp, t, ox*oy
    data_run_train = data_run_train.permute(1,0,2,3)  # uvp, traj, t, ox*oy
    data_run_test = data_run_test.permute(1,0,2,3)
    data_train_re = data_repeat(data_run_train, num_obs, num_state, gap)  # uvp, t, traj, x, y
    data_test_re = data_repeat(data_run_test, num_obs, num_state, gap)
    data_train_re = data_train_re.permute(2,0,1,3,4)  # traj, uvp, t, x, y
    data_test_re = data_test_re.permute(2,0,1,3,4)
    data_run = data_train_re
    data_test = data_test_re

    tg = args.tg
    Ng = args.Ng 
    batch_size=args.batch_size
    dt = tg*0.01
    N0, nt, nx, ny = data_run.shape[0], data_run.shape[2], data_run.shape[3], data_run.shape[4]
    shape = [nx, ny]
    
    input_data =data_run[:, :, :-1, :]  # traj, uvp*2, t, x, y
    output_data = data_run[:, :, 1:, :]
    input_data_test = data_test[:, :, :-1, :]
    output_data_test = data_test[:, :, 1:, :]
    train_data = input_data.permute(0,2,1,3,4) 
    train_labels = output_data.permute(0,2,1,3,4) 
    train_data = train_data[:,:,:,::5,::5].cuda().to(torch.float32)
    train_labels = train_labels[:,:,:,::5,::5].cuda().to(torch.float32)

    train_data = train_data[:-int(N0*unlabel_size)]
    train_labels = train_labels[:-int(N0*unlabel_size)]
    
    train_data = train_data.reshape(-1, train_data.shape[2], train_data.shape[3], train_data.shape[4])
    train_labels = train_labels.reshape(-1, train_labels.shape[2], train_labels.shape[3], train_labels.shape[4])

    train_dataset = MyDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    test_data = input_data_test.permute(0,2,1,3,4) 
    test_labels = output_data_test.permute(0,2,1,3,4) 
    test_data = test_data[:,:,:,::5,::5].cuda().to(torch.float32)
    test_labels = test_labels[:,:,:,::5,::5].cuda().to(torch.float32)
    
    test_data = test_data.reshape(-1, test_data.shape[2], test_data.shape[3], test_data.shape[4])
    test_labels = test_labels.reshape(-1, test_labels.shape[2], test_labels.shape[3], test_labels.shape[4])
    test_dataset = MyDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    time_steps = 1   
    dt = 0.01
    dx = 1.0/32
    dy = 1.0/32
    time_batch_size = time_steps
    steps = 1  
    effective_step = list(range(0, steps))
    n_iters = 6000
    learning_rate = 1e-3
    # save_path = './model/'

    model = RCNN(
        input_channels = 3, 
        hidden_channels = 8,
        input_kernel_size = 5,
        step = steps, 
        effective_step = effective_step).cuda()

    # train the model
    start = time.time()
    cont = False   # if continue training (or use pretrained model), set cont=True
    if not cont:
        pretrain_upscaler(model.UpconvBlock, train_dataloader)
    train_loss_list = train(model, train_dataloader, n_iters, time_batch_size, learning_rate, dt, dx, cont=cont)
    end = time.time()

    print('The training time is: ', (end-start))

    # Do the forward inference
    test_loss_list = test(model, test_dataloader, n_iters, time_batch_size, learning_rate, dt, dx, cont=cont)
    print('Mean test loss', np.mean(np.array(test_loss_list)))
    
    print('save model!!!')
    torch.save({'model_state_dict': model.state_dict()}, './model/checkpoint.pt')
    
    
        
