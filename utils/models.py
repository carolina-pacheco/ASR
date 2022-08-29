import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import gradcheck
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import random
import time
from utils.complex import Convolve, normalize_complex, complex_mag, complex_mul, complex_mm, complex_cj, complex_div
from utils.physics import WAS_xfer, Q_from_T
import gc
import scipy.io as sio

class expjQz(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Qxy,z):
        T_np = np.exp(1.0j/637e-9*z.data.detach().cpu().numpy()*Qxy.data.detach().cpu().numpy())
        kernel = torch.stack([torch.from_numpy(np.real(T_np)),torch.from_numpy(np.imag(T_np))],dim=2).to('cuda')
        ctx.save_for_backward(kernel,z.to('cuda')/637e-9,Qxy/637e-9)
        return kernel
    @staticmethod
    def backward(ctx,grad_output):
        kernel,z_saved,Q_saved = ctx.saved_tensors
        grad_z =   torch.sum(Q_saved*(grad_output[:,:,1]*kernel[:,:,0]-grad_output[:,:,0]*kernel[:,:,1]))
        grad_Qxy = z_saved*(grad_output[:,:,1]*kernel[:,:,0]-grad_output[:,:,0]*kernel[:,:,1])
        return grad_Qxy, grad_z
class SPR_W_B(nn.Module):
    def __init__(self,list_of_params):
        super(SPR_W_B,self).__init__()
        self.Q = list_of_params[1]
        self.N = list_of_params[0]
        self.M1 = self.Q.shape[0]
        self.M2 = self.Q.shape[1]
        self.H = torch.empty((self.N,self.M1,self.M2,2),requires_grad=False)
        self.niter = list_of_params[2]
        self.lam = list_of_params[3]
        self.z = list_of_params[4]
        self.beta = list_of_params[5]
        self.T = torch.empty((self.M1,self.M2,2),requires_grad=False)
        self.W0 = torch.stack([torch.ones((self.N,self.M1,self.M2),requires_grad=False).double(),torch.zeros((self.N,self.M1,self.M2),requires_grad=False).double()],dim=3).to('cuda')
        self.X0 = torch.stack([torch.zeros((self.N,self.M1,self.M2),requires_grad=False).double(),torch.zeros((self.N,self.M1,self.M2),requires_grad=False).double()],dim=3).to('cuda')
        self.B0 = torch.fft(torch.stack([torch.ones(self.N,self.M1,self.M2).double(),torch.zeros((self.N,self.M1,self.M2)).double()],dim=3).to('cuda'),2)
        self.ones = torch.stack([torch.ones((self.N,self.M1,self.M2),dtype=torch.double,requires_grad=False),torch.zeros((self.N,self.M1,self.M2),dtype=torch.double,requires_grad=False)],dim=2).to('cuda')
    def forward(self,input):
        self.T = expjQz.apply(self.Q,self.z)
        self.H = input.to('cuda')
        aux_B = self.B0
        aux_W = self.W0
        aux_X = self.X0
        ini = time.time()
        for i in range(self.niter):
            aux_W = normalize_complex(Convolve.apply(aux_X,self.T)+torch.ifft(aux_B,2))
            Q = complex_mul(self.H,aux_W) 
            aux_B = torch.fft(Q - Convolve.apply(aux_X,self.T),2)  
            largest_v, largest_i  = torch.topk(complex_mag(torch.squeeze(aux_B)).view(-1),self.beta)
            MaskB = torch.stack([torch.where((complex_mag(aux_B)>=torch.min(largest_v.view(-1))),torch.ones((self.N,self.M1,self.M2)).double().to('cuda'),torch.zeros((self.N,self.M1,self.M2)).double().to('cuda')),torch.zeros(self.N,self.M1,self.M2).double().to('cuda')],dim=aux_B.dim()-1)
            aux_B = complex_mul(aux_B,MaskB)
            input_1 = Convolve.apply(Q-torch.ifft(aux_B,2),complex_cj(self.T))
            Mask = torch.stack([torch.where((complex_mag(input_1)<=self.lam),torch.zeros((self.N,self.M1,self.M2)).double().to('cuda'),torch.ones((self.N,self.M1,self.M2)).double().to('cuda')),torch.zeros(self.N,self.M1,self.M2).double().to('cuda')],dim=input_1.dim()-1)
            if i < self.niter-1:
                aux_X = complex_mul(input_1,Mask)
            else:
                aux_X = input_1
            gc.collect()
            torch.cuda.empty_cache()
        return aux_W, torch.squeeze(aux_B)

class ASRmodel_bg(nn.Module):
    def __init__(self,list_of_params):
        super(ASRmodel_bg,self).__init__()
        self.device = list_of_params[3]
        self.Q = list_of_params[1]
        self.N = list_of_params[0]
        self.M1 = self.Q.shape[0]
        self.M2 = self.Q.shape[1]
        self.H = torch.empty((self.N,self.M1,self.M2,2),requires_grad=False)
        self.z = list_of_params[2]
        self.T = torch.empty((self.M1,self.M2,2),requires_grad=False)
        self.B = list_of_params[4]
        self.W = list_of_params[5]
    def forward(self,input):
        self.T = expjQz.apply(self.Q,self.z)
        self.H = input.to(self.device)
        return Convolve.apply(complex_mul(self.H,self.W)-torch.ifft(self.B,2),complex_cj(self.T))

def initialize_as_SPR_bg(H,Q,z,device,lam,beta):
    N = H.shape[0]
    inputs = []
    inputs.append(1)#batchsize
    inputs.append(Q.to(device))
    inputs.append(10)#niter
    inputs.append(lam)#lambda
    inputs.append(z)
    inputs.append(beta)
    model = SPR_W_B(inputs)
    model = model.to(device)
    B = torch.zeros((N,H.shape[1],H.shape[2],2))
    W = torch.zeros(H.shape)
    for i in range(N):
        W[i,:,:,:],B[i,:,:,:] = model(torch.unsqueeze(H[i,:,:,:],dim=0).to(device))
    return W.data.double().detach(), B.data.double().detach()

