import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import random
from utils.complex import Convolve, normalize_complex, complex_mag, complex_mul, complex_mm, complex_cj
from utils.physics import WAS_xfer, Q_from_T
from utils.models import ASRmodel_bg, initialize_as_SPR_bg
class HuberLoss:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    def __call__(self, predicted, true):
        errors = torch.abs(predicted - true)
        mask = errors < self.threshold
        return (0.5 * torch.sum(torch.masked_select(errors,mask)**2) + self.threshold * (torch.sum(torch.masked_select(errors,~mask)) - (~mask).double().sum()* 0.5 * self.threshold))/errors.numel()
def main(case,out_name,depth,pixel_size,wave_len,gamma,beta,niter,eps):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mat = sio.loadmat('data/'+case+'/Holograms.mat')
    H_tensor = torch.from_numpy(mat['H'])
    del mat
    N_holo = H_tensor.shape[0] 
    M1 = H_tensor.shape[1]
    M2 = H_tensor.shape[2]
    #initialization
    z = torch.tensor(depth,dtype=torch.double)#focal depth
    T_numpy,Q_numpy = WAS_xfer(z.data.detach().numpy(), wave_len, (M1,M2), pixel_size)
    T_tensor = torch.stack([torch.from_numpy(np.real(T_numpy)),torch.from_numpy(np.imag(T_numpy))],dim=2)
    Q_numpy = Q_from_T(T_tensor.data.detach().cpu().numpy(),z.data.detach().cpu().numpy())
    Q_tensor = torch.from_numpy(Q_numpy)
    batch_size = 1
    print('Computing initialization for '+str(N_holo)+' samples with SPR...')
    W_tensor, B_tensor = initialize_as_SPR_bg(H_tensor,Q_tensor.data.detach(),z,device,0.1,beta)
    W0 = torch.stack([torch.ones((batch_size,M1,M2),dtype=torch.double),torch.zeros((batch_size,M1,M2),dtype=torch.double)],dim=3)
    B0 = torch.fft(torch.stack([torch.ones((batch_size,M1,M2),dtype=torch.double),torch.zeros((batch_size,M1,M2),dtype=torch.double)],dim=3),2)
    inputs = []
    inputs.append(batch_size)
    inputs.append(Q_tensor.data.detach().to(device))
    inputs.append(z)
    inputs.append(device)
    inputs.append(B0.data.detach().to(device))
    inputs.append(W0.data.detach().to(device))
    model = ASRmodel_bg(inputs)
    model = model.to(device)
    model.Q.requires_grad = True
    model.W.requires_grad = False
    model.B.requires_grad = False
    loss = HuberLoss(threshold = gamma)
    target = torch.zeros((batch_size,M1,M2)).double().to(device)
    filename = 'ASR_'+case+'_'+out_name
    train_loss = np.zeros((niter,1))
    for i in range(niter):
        idx_batch = random.sample([r for r in range(N_holo)],batch_size)
        with torch.no_grad():
            model.W.data.copy_(W_tensor[idx_batch,:,:,:])
            model.B.data.copy_(B_tensor[idx_batch,:])
        X_hat = model(H_tensor[idx_batch,:,:,:].to(device))
        output = loss(complex_mag(X_hat),target)
        output.backward()
        with torch.no_grad():
            model.Q.data.copy_(model.Q.data -eps*model.Q.grad.data)
            model.Q.grad.data.zero_()
            W_tensor[idx_batch,:,:,:] = normalize_complex(Convolve.apply(X_hat.data.double(),model.T.data.double())+torch.ifft(model.B.data.double(),2)).clone().detach().cpu()
            Q = complex_mul(H_tensor[idx_batch,:,:,:],W_tensor[idx_batch,:,:,:])
            aux_B = torch.fft(Q - Convolve.apply(X_hat.data.double().clone().detach().cpu(),model.T.data.double().clone().detach().cpu()),2)
            #select self.beta largest entries of abs(aux_B)
            largest_v, largest_i  = torch.topk(complex_mag(torch.squeeze(aux_B)).view(-1),beta)
            MaskB = torch.stack([torch.where((complex_mag(aux_B)>=torch.min(largest_v.view(-1))),torch.ones((1,M1,M2)).double(),torch.zeros((1,M1,M2)).double()),torch.zeros(1,M1,M2).double()],dim=aux_B.dim()-1)
            B_tensor[idx_batch,:,:,:] = complex_mul(aux_B,MaskB)
        train_loss[i] = output.cpu().detach().numpy()
        #print('Iteration #:'+str(i)+', loss: '+str(train_loss[i]))
        if np.mod(i+1,N_holo)==0:
            print('Running loss at current epoch:'+str(np.mean(train_loss[i-N_holo+1:i+1])))
    Results=dict()
    Results['losses'] = train_loss
    Results['T'] = model.T.data.clone().cpu().detach()#PSF
    torch.save(Results,filename+'_results_tensors.pt')
if __name__=='__main__':
    case = 'Baseline'
    out_name = 'test'
    #parameters for PSF initialization
    depth = 1650e-6#focal depth   
    pixel_size = 1.1*1.67e-6#pixel size
    wave_len = 340e-9#wavelength
    #ASR algorithm parameters
    gamma = 0.1#Huber loss parameter
    beta = 1#Sparsity of background in frequency domain
    eps = 0.1#learning rate
    niter = 10000#number of iterations
    print('Running ASR for '+case+' data, with Huber parameter \gamma='+str(gamma)+' and learning rate \eps='+str(eps)+' for '+str(niter)+' iterations')
    print('Initial PSF obtained with WAS approximation (focal depth '+str(int(depth*1e6))+'um , pixel size '+str(int(pixel_size*1e9))+'nm, and wave length '+str(int(wave_len*1e9))+'nm).')
    main(case,out_name,depth,pixel_size,wave_len,gamma,beta,niter,eps)
