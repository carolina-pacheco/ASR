import torch
from torch.autograd import gradcheck
def normalize_complex(A):
    Mag = torch.sqrt(A[...,0]**2+A[...,1]**2)
    return torch.div(A,torch.stack([Mag,Mag],dim=A.dim()-1)+1e-20)
def complex_mag(A):
    return torch.sqrt(A[...,0]**2+A[...,1]**2)
def complex_phase(A):
    return torch.atan2(A[...,1],A[...,0])
def complex_mul(A,B):
#pointwise multiplication of 2D complex tensors shape [M,N,2]
    C = torch.empty_like(A,dtype=torch.float64,requires_grad=False)   
    C[...,0]=A[...,0]*B[...,0]-A[...,1]*B[...,1]
    C[...,1]=A[...,0]*B[...,1]+A[...,1]*B[...,0]
    return C   

def complex_div(A,B):
#pointwise division of 2D complex tensors shape [M,N,2]
    C = torch.empty_like(A,dtype=torch.float64,requires_grad=False)
    C[...,0]=(A[...,0]*B[...,0]+A[...,1]*B[...,1])/(B[...,0]**2+B[...,1]**2)
    C[...,1]=(A[...,1]*B[...,0]-A[...,0]*B[...,1])/(B[...,0]**2+B[...,1]**2)
    return C

def complex_mm(A,B):
#matrix multiplication of 2D complex tensors shape [M,N,2]
    if A.dim()>3:
        C = torch.empty(A.shape[0],A.shape[1],B.shape[-2],2,dtype=torch.float64,requires_grad=False)
    else:
        C = torch.empty(A.shape[0],B.shape[-2],2,dtype=torch.float64,requires_grad=False)
    C[...,0]=torch.matmul(A[...,0],B[...,0])-torch.matmul(A[...,1],B[...,1])
    C[...,1]=torch.matmul(A[...,0],B[...,1])+torch.matmul(A[...,1],B[...,0])
    return C.to('cuda')
def complex_cj(A):
    #complex conjugate
    C = torch.empty_like(A,dtype=torch.float64,requires_grad=False)
    C[...,0]=A[...,0]
    C[...,1]=-1*A[...,1]
    return C
class Convolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, T):
        #input in spatial domain, T in Fourier domain
        Xfourier = torch.fft(input,2)
        ctx.save_for_backward(T,Xfourier)
        output = torch.ifft(complex_mul(Xfourier,T.repeat(input.shape[0],1,1,1)),2)
        return output
    @staticmethod
    def backward(ctx,grad_output):
        Tkernel, X_F = ctx.saved_tensors
        gradout_Fourier = torch.fft(grad_output,2)
        grad_input = torch.ifft(complex_mul(gradout_Fourier,complex_cj(Tkernel.repeat(X_F.shape[0],1,1,1))),2)
        grad_T = torch.sum(complex_mul(gradout_Fourier,complex_cj(X_F)),dim=0)/(Tkernel.shape[0]**2)
        return grad_input, grad_T

