import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LISTA(nn.Module):
    def __init__(self, n, m, T=3, lambd=1.0, D=None):
        super(LISTA, self).__init__()
        self.n, self.m = n, m
        self.D = D
        self.T = T  # ISTA Iterations
        self.lambd = lambd  # Lagrangian Multiplier
        self.A = nn.Linear(n, m, bias=False)  # Weight Matrix
        self.B = nn.Linear(m, m, bias=False)  # Weight Matrix
        # ISTA Stepsizes eta = 1/L
        self.etas = nn.Parameter(torch.ones(T + 1, 1, 1), requires_grad=True) #1/L trainable param
        self.gammas = nn.Parameter(torch.ones(T + 1, 1, 1), requires_grad=True) 
        # Initialization
        if D is not None:
            self.A.weight.data = D.t()
            self.B.weight.data = D.t() @ D
        self.reinit_num = 0  # Number of re-initializations

    def shrink(self, x, eta):
        return eta * F.softshrink(x / eta, lambd=self.lambd)

    def forward(self, y, D=None):
        # initial value of x
        x = self.shrink(self.gammas[0, :, :] * self.A(y), self.etas[0, :, :])   # 1/L*D^T*b
        for i in range(1, self.T + 1):
            # x = self.shrink(self.gammas[i, :, :] * (self.B(x) + self.A(y)), self.etas[i, :, :]) 
            x = self.shrink(x - self.gammas[i, :, :] * self.B(x) + self.gammas[i, :, :] * self.A(y),self.etas[i, :, :])
            print(x)
        return x

    # def reinit(self):
    #     reinit_num = self.reinit_num + 1
    #     self.__init__(n=self.n, m=self.m, T=self.T, lambd=self.lambd, D=self.D)
    #     self.reinit_num = reinit_num



if __name__=="__main__":
    model=LISTA(2,2)
    model_gpu = model.to(device)
    summary(model_gpu, (2,2))