
import torch 


def ackley_f(x):
    # x = x.cuda()
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    
    lhs = 20 * torch.exp((-0.2) * torch.sqrt((x ** 2).mean()))
    rhs = torch.cos(2*pi*x).mean().exp()
    result = -1*(lhs + rhs) + 20 + torch.exp(torch.Tensor([1]))#.cuda()
    
    return result 

def weierstrass_f(x, a=0.5, b=3, kmax=20):
    # x = x.cuda()
    x *= 0.005
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
        
    D = x.shape[0]
    
    lhs = 0
    rhs = 0
    for k in range(1,kmax+1):
        lhs += torch.sum((a**k) * torch.cos(2*pi*(b**k)*(x+0.5)))
        rhs += (a**k)*torch.cos(pi*(b**k))
    result = lhs + D*rhs
    return result