import torch
import torch.nn.functional as F


def shrink(x, theta):
    # print(x.sign() * F.relu(x.abs() - theta))
    return x.sign() * F.relu(x.abs() - theta)


x=torch.Tensor([[1,-2],[0,-3]])
if __name__ == "__main__":
    shrink(x,.1)










# def shrink_ss(x, theta, p):
#     x_abs = x.abs()
#     threshold = torch.quantile(x_abs, 1-p, dim=1, keepdims=True)
#     if isinstance(p, torch.Tensor) and p.numel() > 1:
#         threshold = torch.stack([threshold[i,i,0] for i in range(p.numel())]).unsqueeze(1)

#     bypass = torch.logical_and(x_abs >= threshold, x_abs >= theta).detach()
#     output = torch.where(bypass, x, shrink(x, theta))

#     return output