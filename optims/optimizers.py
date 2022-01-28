import torch

def init_optim(optim, params, lr, weight_decay):
    if optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.95)
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optim == 'adamw':
        return torch.optim.AdamW(params, eps=1e-8, betas=(0.9, 0.999), lr=5e-4, weight_decay=0.05)
    else:
        raise KeyError("Unsupported Optimizer: {}".format(optim))