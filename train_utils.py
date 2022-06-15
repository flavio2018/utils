import torch

def get_optimizer(model, cfg):
    if cfg.train.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    elif cfg.train.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=cfg.train.lr)
    else:
        return torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
