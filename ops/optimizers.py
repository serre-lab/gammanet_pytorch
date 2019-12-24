from torch import optim 

def get_optimizer(optim_name, **kwargs):
    assert hasattr(optim, optim_name), 'optimizer unkown' 

    return getattr(optim, optim_name)