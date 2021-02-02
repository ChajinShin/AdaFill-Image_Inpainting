import torch.optim as optim
from . import module


def model_setting(opt, dev):
    # model init
    inpaint_net = module.InpaintNet(opt).to(dev)
    disc_net = module.Discriminator(opt).to(dev)

    # optimizer
    inpaint_kwargs = {
        'lr': float(opt['lr_inpaint']),
        'beta1': opt['beta1'],
        'beta2': opt['beta2'],
        'eps': 1e-8,
        'parameter': inpaint_net.parameters()
    }
    disc_kwargs = {
        'lr': float(opt['lr_disc']),
        'beta1': opt['beta1'],
        'beta2': opt['beta2'],
        'eps': 1e-8,
        'parameter': disc_net.parameters()
    }
    inpaint_optim = make_optimizer(inpaint_kwargs)
    disc_optim = make_optimizer(disc_kwargs)

    # lr scheduler
    inpaint_scheduler = make_scheduler(opt, inpaint_optim)
    disc_scheduler = make_scheduler(opt, disc_optim)

    return inpaint_net, disc_net, inpaint_optim, disc_optim, inpaint_scheduler, disc_scheduler


def make_optimizer(kwargs):
    component = {
        'betas': (kwargs['beta1'], kwargs['beta2']),
        'eps': kwargs['eps']
    }
    component['lr'] = kwargs['lr']
    return optim.Adam(kwargs['parameter'], **component)


def make_scheduler(opt, optimizer):
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=opt['gamma']
    )
    return scheduler

