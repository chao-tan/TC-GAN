import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

def get_norm_layer(norm_type):

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, config):

    if config['lr_policy'] == 'linear':
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + 1 - int(config['niter'])) / float(int(config['niter_decay']) + 1)

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif config['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(config['lr_decay_iters']), gamma=0.1)
    elif config['lr_policy'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config['lr_policy'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config['niter']), eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config['lr_policy'])

    return scheduler


def init_weights(net, init_type, init_gain):

    # define initial function
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)  # init all the bias to 0
        elif classname.find('BatchNorm2d') != -1:  # Batchnorm can only be initialed by "normal" type
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
    init_weights(net, init_type, init_gain=init_gain)

    return net