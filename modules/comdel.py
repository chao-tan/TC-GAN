import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from modules import network

class cmodel(ABC):
    def __init__(self, config):
        """"""
        """ initial abstract class
        """

        # get device name: cpu/gpu
        self.device = torch.device('cuda:{}'.format(config['gpu_ids'][0])) if config['gpu_ids'] else torch.device('cpu')
        self.save_dir = os.path.join(config['checkpoints_dir'], config['name'])
        self.schedulers = None
        self.gpu_ids = config['gpu_ids']

        torch.backends.cudnn.benchmark = True

        # define five init lists
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.test_result = []

        # learning rate parameter when policy is 'plateau'
        self.metric = None



    @abstractmethod
    def set_input(self, inputs):
        pass



    @abstractmethod
    def forward(self):
        pass



    @abstractmethod
    def optimize_parameters(self):
        pass



    def setup(self, config):
        # if training mode, set schedulers by default options
        if config['status'] == "train":
            self.schedulers = [network.get_scheduler(optimizer,config) for optimizer in self.optimizers]

        # if testing_usr mode, load weights from saved module
        if config['status'] == "test":
            self.load_networks(int(config['test_epoch']))
        self.print_networks(bool(config['verbose']))



    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()



    def test(self):
        with torch.no_grad():
            self.forward()



    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step(self.metric)
        lr_generator = self.optimizers[0].param_groups[0]['lr']
        lr_discriminator = self.optimizers[-1].param_groups[0]['lr']
        print('update generator learning rate = %.7f' % lr_generator)
        print('update discriminator learning rate = %.7f' % lr_discriminator)




    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret



    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'LOSS_' + name))
        return errors_ret



    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # torch.save(net.module.cpu().state_dict(), save_path)
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)



    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):

        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)



    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)

                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                for key in list(state_dict.keys()):
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)\



    def print_networks(self, verbose):

        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')



    @staticmethod
    def set_requires_grad(nets, requires_grad):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad