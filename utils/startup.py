import os
import yaml
from utils import tools
import torch


class SetupConfigs(object):
    def __init__(self,config_path):
        self.config_path = config_path


    @staticmethod
    def print_configs(configs):
        message = ''
        message += '----------------- Configs ---------------\n'
        for k in sorted(configs):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(configs[str(k)]), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to disk
        expr_dir = os.path.join(configs['checkpoints_dir'], configs['name'])
        tools.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'configs.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')



    def setup(self):
        with open(self.config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

        self.print_configs(config)

        str_ids = str(config['gpu_ids']).split(',')
        config['gpu_ids'] = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                config['gpu_ids'].append(id)
        if len(config['gpu_ids']) > 0:
            torch.cuda.set_device(config['gpu_ids'][0])
        return config
