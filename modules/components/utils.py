import torch
from torch import nn
import random


class ImagePool(object):
    def __init__(self, pool_size):

        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty image pool if pool_size > 0
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:  # do not do anything when image pool size is 0
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                random_id = random.randint(0, self.num_imgs - 1)
                tmp = self.images[random_id].clone()
                return_images.append(tmp)
            else:
                random_id = random.randint(0, self.pool_size - 1)
                tmp = self.images[random_id].clone()
                self.images[random_id] = image
                return_images.append(tmp)
        return_images = torch.cat(return_images, 0)
        return return_images




class GANLoss(nn.Module):
    def __init__(self, gan_mode,config):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        self.config = config

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == 'bcegan':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)


    def get_target_tensor(self, prediction, target):
        if target is True:
            target_tensor = torch.tensor(1.0).to(
                torch.device('cuda:{}'.format(self.config['gpu_ids'][0])) if self.config['gpu_ids'] else torch.device('cpu'))
        else:
            target_tensor = torch.tensor(0.0).to(
                torch.device('cuda:{}'.format(self.config['gpu_ids'][0])) if self.config['gpu_ids'] else torch.device('cpu'))

        return target_tensor.expand_as(prediction)


    def forward(self, prediction, ground):
        target_tensor = self.get_target_tensor(prediction, ground)
        loss = self.loss(prediction, target_tensor)

        return loss