import torch.nn as nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self, input_nc=3, discriminator_channel_base=64, norm=nn.BatchNorm2d):
        super(Net, self).__init__()

        model = [nn.Conv2d(in_channels=input_nc,
                           out_channels=discriminator_channel_base,
                           kernel_size=4,
                           stride=2,
                           padding=1),
                 nn.LeakyReLU(0.2,inplace=True)]

        model += [nn.Conv2d(in_channels=discriminator_channel_base,
                            out_channels=discriminator_channel_base*2,
                            kernel_size=4,
                            stride=2,
                            padding=1),
                  norm(discriminator_channel_base*2),
                  nn.LeakyReLU(0.2,inplace=True)]

        model += [nn.Conv2d(in_channels=discriminator_channel_base*2,
                            out_channels=discriminator_channel_base*4,
                            kernel_size=4,
                            stride=2,
                            padding=1),
                  norm(discriminator_channel_base*4),
                  nn.LeakyReLU(0.2,inplace=True)]

        model += [nn.Conv2d(in_channels=discriminator_channel_base*4,
                            out_channels=discriminator_channel_base*8,
                            kernel_size=4,
                            padding=1),
                  norm(discriminator_channel_base*8),
                  nn.LeakyReLU(0.2,inplace=True) ]

        model += [nn.Conv2d(in_channels=discriminator_channel_base*8,
                            out_channels=2,
                            kernel_size=4,
                            padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0),-1)
        return x