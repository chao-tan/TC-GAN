from modules import network
from torch import nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self, input_nc, discriminator_channel_base, norm):
        super(Discriminator, self).__init__()

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
                            out_channels=1,
                            kernel_size=4,
                            padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        return F.avg_pool2d(x, x.size()[2:])




def create_discriminator(input_nc, discriminator_channels_base, norm, init_type, init_gain, gpu_ids):

    net = Discriminator(input_nc=input_nc,
                        discriminator_channel_base=discriminator_channels_base,
                        norm=network.get_norm_layer(norm_type=norm))

    return network.init_net(net, init_type, init_gain, gpu_ids)