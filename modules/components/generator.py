from modules import network
from torch import nn


class ResNetGenerator(nn.Module):
    def __init__(self,input_nc, output_nc, num_downs, generator_channels_base, norm_layer):
        super(ResNetGenerator, self).__init__()

        model = []
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels=input_nc,
                            out_channels=generator_channels_base,
                            kernel_size=7,
                            stride=1,
                            padding=0,
                            bias=True),
                  norm_layer(generator_channels_base),
                  nn.ReLU(inplace=True)]

        for i in range(num_downs):
            in_features = min(generator_channels_base * 8, generator_channels_base * (2**i))
            out_features = min(generator_channels_base * 8, generator_channels_base * (2**(i+1)))

            model += [nn.Conv2d(in_channels=in_features,
                                out_channels=out_features,
                                kernel_size=3,
                                stride=2,
                                padding=1),
                      norm_layer(out_features),
                      nn.ReLU(inplace=True)]

        for _ in range(9):
            model += [ResidualBlock(num_channels=min(generator_channels_base*8, generator_channels_base * (2** num_downs)),
                                    norm=norm_layer)]

        for i in range(num_downs):
            in_features = max(generator_channels_base,
                              min(generator_channels_base*8, generator_channels_base * (2**(num_downs-i))))
            out_features = max(generator_channels_base,
                               min(generator_channels_base*8, generator_channels_base * (2**(num_downs-i))//2))

            model += [nn.ConvTranspose2d(in_channels=in_features,
                                         out_channels=out_features,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1),
                      norm_layer(out_features),
                      nn.ReLU(inplace=True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels=generator_channels_base,
                            out_channels=output_nc,
                            kernel_size=7,
                            padding=0,
                            stride=1,
                            bias=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return (self.model(x)+x).tanh()



class ResidualBlock(nn.Module):
    def __init__(self, num_channels,norm):
        super(ResidualBlock, self).__init__()
        self.num_channels = num_channels
        self.norm = norm

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels=self.num_channels,
                                out_channels=self.num_channels,
                                kernel_size=3,
                                stride=1,
                                bias=True),
                      self.norm(self.num_channels),
                      nn.ReLU(inplace=True),

                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels=self.num_channels,
                                out_channels=self.num_channels,
                                kernel_size=3,
                                stride=1,
                                bias=True),
                      self.norm(self.num_channels)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)



def create_generator(input_nc, output_nc, generator_channels_base, norm, init_type, init_gain, gpu_ids):

    net = ResNetGenerator(input_nc=input_nc,
                          output_nc=output_nc,
                          generator_channels_base=generator_channels_base,
                          norm_layer=network.get_norm_layer(norm_type=norm),
                          num_downs=2)

    return network.init_net(net, init_type, init_gain, gpu_ids)