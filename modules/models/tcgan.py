from modules.comdel import cmodel
from modules.components import generator,discriminator,utils
from torch.nn import L1Loss
from torch.nn import functional as F
import torch
import itertools


class network(cmodel):

    def __init__(self, config):
        cmodel.__init__(self,config)

        self.config = config
        self.loss_names = ['D','G_GAN','IDT','TC']
        self.visual_names = ['REAL_A1','FAKE_B1','FAKE_B2']
        self.model_names = ["G1",'D1','G2','D2']

        self.netG1 = generator.create_generator(input_nc=int(config['input_nc']),
                                                output_nc=int(config['output_nc']),
                                                generator_channels_base=int(config['generator_channels_base']),
                                                norm=config['norm'],
                                                init_type=config['init_type'],
                                                init_gain=float(config['init_gain']),
                                                gpu_ids=config['gpu_ids'])

        self.netG2 = generator.create_generator(input_nc=int(config['input_nc']),
                                                output_nc=int(config['output_nc']),
                                                generator_channels_base=int(config['generator_channels_base']),
                                                norm=config['norm'],
                                                init_type=config['init_type'],
                                                init_gain=float(config['init_gain']),
                                                gpu_ids=config['gpu_ids'])


        self.netD1 = discriminator.create_discriminator(input_nc=int(config['output_nc']),
                                                        discriminator_channels_base=int(config['discriminator_channels_base']),
                                                        norm=config['norm'],
                                                        init_type=config['init_type'],
                                                        init_gain=float(config['init_gain']),
                                                        gpu_ids=config['gpu_ids'])

        self.netD2 = discriminator.create_discriminator(input_nc=int(config['output_nc']),
                                                        discriminator_channels_base=int(config['discriminator_channels_base']),
                                                        norm=config['norm'],
                                                        init_type=config['init_type'],
                                                        init_gain=float(config['init_gain']),
                                                        gpu_ids=config['gpu_ids'])

        if config['status'] == 'train':
            self.REAL_B1_POOL = utils.ImagePool(config['pool_size'])
            self.REAL_B2_POOL = utils.ImagePool(config['pool_size'])

            self.criterionGAN = utils.GANLoss(config['gan_mode'],self.config).to(self.device)
            self.criterionIDT = L1Loss()
            self.criterionTC = L1Loss()

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG1.parameters(),self.netG2.parameters()),lr=float(config['lr_generator']),betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD1.parameters(),self.netD2.parameters()),lr=float(config['lr_discriminator']),betas=(0.5, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if config['status'] == 'test':
            self.classifier = torch.load('classifier.pkl')

        self.REAL_A1 = None
        self.REAL_A2 = None
        self.REAL_B1 = None
        self.REAL_B2 = None
        self.FAKE_B1 = None
        self.FAKE_B2 = None
        self.IDT1 = None
        self.IDT2 = None

        self.LOSS_D = None
        self.LOSS_G_GAN = None
        self.LOSS_G = None
        self.LOSS_IDT = None
        self.LOSS_TC = None



    def set_input(self, inputs):
        self.REAL_A1 = inputs['A'].to(self.device)
        self.REAL_A2 = inputs['A'].to(self.device)
        self.REAL_B1 = inputs['B'].to(self.device)
        self.REAL_B2 = inputs['B'].to(self.device)



    def forward(self):
        if self.config['status'] == 'train':
            self.FAKE_B1 = self.netG1(self.REAL_A1)
            self.FAKE_B2 = self.netG2(self.REAL_A2)

        if self.config['status'] == 'test':
            with torch.no_grad():
                self.test_result = []
                self.FAKE_B1 = self.netG1(self.REAL_A1)
                self.FAKE_B2 = self.netG2(self.REAL_A2)

                FAKE_B1_PROB = F.softmax(self.classifier(self.FAKE_B1),dim=1)[0][1]
                FAKE_B2_PROB = F.softmax(self.classifier(self.FAKE_B2),dim=1)[0][1]

                if FAKE_B1_PROB > FAKE_B2_PROB : self.test_result.append(["FAKE_B", self.FAKE_B1])
                else: self.test_result.append(["FAKE_B", self.FAKE_B2])



    def backward_D_(self,netD1,REAL1,FAKE1,netD2,REAL2,FAKE2):
        PRED_REAL1 = netD1(REAL1)
        LOSS_D_REAL1 = self.criterionGAN(PRED_REAL1,True)
        PRED_FAKE1 = netD1(FAKE1.detach())
        LOSS_D_FAKE1 = self.criterionGAN(PRED_FAKE1,False)
        LOSS_D = (LOSS_D_REAL1 + LOSS_D_FAKE1)*0.5

        PRED_REAL2 = netD2(REAL2)
        LOSS_D_REAL2 = self.criterionGAN(PRED_REAL2,True)
        PRED_FAKE2 = netD2(FAKE2.detach())
        LOSS_D_FAKE2 = self.criterionGAN(PRED_FAKE2,False)
        LOSS_D = LOSS_D + (LOSS_D_REAL2 + LOSS_D_FAKE2)*0.5

        LOSS_D.backward()
        return LOSS_D


    def backward_D(self):
        REAL_B1 = self.REAL_B1_POOL.query(self.REAL_B1)
        REAL_B2 = self.REAL_B2_POOL.query(self.REAL_B2)
        self.LOSS_D = self.backward_D_(self.netD1,REAL_B1,self.FAKE_B1,self.netD2,REAL_B2,self.FAKE_B2)



    def COMPUTE_TC_LOSS(self,FAKE1,FAKE2):
        LOSS_TC = self.criterionTC(FAKE1,FAKE2.clone().detach())
        LOSS_TC = LOSS_TC + self.criterionTC(FAKE2,FAKE1.clone().detach())
        LOSS_TC = LOSS_TC *20.
        return LOSS_TC


    def backward_G(self):
        self.IDT1 = self.netG1(self.REAL_B1)
        self.IDT2 = self.netG2(self.REAL_B2)

        self.LOSS_G_GAN = self.criterionGAN(self.netD1(self.FAKE_B1),True)
        self.LOSS_G_GAN = self.LOSS_G_GAN + self.criterionGAN(self.netD2(self.FAKE_B2),True)

        self.LOSS_TC = self.COMPUTE_TC_LOSS(self.FAKE_B1,self.FAKE_B2)
        self.LOSS_IDT = self.criterionIDT(self.IDT1,self.REAL_B1) * 5.0
        self.LOSS_IDT = self.LOSS_IDT + self.criterionIDT(self.IDT2,self.REAL_B2) * 5.0

        self.LOSS_G = self.LOSS_G_GAN + self.LOSS_TC + self.LOSS_IDT

        self.LOSS_G.backward()


    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad([self.netD1,self.netD2],False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD1,self.netD2],True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

