import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
from torch.utils.data import DataLoader

from u_net import RecurrentUNet
from discriminator import Discriminator, FFTDiscriminator
from vgg_19 import VGG19
from model_wrapper import ModelWrapper
from dataset import REDS

if __name__ == '__main__':
    # Init networks
    generator_network = RecurrentUNet().cuda()
    discriminator_network = Discriminator().cuda()
    fft_discriminator_network = FFTDiscriminator().cuda()
    vgg_19 = VGG19().cuda()
    # Init optimizers
    generator_network_optimizer = torch.optim.Adam(generator_network.parameters(), lr=3e-4)
    discriminator_network_optimizer = torch.optim.Adam(discriminator_network.parameters(), lr=3e-4)
    fft_discriminator_network_optimizer = torch.optim.Adam(fft_discriminator_network.parameters(), lr=3e-4)
    # Init model wrapper
    model_wrapper = ModelWrapper(generator_network=generator_network,
                                 discriminator_network=discriminator_network,
                                 fft_discriminator_network=fft_discriminator_network,
                                 vgg_19=vgg_19,
                                 generator_network_optimizer=generator_network_optimizer,
                                 discriminator_network_optimizer=discriminator_network_optimizer,
                                 fft_discriminator_network_optimizer=fft_discriminator_network_optimizer,
                                 training_dataloader=DataLoader(
                                     REDS(path='/home/creich/REDS/train/train_sharp'), batch_size=1, shuffle=False),
                                 validation_dataloader=DataLoader(
                                     REDS(path='/home/creich/REDS/val/val_sharp'), batch_size=1, shuffle=False),
                                 test_dataloader=None)
    # Perform training
    model_wrapper.train()
