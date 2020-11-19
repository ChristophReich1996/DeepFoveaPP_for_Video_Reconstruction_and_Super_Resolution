from argparse import ArgumentParser
import os

# Manage command line arguments
parser = ArgumentParser()
parser.add_argument("--train", default=False, action="store_true",
                    help="Binary flag. If set training will be performed.")
parser.add_argument("--val", default=False, action="store_true",
                    help="Binary flag. If set validation will be performed.")
parser.add_argument("--test", default=False, action="store_true",
                    help="Binary flag. If set testing will be performed.")
parser.add_argument("--inference", default=False, action="store_true",
                    help="Binary flag. If set inference will be performed.")
parser.add_argument("--inference_data", default="", type=str,
                    help="Path to inference data to be loaded.")
parser.add_argument("--cuda_devices", default="0", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")
parser.add_argument("--data_parallel", default=False, action="store_true",
                    help="Binary flag. If multi GPU training should be utilized set flag.")
parser.add_argument("--load_model", default="", type=str,
                    help="Path to model to be loaded.")

# Get arguments
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import RecurrentUNet
from discriminator import Discriminator, FFTDiscriminator
from vgg_19 import VGG19
from pwc_net import PWCNet
from model_wrapper import ModelWrapper
from dataset import REDS, REDSFovea, REDSParallel, REDSFoveaParallel, reds_parallel_collate_fn
from lossfunction import AdaptiveRobustLoss
from resample.resample2d import Resample2d
import misc

if __name__ == '__main__':
    # Init networks
    generator_network = nn.DataParallel(RecurrentUNet().cuda())
    if args.load_model != "":
        generator_network.load_state_dict(torch.load(args.load_model))
    # Init additional models
    discriminator_network = nn.DataParallel(Discriminator().cuda())
    fft_discriminator_network = nn.DataParallel(FFTDiscriminator().cuda())
    vgg_19 = nn.DataParallel(VGG19().cuda())
    pwc_net = nn.DataParallel(PWCNet().cuda())
    resample = nn.DataParallel(Resample2d().cuda())
    # Init adaptive loss
    loss_function = AdaptiveRobustLoss(device='cuda:0', num_of_dimension=3 * 6 * 768 * 1024)
    # Init optimizers
    generator_network_optimizer = torch.optim.Adam(
        list(generator_network.parameters()) + list(loss_function.parameters()), lr=3e-4, betas=(0.1, 0.95))
    discriminator_network_optimizer = torch.optim.Adam(discriminator_network.parameters(), lr=1e-4, betas=(0.1, 0.95))
    fft_discriminator_network_optimizer = torch.optim.Adam(fft_discriminator_network.parameters(), lr=1e-4,
                                                           betas=(0.1, 0.95))
    # Init model wrapper
    model_wrapper = ModelWrapper(generator_network=generator_network,
                                 discriminator_network=discriminator_network,
                                 fft_discriminator_network=fft_discriminator_network,
                                 vgg_19=vgg_19,
                                 pwc_net=pwc_net,
                                 resample=resample,
                                 generator_network_optimizer=generator_network_optimizer,
                                 discriminator_network_optimizer=discriminator_network_optimizer,
                                 fft_discriminator_network_optimizer=fft_discriminator_network_optimizer,
                                 loss_function=loss_function,
                                 training_dataloader=DataLoader(
                                     REDSFoveaParallel(path='/home/creich/REDS/train/train_sharp', number_of_gpus=2),
                                     batch_size=2, shuffle=False, num_workers=1, collate_fn=reds_parallel_collate_fn),
                                 validation_dataloader=DataLoader(
                                     REDSFovea(path='/home/creich/REDS/val/val_sharp'), batch_size=1, shuffle=False,
                                     num_workers=1),
                                 test_dataloader=None)
    # Perform training
    if args.train:
        model_wrapper.train(epochs=20)
    # Perform final validation
    if args.val:
        model_wrapper.validate()
    # Perform testing
    if args.test:
        model_wrapper.test()
    # Perform validation
    if args.inference:
        # Load data
        inference_data = misc.load_inference_data(args.inference_data)
        model_wrapper.inference(inference_data)
