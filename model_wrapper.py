from itertools import islice
from typing import Callable, Union, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.autograd
import torchvision
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime

import lossfunction
import misc


class ModelWrapper(object):
    """
    This class wraps all modules and implements train, validation, test and inference methods
    """

    def __init__(self, generator_network: Union[nn.Module, nn.DataParallel],
                 discriminator_network: Union[nn.Module, nn.DataParallel],
                 fft_discriminator_network: Union[nn.Module, nn.DataParallel],
                 vgg_19: Union[nn.Module, nn.DataParallel],
                 pwc_net: Union[nn.Module, nn.DataParallel],
                 resample: Union[nn.Module, nn.DataParallel],
                 generator_network_optimizer: torch.optim.Optimizer,
                 discriminator_network_optimizer: torch.optim.Optimizer,
                 fft_discriminator_network_optimizer: torch.optim.Optimizer, training_dataloader: DataLoader,
                 validation_dataloader: DataLoader, test_dataloader: DataLoader,
                 loss_function: nn.Module = lossfunction.AdaptiveRobustLoss(device='cuda:0',
                                                                            num_of_dimension=3 * 6 * 768 * 1024),
                 perceptual_loss: nn.Module = lossfunction.PerceptualLoss(),
                 flow_loss: nn.Module = nn.L1Loss(),
                 generator_loss: nn.Module = lossfunction.NonSaturatingLogisticGeneratorLoss(),
                 discriminator_loss: nn.Module = lossfunction.NonSaturatingLogisticDiscriminatorLoss(),
                 device='cuda', save_data_path: str = 'saved_data') -> None:
        """
        Constructor method
        :param generator_network: (nn.Module) Generator models
        :param discriminator_network: (nn.Module) Discriminator model
        :param fft_discriminator_network: (nn.Module) FFT discriminator model
        :param vgg_19: (nn.Module) Pre-trained VGG19 network
        :param pwc_net: (nn.Module) PWC-Net for optical flow estimation
        :param resample: (nn.Module) Resampling module
        :param generator_network_optimizer: (torch.optim.Optimizer) Generator optimizer module
        :param discriminator_network_optimizer: (torch.optim.Optimizer) Discriminator optimizer
        :param fft_discriminator_network_optimizer: (torch.optim.Optimizer) FFT discriminator model
        :param training_dataloader: (DataLoader) Training dataloader including the training dataset
        :param validation_dataloader: (DataLoader) Validation dataloader including the validation dataset
        :param test_dataloader: (DataLoader) Test dataloader including the test dataset
        :param loss_function: (nn.Module) Main supervised loss function
        :param perceptual_loss: (nn.Module) Perceptual loss function which takes two lists of tensors as input
        :param flow_loss: (nn.Module) Flow loss function
        :param generator_loss: (nn.Module) Adversarial generator loss function
        :param discriminator_loss: (nn.Module) Adversarial discriminator loss function
        :param device: (str) Device to be utilized (cpu not available if deformable convolutions are utilized)
        :param save_data_path: (str) Path to store logs, models and plots
        """
        # Save arguments
        self.generator_network = generator_network
        self.discriminator_network = discriminator_network
        self.fft_discriminator_network = fft_discriminator_network
        self.vgg_19 = vgg_19
        self.pwc_net = pwc_net
        self.resample = resample
        self.generator_network_optimizer = generator_network_optimizer
        self.discriminator_network_optimizer = discriminator_network_optimizer
        self.fft_discriminator_network_optimizer = fft_discriminator_network_optimizer
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.loss_function = loss_function
        self.perceptual_loss = perceptual_loss
        self.flow_loss = flow_loss
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.device = device
        self.save_data_path = save_data_path
        # Init logger
        self.logger = misc.Logger()
        # Make directories to save logs, plots and models during training
        # Not compatible with windows!!!
        time_and_date = str(datetime.now())
        self.path_save_models = os.path.join(save_data_path, 'models_' + time_and_date)
        if not os.path.exists(self.path_save_models):
            os.makedirs(self.path_save_models)
        self.path_save_plots = os.path.join(save_data_path, 'plots_' + time_and_date)
        if not os.path.exists(self.path_save_plots):
            os.makedirs(self.path_save_plots)
        self.path_save_metrics = os.path.join(save_data_path, 'metrics_' + time_and_date)
        if not os.path.exists(self.path_save_metrics):
            os.makedirs(self.path_save_metrics)
        # Log hyperparameter
        self.logger.hyperparameter['generator_network'] = str(generator_network)
        self.logger.hyperparameter['discriminator_network'] = str(discriminator_network)
        self.logger.hyperparameter['fft_discriminator_network'] = str(fft_discriminator_network)
        self.logger.hyperparameter['vgg_19'] = str(vgg_19)
        self.logger.hyperparameter['pwc_net'] = str(pwc_net)
        self.logger.hyperparameter['resample'] = str(resample)
        self.logger.hyperparameter['generator_network_optimizer'] = str(generator_network)
        self.logger.hyperparameter['generator_network'] = str(generator_network_optimizer)
        self.logger.hyperparameter['discriminator_network_optimizer'] = str(discriminator_network_optimizer)
        self.logger.hyperparameter['fft_discriminator_network_optimizer'] = str(fft_discriminator_network_optimizer)
        self.logger.hyperparameter['training_dataloader'] = str(training_dataloader)
        self.logger.hyperparameter['validation_dataloader'] = str(validation_dataloader)
        self.logger.hyperparameter['test_dataloader'] = str(test_dataloader)
        self.logger.hyperparameter['loss_function'] = str(loss_function)
        self.logger.hyperparameter['perceptual_loss'] = str(perceptual_loss)
        self.logger.hyperparameter['flow_loss'] = str(flow_loss)
        self.logger.hyperparameter['generator_loss'] = str(generator_loss)
        self.logger.hyperparameter['discriminator_loss'] = str(discriminator_loss)

    def train(self, epochs: int = 1, save_models_after_n_epochs: int = 1, validate_after_n_epochs: int = 1,
              w_supervised_loss: float = 5.0, w_adversarial: float = 1.0 / 100,
              w_fft_adversarial: float = 1.0 / 100, w_perceptual: float = 1.0, w_flow: float = 2.0,
              plot_after_n_iterations: int = 144) -> None:
        """
        Train method
        Note: GPU memory issues if all losses are computed at one. Solution: Calc losses independently. Drawback:
        Multiple forward passes are needed -> Slower training. Additionally gradients are not as smooth.
        :param epochs: (int) Number of epochs to perform
        :param save_models_after_n_epochs: (int) Epochs after models and optimizers gets saved
        :param validate_after_n_epochs: (int) Perform validation after a given number of epochs
        :param w_supervised_loss: (float) Weight factor for the supervised loss
        :param w_adversarial: (float) Weight factor for adversarial generator loss
        :param w_fft_adversarial: (float) Weight factor for fft adversarial generator loss
        :param w_perceptual: (float) Weight factor for perceptual loss
        :param w_flow: (float) Weight factor for flow loss
        :param inference_plot_after_n_iterations: (int) Make training plot after a given number of iterations
        """
        # Log weights in hyperparameters
        self.logger.hyperparameter['w_supervised_loss'] = str(w_supervised_loss)
        self.logger.hyperparameter['w_adversarial'] = str(w_adversarial)
        self.logger.hyperparameter['w_fft_adversarial'] = str(w_fft_adversarial)
        self.logger.hyperparameter['w_perceptual'] = str(w_perceptual)
        self.logger.hyperparameter['w_flow'] = str(w_flow)
        # Model into training mode
        self.generator_network.train()
        self.discriminator_network.train()
        self.fft_discriminator_network.train()
        # PWC-Net into eval mode
        self.pwc_net.eval()
        # Vgg into eval mode
        self.vgg_19.eval()
        # Models to device
        self.generator_network.to(self.device)
        self.discriminator_network.to(self.device)
        self.fft_discriminator_network.to(self.device)
        self.vgg_19.to(self.device)
        # Init progress bar
        self.progress_bar = tqdm(total=epochs * len(self.training_dataloader.dataset))
        # Main loop
        for epoch in range(epochs):
            for input, label, new_sequence in self.training_dataloader:
                # Update progress bar
                self.progress_bar.update(n=input.shape[0])
                # Reset gradients of networks
                self.generator_network.zero_grad()
                self.discriminator_network.zero_grad()
                self.fft_discriminator_network.zero_grad()
                self.vgg_19.zero_grad()
                # Data to device
                input = input.to(self.device)
                label = label.to(self.device)
                # Reset recurrent tensor
                '''
                if bool(new_sequence):
                    if isinstance(self.generator_network, nn.DataParallel):
                        self.generator_network.module.reset_recurrent_tensor()
                    else:
                        self.generator_network.reset_recurrent_tensor()
                '''
                ############# Supervised training (+ perceptrual training) #############
                # Make prediction
                prediction = self.generator_network(input.detach())
                # Reshape prediction and label for vgg19
                prediction_reshaped_4d = prediction.reshape(prediction.shape[0] * (prediction.shape[1] // 3), 3,
                                                            prediction.shape[2], prediction.shape[3])
                label_reshaped_4d = label.reshape(label.shape[0] * (label.shape[1] // 3), 3, label.shape[2],
                                                  label.shape[3])
                # Call supervised loss
                loss_supervised = w_supervised_loss * self.loss_function(prediction, label)

                loss_perceptual = w_perceptual * self.perceptual_loss(
                    self.vgg_19(F.avg_pool2d(prediction_reshaped_4d, kernel_size=2)),
                    self.vgg_19(F.avg_pool2d(label_reshaped_4d, kernel_size=2)))
                # Calc gradients
                (loss_supervised + loss_perceptual).backward()
                # Optimize generator
                self.generator_network_optimizer.step()
                # Reset gradients of generator network
                self.generator_network.zero_grad()
                ############# Adversarial training #############
                # Make prediction
                prediction = self.generator_network(input.detach())
                # Calc discriminator loss
                loss_discriminator_real, loss_discriminator_fake = self.discriminator_loss(
                    self.discriminator_network(label),
                    self.discriminator_network(prediction.detach()))
                # Calc gradients
                (loss_discriminator_fake + loss_discriminator_real).backward()
                # Optimize discriminator
                self.discriminator_network_optimizer.step()
                # Reset gradients of generator
                self.generator_network.zero_grad()
                # Calc generator loss
                loss_generator = w_adversarial * self.generator_loss(
                    self.discriminator_network(prediction))
                # Calc gradients
                loss_generator.backward()
                # Optimize generator and discriminator
                self.generator_network_optimizer.step()
                ############# Flow training #############
                # Reset gradients of generator network
                self.generator_network.zero_grad()
                # Make prediction
                prediction = self.generator_network(input.detach())
                # Reshape prediction and label for vgg19
                prediction_reshaped_4d = prediction.reshape(prediction.shape[0] * (prediction.shape[1] // 3), 3,
                                                            prediction.shape[2], prediction.shape[3])
                prediction_pair = torch.cat((prediction_reshaped_4d[:-1].detach(), prediction_reshaped_4d[1:].detach()),
                                            dim=1)
                # Get flow
                with torch.no_grad():
                    flow = self.pwc_net(prediction_pair)
                    # Get resampled images
                    resampled_images = self.resample(prediction_reshaped_4d[1:], flow)
                # Calc flow loss
                loss_flow = self.flow_loss(prediction_reshaped_4d[:-1], resampled_images)
                # Calc gradients
                loss_flow.backward()
                # Optimizer generator
                self.generator_network_optimizer.step()
                # Reset gradients of generator network
                self.generator_network.zero_grad()
                ############# Adversarial training (FFT) #############
                # Make prediction
                prediction = self.generator_network(input.detach())
                # Calc discriminator loss
                loss_fft_discriminator_real, loss_fft_discriminator_fake = self.discriminator_loss(
                    self.fft_discriminator_network(label),
                    self.fft_discriminator_network(prediction.detach()))
                # Calc gradients
                (loss_fft_discriminator_fake + loss_fft_discriminator_real).backward()
                # Optimize discriminator
                self.fft_discriminator_network_optimizer.step()
                # Reset grad of generator
                self.generator_network.zero_grad()
                # Calc generator loss
                loss_fft_generator = w_fft_adversarial * self.generator_loss(
                    self.fft_discriminator_network(prediction))
                # Calc gradients
                loss_fft_generator.backward()
                # Optimize generator
                self.generator_network_optimizer.step()
                # Update progress bar
                self.progress_bar.set_description(
                    'SV Loss={:.3f}, P Loss={:.3f}, F Loss={:.3f}, A.G. Loss={:.3f}, A.D. Loss={:.3f}, A.FFT G. Loss={:.3f}, A.FFT D. Loss={:.3f}'
                        .format(loss_supervised.item(),
                                loss_perceptual.item(),
                                loss_flow.item(),
                                loss_generator.item(),
                                loss_discriminator_real.item() + loss_discriminator_fake.item(),
                                loss_fft_generator.item(),
                                loss_fft_discriminator_real.item() + loss_fft_discriminator_fake.item()))
                # Log losses
                self.logger.log(metric_name='training_iteration', value=self.progress_bar.n)
                self.logger.log(metric_name='epoch', value=epoch)
                self.logger.log(metric_name='loss_supervised', value=loss_supervised.item())
                self.logger.log(metric_name='loss_perceptual', value=loss_perceptual.item())
                self.logger.log(metric_name='loss_generator', value=loss_generator.item())
                self.logger.log(metric_name='loss_discriminator',
                                value=loss_discriminator_real.item() + loss_discriminator_fake.item())
                self.logger.log(metric_name='loss_flow', value=loss_flow.item())
                self.logger.log(metric_name='loss_fft_generator', value=loss_fft_generator.item())
                self.logger.log(metric_name='loss_fft_discriminator',
                                value=loss_fft_discriminator_real.item() + loss_fft_discriminator_fake.item())
                # Plot training prediction
                if (self.progress_bar.n) % (plot_after_n_iterations) == 0:
                    prediction_batched = prediction.reshape(
                        prediction.shape[0] * self.validation_dataloader.dataset.number_of_frames, 3,
                        prediction.shape[2], prediction.shape[3])
                    label_batched = label.reshape(label.shape[0] * self.validation_dataloader.dataset.number_of_frames,
                                                  3, label.shape[2], label.shape[3])
                    # Normalize images batch wise to range of [0, 1]
                    prediction_batched = misc.normalize_0_1_batch(prediction_batched)
                    label_batched = misc.normalize_0_1_batch(label_batched)
                    # Make plots
                    torchvision.utils.save_image(
                        prediction_batched,
                        filename=os.path.join(self.path_save_plots,
                                              'prediction_train_{}.png'.format(self.progress_bar.n)),
                        nrow=self.validation_dataloader.dataset.number_of_frames)
                    torchvision.utils.save_image(
                        label_batched,
                        filename=os.path.join(self.path_save_plots,
                                              'label_train_{}.png'.format(self.progress_bar.n)),
                        nrow=self.validation_dataloader.dataset.number_of_frames)
            # Save models and optimizer
            if epoch % save_models_after_n_epochs == 0:
                # Save models
                torch.save(self.generator_network.state_dict(),
                           os.path.join(self.path_save_models, 'generator_network_{}.pt'.format(epoch)))
                torch.save(self.discriminator_network.state_dict(),
                           os.path.join(self.path_save_models, 'discriminator_network_{}.pt'.format(epoch)))
                torch.save(self.fft_discriminator_network.state_dict(),
                           os.path.join(self.path_save_models, 'fft_discriminator_network_{}.pt'.format(epoch)))
                # Save optimizers
                torch.save(self.generator_network_optimizer,
                           os.path.join(self.path_save_models, 'generator_network_optimizer_{}.pt'.format(epoch)))
                torch.save(self.discriminator_network_optimizer,
                           os.path.join(self.path_save_models, 'discriminator_network_optimizer_{}.pt'.format(epoch)))
                torch.save(self.fft_discriminator_network_optimizer,
                           os.path.join(self.path_save_models,
                                        'fft_discriminator_network_optimizer_{}.pt'.format(epoch)))
            if epoch % validate_after_n_epochs == 0:
                # Validation
                self.progress_bar.set_description('Validate...')
                self.validate()
                # Log validation epoch
                self.logger.log(metric_name='validation_epoch', value=epoch)
            # Save logs
            self.logger.save_metrics(self.path_save_metrics)
        # Close progress bar
        self.progress_bar.close()
        # Save logs finally
        self.logger.save_metrics(self.path_save_metrics)

    @torch.no_grad()
    def validate(self,
                 validation_metrics: Tuple[Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]]
                 = (nn.L1Loss(reduction='mean'), nn.MSELoss(reduction='mean'), misc.psnr, misc.ssim),
                 sequences_to_plot: Tuple[int, ...] = (1, 2, 3, 4, 76, 83, 124, 150, 220, 432)) -> None:
        """
        Validation method which produces validation metrics and plots
        :param validation_metrics: (Tuple[Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]]) Tuple
        of callable validation metric to be computed
        :param sequences_to_plot: (Tuple[int, ...]) Tuple of validation dataset indexes to be plotted
        """
        # Generator model to device
        self.generator_network.to(self.device)
        # Generator into eval mode
        self.generator_network.eval()
        # Init dict to store metrics
        metrics = dict()
        # Main loop
        for index_sequence, batch in enumerate(self.validation_dataloader):
            # Unpack batch
            input, label, new_sequence = batch
            # Data to device
            input = input.to(self.device)
            label = label.to(self.device)
            # Reset recurrent tensor
            '''
            if bool(new_sequence):
                if isinstance(self.generator_network, nn.DataParallel):
                    self.generator_network.module.reset_recurrent_tensor()
                else:
                    self.generator_network.reset_recurrent_tensor()
            '''
            # Make prediction
            prediction = self.generator_network(input)
            # Calc validation metrics
            for validation_metric in validation_metrics:
                # Calc metric
                metric = validation_metric(prediction, label).item()
                # Case if validation metric is a nn.Module
                if isinstance(validation_metric, nn.Module):
                    # Save metric and name of metric
                    if validation_metric.__class__.__name__ in metrics.keys():
                        metrics[validation_metric.__class__.__name__].append(metric)
                    else:
                        metrics[validation_metric.__class__.__name__] = [metric]
                # Case if validation metric is a callable function
                else:
                    # Save metric and name of metric
                    if validation_metric.__name__ in metrics.keys():
                        metrics[validation_metric.__name__].append(metric)
                    else:
                        metrics[validation_metric.__name__] = [metric]
            # Plot prediction label and input
            if index_sequence in sequences_to_plot:
                # Reshape tensors
                prediction_batched = prediction.reshape(self.validation_dataloader.dataset.number_of_frames, 3,
                                                        prediction.shape[2], prediction.shape[3])
                input_batched = input.reshape(self.validation_dataloader.dataset.number_of_frames, 3,
                                              input.shape[2], input.shape[3])
                label_batched = label.reshape(self.validation_dataloader.dataset.number_of_frames, 3,
                                              label.shape[2], label.shape[3])
                # Normalize images batch wise to range of [0, 1]
                prediction_batched = misc.normalize_0_1_batch(prediction_batched)
                input_batched = misc.normalize_0_1_batch(input_batched)
                label_batched = misc.normalize_0_1_batch(label_batched)
                # Make plots
                torchvision.utils.save_image(
                    prediction_batched,
                    filename=os.path.join(self.path_save_plots,
                                          'prediction_{}_{}.png'.format(index_sequence, str(datetime.now()))),
                    nrow=self.validation_dataloader.dataset.number_of_frames)
                torchvision.utils.save_image(
                    label_batched,
                    filename=os.path.join(self.path_save_plots,
                                          'label_{}_{}.png'.format(index_sequence, str(datetime.now()))),
                    nrow=self.validation_dataloader.dataset.number_of_frames)
                torchvision.utils.save_image(
                    input_batched,
                    filename=os.path.join(self.path_save_plots,
                                          'input_{}_{}.png'.format(index_sequence, str(datetime.now()))),
                    nrow=self.validation_dataloader.dataset.number_of_frames)
        # Average metrics and save them in logs
        for metric_name in metrics:
            self.logger.log(metric_name=metric_name, value=float(np.mean(metrics[metric_name])))
        # Save metrics
        self.logger.save_metrics(path=self.path_save_metrics)

    @torch.no_grad()
    def test(self) -> None:
        pass

    @torch.no_grad()
    def inference(self, sequences: List[torch.Tensor] = None, apply_fovea_filter: bool = True) -> None:
        """
        Inference method generates the reconstructed image to the corresponding input and saves the input, label and
        output as an image
        :param sequences: (List[torch.Tensor]) List of video sequences
        :param apply_fovea_filter: (bool) If true the fovea filter is applied to the input sequence
        """
        # Generator into eval mode
        self.generator_network.eval()
        # Model to device
        self.generator_network.to(self.device)
        # Reset recurrent tensor in generator
        # self.generator_network.reset_recurrent_tensor()
        for index, sequence in enumerate(sequences):
            # Apply fovea mask if utilized
            if apply_fovea_filter:
                if index == 0:
                    # Get fovea mask and probability mask
                    fovea_mask, p_mask = misc.get_fovea_mask((sequence.shape[2], sequence.shape[3]), return_p_mask=True)
                else:
                    # Get fovea mask
                    fovea_mask = misc.get_fovea_mask((sequence.shape[2], sequence.shape[3]), p_mask=p_mask,
                                                     return_p_mask=False)
                # Apply fovea mask
                sequence = sequence * fovea_mask.view(1, 1, sequence.shape[2], sequence.shape[3])
            # Sequence to device
            sequence = sequence.to(self.device)
            # Make prediction
            prediction = self.generator_network(sequence)
            # Reshape tensors
            prediction_batched = prediction.reshape(self.validation_dataloader.dataset.number_of_frames, 3,
                                                    prediction.shape[2], prediction.shape[3])
            # Normalize images batch wise to range of [0, 1]
            prediction_batched = misc.normalize_0_1_batch(prediction_batched)
            # Make plots
            torchvision.utils.save_image(
                prediction_batched,
                filename=os.path.join(self.path_save_plots,
                                      'prediction_inf_{}_{}.png'.format(index, str(datetime.now()))),
                nrow=self.validation_dataloader.dataset.number_of_frames)
        # Generator back into train mode
        self.generator_network.train()
        # Reset recurrent tensor in generator
        # self.generator_network.reset_recurrent_tensor()
