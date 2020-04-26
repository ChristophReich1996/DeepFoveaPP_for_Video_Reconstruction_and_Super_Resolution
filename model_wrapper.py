from typing import Callable, Union, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np

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
                 generator_network_optimizer: torch.optim.Optimizer,
                 discriminator_network_optimizer: torch.optim.Optimizer,
                 fft_discriminator_network_optimizer: torch.optim.Optimizer, training_dataloader: DataLoader,
                 validation_dataloader: DataLoader, test_dataloader: DataLoader,
                 loss_function: nn.Module = lossfunction.AdaptiveRobustLoss(device='cuda:0',
                                                                            num_of_dimension=3 * 12 * 1024 * 768),
                 perceptual_loss: nn.Module = lossfunction.PerceptualLoss(),
                 generator_loss: nn.Module = lossfunction.WassersteinGeneratorLoss(),
                 discriminator_loss: nn.Module = lossfunction.WassersteinDiscriminatorLoss(), device='cuda',
                 save_data_path: str = 'saved_data') -> None:
        """
        Constructor method
        :param generator_network: (nn.Module) Generator models
        :param discriminator_network: (nn.Module) Discriminator model
        :param fft_discriminator_network: (nn.Module) FFT discriminator model
        :param vgg_19: (nn.Module) Pre-trained VGG19 network
        :param generator_network_optimizer: (torch.optim.Optimizer) Generator optimizer module
        :param discriminator_network_optimizer: (torch.optim.Optimizer) Discriminator optimizer
        :param fft_discriminator_network_optimizer: (torch.optim.Optimizer) FFT discriminator model
        :param training_dataloader: (DataLoader) Training dataloader including the training dataset
        :param validation_dataloader: (DataLoader) Validation dataloader including the validation dataset
        :param test_dataloader: (DataLoader) Test dataloader including the test dataset
        :param loss_function: (nn.Module) Main supervised loss function
        :param perceptual_loss: (nn.Module) Perceptual loss function which takes two lists of tensors as input
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
        self.generator_network_optimizer = generator_network_optimizer
        self.discriminator_network_optimizer = discriminator_network_optimizer
        self.fft_discriminator_network_optimizer = fft_discriminator_network_optimizer
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.loss_function = loss_function
        self.perceptual_loss = perceptual_loss
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
        self.logger.hyperparameter['generator_network_optimizer'] = str(generator_network)
        self.logger.hyperparameter['generator_network'] = str(generator_network_optimizer)
        self.logger.hyperparameter['discriminator_network_optimizer'] = str(discriminator_network_optimizer)
        self.logger.hyperparameter['fft_discriminator_network_optimizer'] = str(fft_discriminator_network_optimizer)
        self.logger.hyperparameter['training_dataloader'] = str(training_dataloader)
        self.logger.hyperparameter['validation_dataloader'] = str(validation_dataloader)
        self.logger.hyperparameter['test_dataloader'] = str(test_dataloader)
        self.logger.hyperparameter['loss_function'] = str(loss_function)
        self.logger.hyperparameter['perceptual_loss'] = str(perceptual_loss)
        self.logger.hyperparameter['generator_loss'] = str(generator_loss)
        self.logger.hyperparameter['discriminator_loss'] = str(discriminator_loss)

    def train(self, epochs: int = 1, save_models_after_n_epochs: int = 4, validate_after_n_epochs: int = 4,
              w_supervised_loss: float = 1.0, w_adversarial: float = 1.0, w_fft_adversarial: float = 1.0) -> None:
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
        """
        # Log weights in hyperparameters
        self.logger.hyperparameter['w_supervised_loss'] = str(w_supervised_loss)
        self.logger.hyperparameter['w_adversarial'] = str(w_adversarial)
        self.logger.hyperparameter['w_fft_adversarial'] = str(w_fft_adversarial)
        # Model into training mode
        self.generator_network.train()
        self.discriminator_network.train()
        self.fft_discriminator_network.train()
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
                if bool(new_sequence):
                    if isinstance(self.generator_network, nn.DataParallel):
                        self.generator_network.module.reset_recurrent_tensor()
                    else:
                        self.generator_network.reset_recurrent_tensor()
                ############# Supervised training (+ perceptrual training) #############
                # Make prediction
                prediction = self.generator_network(input)[-1]
                # Reshape prediction and label for vgg19
                prediction_reshaped_4d = prediction.reshape(prediction.shape[0] * (prediction.shape[1] // 3), 3,
                                                            prediction.shape[2], prediction.shape[3])
                label_reshaped_4d = label.reshape(label.shape[0] * (label.shape[1] // 3), 3, label.shape[2],
                                                  label.shape[3])
                # Call supervised loss
                loss_supervised = w_supervised_loss * self.loss_function(prediction, label)
                '''
                loss_supervised = w_supervised_loss * self.loss_function(prediction, label) \
                                  + self.perceptual_loss(self.vgg_19(prediction_reshaped_4d),
                                                         self.vgg_19(label_reshaped_4d))
                '''
                # Calc gradients
                loss_supervised.backward()
                # Optimize generator
                self.generator_network_optimizer.step()
                # Reset gradients of generator network and vgg 19
                self.generator_network.zero_grad()
                self.vgg_19.zero_grad()
                ############# Adversarial training #############
                # Make prediction
                prediction = self.generator_network(input)[-1]
                # Reshape label and prediction for discriminator
                prediction_reshaped_5d = prediction.reshape(prediction.shape[0], 3, prediction.shape[1] // 3,
                                                            prediction.shape[2], prediction.shape[3])
                label_reshaped_5d = label.reshape(label.shape[0], 3, label.shape[1] // 3, label.shape[2],
                                                  label.shape[3])
                # Calc discriminator loss
                loss_discriminator = self.discriminator_loss(self.discriminator_network(label_reshaped_5d),
                                                             self.discriminator_network(prediction_reshaped_5d))
                # Calc gradients and retain graph of generator gradients
                loss_discriminator.backward(retain_graph=True)
                # Calc generator loss
                loss_generator = w_adversarial * self.generator_loss(self.discriminator_network(prediction_reshaped_5d))
                # Calc gradients
                loss_generator.backward()
                # Optimize generator and discriminator
                self.generator_network_optimizer.step()
                self.discriminator_network_optimizer.step()
                # Reset gradients of generator and discriminator
                self.generator_network.zero_grad()
                self.discriminator_network.zero_grad()
                ############# Adversarial training (FFT) #############
                # Make prediction
                prediction = self.generator_network(input)[-1]
                # Reshape label and prediction for discriminator
                prediction_reshaped_5d = prediction.reshape(prediction.shape[0], 3, prediction.shape[1] // 3,
                                                            prediction.shape[2], prediction.shape[3])
                label_reshaped_5d = label.reshape(label.shape[0], 3, label.shape[1] // 3, label.shape[2],
                                                  label.shape[3])
                # Calc discriminator loss
                loss_fft_discriminator = self.discriminator_loss(self.fft_discriminator_network(label_reshaped_5d),
                                                                 self.fft_discriminator_network(prediction_reshaped_5d))
                # Calc gradients and retain graph of generator gradients
                loss_fft_discriminator.backward(retain_graph=True)
                # Calc generator loss
                loss_fft_generator = w_fft_adversarial * self.generator_loss(
                    self.fft_discriminator_network(prediction_reshaped_5d))
                # Calc gradients
                loss_fft_generator.backward()
                # Optimize generator and discriminator
                self.generator_network_optimizer.step()
                self.fft_discriminator_network_optimizer.step()
                # Reset gradients of generator and discriminator
                self.generator_network.zero_grad()
                self.fft_discriminator_network.zero_grad()
                # Update progress bar
                self.progress_bar.set_description(
                    'SV Loss={:.4f}, Adv. G. Loss={:.4f}, Adv. D. Loss={:.4f}, Adv. FFTG. Loss={:.4f}, Adv. FFTD. Loss={:.4f}'
                        .format(loss_supervised.item(), loss_generator.item(), loss_discriminator.item(),
                                loss_fft_generator.item(), loss_fft_discriminator.item()))
                # Log losses
                self.logger.log(metric_name='training_iteration', value=self.progress_bar.n)
                self.logger.log(metric_name='epoch', value=epoch)
                self.logger.log(metric_name='loss_supervised', value=loss_supervised.item())
                self.logger.log(metric_name='loss_generator', value=loss_generator.item())
                self.logger.log(metric_name='loss_discriminator', value=loss_discriminator.item())
                self.logger.log(metric_name='loss_fft_generator', value=loss_fft_generator.item())
                self.logger.log(metric_name='loss_fft_discriminator', value=loss_fft_discriminator.item())
            # Save models and optimizer
            if epoch % save_models_after_n_epochs == 0:
                # Save models
                torch.save(self.generator_network.state_dict(), 'generator_network_{}.pt'.format(epoch))
                torch.save(self.discriminator_network.state_dict(), 'discriminator_network_{}.pt'.format(epoch))
                torch.save(self.fft_discriminator_network.state_dict(), 'fft_discriminator_network_{}.pt'.format(epoch))
                # Save optimizers
                torch.save(self.generator_network_optimizer, 'generator_network_optimizer_{}.pt'.format(epoch))
                torch.save(self.discriminator_network_optimizer, 'discriminator_network_optimizer_{}.pt'.format(epoch))
                torch.save(self.fft_discriminator_network_optimizer,
                           'fft_discriminator_network_optimizer_{}.pt'.format(epoch))
            if epoch % validate_after_n_epochs == 0:
                # Validation
                self.validate()
                # Log validation epoch
                self.logger.log(metric_name='validation_epoch', value=epoch)
            # Save logs
            self.logger.save_metrics(self.path_save_metrics)
        # Close progress bar
        self.progress_bar.close()

    @torch.no_grad()
    def validate(self,
                 validation_metrics: Tuple[Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]]
                 = (nn.L1Loss(reduction='mean'), nn.MSELoss(reduction='mean'), misc.psnr, misc.ssim)) -> None:
        # Generator model to device
        self.generator_network.to(self.device)
        # Generator into eval mode
        self.generator_network.eval()
        # Init dict to store metrics
        metrics = dict()
        # Main loop
        for input, label, new_sequence in self.validation_dataloader:
            # Data to device
            input = input.to(self.device)
            label = label.to(self.device)
            # Reset recurrent tensor
            if bool(new_sequence):
                self.generator_network.reset_recurrent_tensor()
            # Make prediction
            prediction = self.generator_network(input)
            # Calc validation metrics
            for validation_metric in validation_metrics:
                # Calc metric
                metric = validation_metric(prediction, label)
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
        # Average metrics and save them in logs
        for metric_name in metrics:
            self.logger.log(metric_name=metric_name, value=float(np.mean(metrics[metric_name])))

    @torch.no_grad()
    def test(self) -> None:
        pass

    @torch.no_grad()
    def inference(self) -> None:
        pass
