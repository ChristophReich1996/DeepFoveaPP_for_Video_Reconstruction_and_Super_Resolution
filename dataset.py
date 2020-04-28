from typing import Tuple

import torch
from torch.utils.data.dataset import Dataset
from torch.nn.functional import interpolate
from torch.nn.functional import pad
import torchvision.transforms.functional as tf
from PIL import Image
import numpy as np
import os


class REDS(Dataset):
    """
    This class implements a video dataset for super resolution
    """

    def __init__(self, path: str = '/home/creich/REDS/train/train_sharp', number_of_frames: int = 6,
                 overlapping_frames: int = 2, frame_format='png') -> None:
        """
        Constructor method
        :param path: (str) Path to data
        :param number_of_frames: (int) Number of frames in one dataset element
        :param overlapping_frames: (int) Number of overlapping frames of two consecutive dataset elements
        :param frame_format: (str) Frame format to detect frames in path
        """
        # Call super constructor
        super(REDS, self).__init__()
        # Save arguments
        self.number_of_frames = number_of_frames
        # Init previously loaded frames
        self.previously_loaded_frames = None
        # Init list to store all path to frames
        self.data_path = []
        # Get all objects in path an search for video folders
        for video in sorted(os.listdir(path=path)):
            # Case that object in path is a folder
            if os.path.isdir(os.path.join(path, video)):
                # Init frame counter
                frame_counter = 0
                # Init frame index
                frame_index = 0
                # Iterate over all frames in video folder
                while frame_index < len(os.listdir(path=os.path.join(path, video))):
                    # Get current frame name
                    current_frame = sorted(os.listdir(path=os.path.join(path, video)))[frame_index]
                    # Check object is a frame of the desired format
                    if frame_format in current_frame:
                        # Add new list to data path in case of a new frame sequence
                        if frame_counter == 0:
                            self.data_path.append([])
                        # Add frame to last data path under list
                        self.data_path[-1].append(os.path.join(path, video, current_frame))
                        # Increment frame counter
                        frame_counter += 1
                        # Reset frame counter if number of frames in one element are reached
                        if frame_counter == number_of_frames:
                            frame_counter = 0
                            # Decrement frame index by the number of overlapping frames
                            frame_index -= overlapping_frames
                        # Increment frame index
                        frame_index += 1
                # Remove last list element of data_path if number of frames is not matched
                if len(self.data_path[-1]) != number_of_frames:
                    del self.data_path[-1]

    def __len__(self) -> int:
        """
        Method returns the length of the dataset
        :return: (int) Length of the dataset
        """
        return len(self.data_path)

    @torch.no_grad()
    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Method returns one instance of the dataset for a given index
        :param item: (int) Index to get element
        :return: (Tuple[torch.Tensor, torch.Tensor]) Low res sequence, high res sequence, new video flag
        """
        # Check if current frame sequence is a new video sequence
        if self.previously_loaded_frames is None or self.previously_loaded_frames[0].split('/')[-2] != \
                self.data_path[item][0].split('/')[-2]:
            new_video = True
        else:
            new_video = False
        # Set current data path to previously loaded frames
        self.previously_loaded_frames = self.data_path[item]
        # Load frames
        frames_low_res = []
        frames_label = []
        for frame in self.data_path[item]:
            # Load images as PIL image, and convert to tensor
            image = tf.to_tensor(Image.open(frame))
            # Normalize image to a mean of zero and a std of one
            image = image.sub_(image.mean()).div_(image.std())
            # Downsampled frames
            image_low_res = interpolate(image[None], scale_factor=0.25, mode='bilinear', align_corners=False)[0]
            # Crop normal image
            image = image[:, :, 128:-128]
            image = pad(image[None], pad=[0, 0, 24, 24], mode="constant", value=0)[0]
            # Crop low res masked image
            image_low_res = image_low_res[:, :, 32:-32]
            image_low_res = pad(image_low_res[None], pad=[0, 0, 6, 6], mode="constant", value=0)[0]
            # Add to list
            frames_low_res.append(image_low_res)
            # Add to list
            frames_label.append(image)
        # Concatenate frames to tensor of shape (3 * number of frames, height (/ 4), width (/ 4))
        frames_low_res = torch.cat(frames_low_res, dim=0)
        frames_label = torch.cat(frames_label, dim=0)
        return frames_low_res, frames_label, new_video


class REDSFovea(REDS):
    """
    Class implements the REDS dataset with a fovea sampled low resolution input sequence and a high resolution label
    """

    def __init__(self, path: str = '/home/creich/REDS/train/train_sharp') -> None:
        # Call super constructor
        super(REDSFovea, self).__init__(path=path)
        # Init probability of mask
        self.p_mask = None

    def get_mask(self, shape: Tuple[int, int]) -> torch.Tensor:
        """
        Method returns a binary fovea mask
        :param new_video: (bool) Flag if a new video is present
        :param shape: (Tuple[int, int]) Image shape
        :return: (torch.Tensor) Fovea mask
        """
        if self.p_mask is None:
            # Get all indexes of image
            indexes = np.stack(np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0])), axis=0).reshape((2, -1))
            # Make center point
            center = np.array(
                [np.random.uniform(50, shape[1] - 50), np.random.uniform(50, shape[0] - 50)])
            # Calc euclidean distances
            distances = np.linalg.norm(indexes - center.reshape((2, 1)), ord=2, axis=0)
            # Calc probability mask
            self.p_mask = np.where(distances < 15, 0.98, 0.0) + np.where(distances > 40, 0.05, 0.0) \
                          + np.where(np.logical_and(distances >= 15, distances <= 40), -0.031 * distances + 1.445, 0.0)
        # Make mask
        mask = torch.from_numpy(self.p_mask >= np.random.uniform(low=0, high=1, size=shape[0] * shape[1])).reshape(
            (shape[0], shape[1]))
        return mask.float()

    @torch.no_grad()
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Get item method returns the fovea masked downsampled frame sequence, the high resolution sequence, and a bool
        if the new sequence is the start of a new video
        :param item: (int) Index to get element
        :return: (Tuple[torch.Tensor, torch.Tensor]) Low res fovea sampled sequence, high res sequence, new video flag
        """
        # Check if current frame sequence is a new video sequence
        if self.previously_loaded_frames is None or self.previously_loaded_frames[0].split('/')[-2] != \
                self.data_path[item][0].split('/')[-2]:
            new_video = True
            self.p_mask = None
        else:
            new_video = False
        # Set current data path to previously loaded frames
        self.previously_loaded_frames = self.data_path[item]
        # Load frames
        frames_masked = []
        frames_label = []
        for frame in self.data_path[item]:
            # Load images as PIL image, and convert to tensor
            image = tf.to_tensor(Image.open(frame))
            # Normalize image to a mean of zero and a std of one
            image = image.sub_(image.mean()).div_(image.std())
            # Downsampled frames
            image_low_res = interpolate(image[None], scale_factor=0.25, mode='bilinear', align_corners=False)[0]
            # Apply mask to image
            image_low_res_masked = image_low_res * self.get_mask(shape=(image_low_res.shape[1], image_low_res.shape[2]))
            # Crop normal image
            image = image[:, :, 128:-128]
            image = pad(image[None], pad=[0, 0, 24, 24], mode="constant", value=0)[0]
            # Crop low res masked image
            image_low_res_masked = image_low_res_masked[:, :, 32:-32]
            image_low_res_masked = pad(image_low_res_masked[None], pad=[0, 0, 6, 6], mode="constant", value=0)[0]
            # Add to list
            frames_masked.append(image_low_res_masked)
            # Add to list
            frames_label.append(image)
        # Concatenate frames to tensor of shape (3 * number of frames, height (/ 4), width (/ 4))
        frames_masked = torch.cat(frames_masked, dim=0)
        frames_label = torch.cat(frames_label, dim=0)
        return frames_masked, frames_label, new_video


class PseudoDataset(Dataset):
    """
    This class implements a pseudo dataset to test the implemented architecture
    """

    def __init__(self, length: int = 100) -> None:
        """
        Constructor method
        :param length: (int) Pseudo dataset length
        """
        # Call super constructor
        super(PseudoDataset, self).__init__()
        # Save length parameter
        self.length = length

    def __len__(self) -> int:
        """
        Method to get length of the dataset
        :return: (int) Length
        """
        return self.length

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method returns a tensor of the shape (rgb * 16 frames, height, width) and the corresponding high res. label
        :param item: (int) Index
        :return: (torch.Tensor) Pseudo image sequence and corresponding label
        """
        if item >= len(self):
            raise IndexError
        return torch.ones([3 * 16, 64, 64], dtype=torch.float), torch.ones([3 * 16, 256, 256], dtype=torch.float)


if __name__ == '__main__':
    dataset = REDSFovea()
    frames_input, frames_label, _ = dataset[1000]

    import matplotlib.pyplot as plt

    plt.imshow(frames_input.numpy().transpose(1, 2, 0)[:, :, 0:3])
    plt.savefig('Input.png')
    plt.show()
