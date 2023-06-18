import cv2
import numpy as np
import os
from typing import Tuple


class ImageMeanSTD:
    def __init__(self, path: str) -> None:
        self.path = path


    def calculate(self) -> Tuple[np.ndarray, np.ndarray]: 
        # Define the path to the directory containing your image dataset
        # dataset_path = '/path/to/dataset'
        dataset_path = self.path
        # Initialize variables to accumulate the channel-wise sums
        channel_sums = np.zeros(3)
        channel_sums_squared = np.zeros(3)
        count = 0

        # Iterate over the images in the dataset
        for image_file in os.listdir(dataset_path):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                # Load the image
                image = pad_image(os.path.join(dataset_path, image_file))
                
                # Normalize the image pixel values to the range [0, 1]
                normalized_image = image / 255.0

                # Calculate the sums of each channel
                channel_sums += np.sum(normalized_image, axis=(0, 1))
                channel_sums_squared += np.sum(normalized_image ** 2, axis=(0, 1))

                # Increment the count
                count += image.shape[0] * image.shape[1]

        # Calculate the means for each channel
        channel_means = channel_sums / count

        # Calculate the standard deviations for each channel
        channel_stds = np.sqrt((channel_sums_squared / count) - (channel_means ** 2))
        return channel_means, channel_stds

    def calculate_mean(self, path: str) -> np.ndarray:
        mean, std = self.calculate(path)
        return mean

    def calculate_std(self, path: str) -> np.ndarray:
        mean, std = self.calculate(path)
        return std


def pad_image(path, output_size = 224, padding_ratio = 0.9):
    # Load the image
    image = cv2.imread(path)

    # Get the dimensions of the original image
    height, width, _ = image.shape

    unpadded_width = int(output_size * padding_ratio)

    # Calculate the scale factor for resizing
    scale = unpadded_width / width

    # Calculate the new height while maintaining the aspect ratio
    unpadded_height = int(height * scale)

    # Resize the image using the calculated dimensions
    resized_image = cv2.resize(image, (unpadded_width, unpadded_height))

    height, width, _ = resized_image.shape

    # Define the desired output size
    desired_size = output_size

    # Calculate the padding required on each side
    padding_height = (desired_size - height) // 2
    padding_width = (desired_size - width) // 2

    # Create a new blank image with the desired size and black color
    padded_image = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)

    # Calculate the positions to place the original image in the center
    start_height = padding_height
    end_height = padding_height + height
    start_width = padding_width
    end_width = padding_width + width

    # Place the original image in the center of the padded image
    padded_image[start_height:end_height, start_width:end_width, :] = resized_image

    return padded_image