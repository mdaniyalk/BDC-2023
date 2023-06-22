import cv2
import numpy as np
import imgaug.augmenters as iaa
from tqdm import tqdm 
from typing import Tuple, Union


class AugmentImage:
    """
    A class for performing image augmentation on a set of images.

    This class applies various augmentation techniques to a given set of images to generate augmented images for data augmentation purposes. It utilizes the `imgaug` library to apply transformations such as rotation, skewing, blur, noise addition, contrast normalization, and hue and saturation multiplication.

    Example usage:
    image_array = np.array([...])  # Array of input images
    label_array = np.array([...])  # Array of corresponding labels
    augmenter = AugmentImage(image_array, label_array, num_augmentations=3)
    augmented_images, augmented_labels = augmenter.transform()
    """


    def __init__(self, 
                 image_array: Union[np.ndarray, list], 
                 label_array: Union[np.ndarray, list], 
                 num_augmentations: int):
        """
        Initializes an image augmentation object.

        Args:
        - image_array (np.ndarray | list): The array of input images.
        - label_array (np.ndarray | list): The array of corresponding labels.
        - num_augmentations (int): The number of augmentations to generate for each image.

        This class performs image augmentation on a given set of images. It takes an array of input images (`image_array`), an array of corresponding labels (`label_array`), and the number of augmentations to generate for each image (`num_augmentations`).

        The image augmentation is performed using the `imgaug` library. Several augmentation techniques are applied sequentially using the `Sequential` function from `imgaug.augmenters` module. These techniques include affine transformations (rotation and skewing), Gaussian blur, additive Gaussian noise, pixel value multiplication, contrast normalization, and hue and saturation multiplication.

        The augmented images and their corresponding labels are stored in `self.output_images` and `self.output_labels` respectively.

        Example usage:
        image_array = np.array([...])  # Array of input images
        label_array = np.array([...])  # Array of corresponding labels
        augmenter = AugmentImage(image_array, label_array, num_augmentations=3)
        augmented_images, augmented_labels = augmenter.transform()
        """


        self.image_array = image_array
        self.label_array = label_array
        self.num_augmentations = num_augmentations
        self.augmenter = iaa.Sequential([
            iaa.Affine(rotate=(-20, 20)),  # Rotate the image by a random angle between -10 and 10 degrees
            iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply random Gaussian blur with a sigma between 0 and 1.0
            iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255)),  # Add random Gaussian noise to the image
            iaa.Multiply((0.8, 1.2)),  # Multiply pixel values by a random value between 0.8 and 1.2
            iaa.ContrastNormalization((0.8, 1.2)),  # Apply contrast normalization to the image
            iaa.MultiplyHueAndSaturation((0.8, 1.2), per_channel=True),  # Multiply hue and saturation by a random value
            iaa.Affine(shear=(-15, 15)),  # Add skewing with shear transformation
            iaa.AdditiveLaplaceNoise(scale=(0, 0.025 * 255)),  # Additional noise
            iaa.GammaContrast(gamma=(0.8, 1.2)),  # Random gamma and contrast adjustment
            ])
        self.output_images = []
        self.output_labels = []



    def transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies image augmentation to the input images.

        Returns:
        - output_images (np.ndarray): Augmented images.
        - output_labels (np.ndarray): Corresponding labels.

        This method applies image augmentation to the input images using the configured augmentation techniques. It iterates over each image in `self.image_array`, adds the original image and its corresponding label to the output lists, and then applies the specified number of augmentations. Each augmentation creates a new augmented image using the `augment_image` method of the `self.augmenter` object.

        Finally, the augmented images and their corresponding labels are converted to NumPy arrays and returned.

        Example usage:
        augmenter = AugmentImage(image_array, label_array, num_augmentations=3)
        augmented_images, augmented_labels = augmenter.transform()
        """


        for idx, image in enumerate(tqdm(self.image_array, desc='Augement Images')):
            # self.output_images.append(image)
            label = self.label_array[idx]
            # self.output_labels.append(label)
            for _ in range(self.num_augmentations):
                # Apply augmentation to the image
                augmented_image = self.augmenter.augment_image(image)
                # Append the augmented image and its corresponding label to the lists
                self.output_images.append(augmented_image)
                self.output_labels.append(label)

        self.output_images = np.asarray(self.output_images)
        self.output_labels = np.asarray(self.output_labels)
        return self.output_images, self.output_labels
        
        