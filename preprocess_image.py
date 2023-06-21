import numpy as np
import cv2
from typing import Union, Literal
from tqdm import tqdm


class PreprocessImage:
    """
    Class for preprocessing image data.

    This class provides methods for preprocessing image data, including resizing, padding, conversion to black and white, segmentation of white regions, segmentation of number images, and conversion to grayscale.

    Example usage:
    preprocessor = PreprocessImage()
    preprocessor.bw_image(masked_img=[0, 2])
    preprocessor.segment_white(masked_img=[1])
    preprocessor.segment_number_image()
    preprocessor.gray_image(masked_img=[1, 2])
    """


    def __init__(self, 
                 img_paths: Union[np.ndarray, None], 
                 output_pixel: Union[int, Literal[224]],
                 padding_ratio: Union[float, Literal[1]],
                 path_prefix: Union[str, None], 
                 include_original: Union[bool, None]):
        """
        Preprocesses images for further analysis and feature extraction.

        Parameters:
        - img_paths (ndarray | None): Array of image paths.
        - output_pixel (int | 224): The desired output size of the images.
        - padding_ratio (float | 0.9): The ratio used for padding the images.
        - path_prefix (str | None): Prefix for the image paths, if applicable.
        - include_original (bool | None): Indicates whether to include the original images in the output.

        This class performs various preprocessing operations on images to prepare them for further analysis and feature extraction. It takes an array of image paths as input and performs the specified operations on each image.

        The `output_pixel` parameter determines the desired output size of the images. The `padding_ratio` parameter specifies the ratio used for padding the images to the desired size. The `path_prefix` parameter allows for adding a prefix to the image paths, if necessary. The `include_original` parameter determines whether to include the original images in the output.

        Example usage:
        img_paths = np.array(["image1.jpg", "image2.jpg", "image3.jpg"])
        preprocessor = PreprocessImage(img_paths=img_paths, output_pixel=224, padding_ratio=0.9, include_original=True)
        preprocessor.bw_image(masked_img=[0, 2])
        preprocessor.segment_white(masked_img=[1])
        preprocessor.segment_number_image()
        preprocessor.gray_image(masked_img=[1, 2])
        """


        self.output_pixel = output_pixel
        self.padding_ratio = padding_ratio
        self.include_original = include_original
        self.output = []
        img_array = []
        self.path_prefix = path_prefix
        for path in tqdm(img_paths, desc="Initial Preprocess: Add Padding"):
            if path_prefix is not None:
                path = f'{path_prefix}/{path}'
            img_array.append(self.image_padding(path))
        self.img_array = np.asarray(img_array)




    def image_padding(self, path: str) -> np.ndarray:
        """
        Perform padding on the image.

        Parameters:
        - path (str): The path to the image.

        Returns:
        - padded_image (ndarray): The padded image.

        This method loads the image from the specified path and adds padding to it. It resizes the image while maintaining the aspect ratio and then adds padding to achieve the desired output size.

        The method calculates the necessary padding on each side based on the desired output size and the resized image's dimensions. It creates a new blank image with the desired size and black color, and then places the resized image in the center by calculating the appropriate positions.

        Example usage:
        image_path = 'image.jpg'
        preprocess = PreprocessImage()
        padded_image = preprocess.image_padding(image_path)
        """


        # Load the image
        image = cv2.imread(path)

        # Get the dimensions of the original image
        height, width, _ = image.shape

        unpadded_width = int(self.padding_ratio * self.output_pixel)

        # Calculate the scale factor for resizing
        scale = unpadded_width / width

        # Calculate the new height while maintaining the aspect ratio
        unpadded_height = int(height * scale)

        # Resize the image using the calculated dimensions
        resized_image = cv2.resize(image, (unpadded_width, unpadded_height))

        height, width, _ = resized_image.shape

        # Define the desired output size
        desired_size = self.output_pixel

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

        padded_image[start_height:end_height, start_width:end_width, :] = resized_image
        return padded_image



    def bw_image(self, 
                 masked_img: Union[np.ndarray, list, None])  -> np.ndarray :
        """
        Convert images to black and white (BW) and high contrast images.

        Parameters:
        - masked_img (ndarray | list | None): Masked images to exclude from conversion.

        Returns:
        - output (ndarray): The converted images.

        This method converts the images to black and white (BW) and high contrast images. It uses adaptive thresholding to convert the grayscale images to binary images and then converts them back to RGB.

        The `masked_img` parameter allows excluding specific images from the conversion if provided.

        Example usage:
        preprocess = PreprocessImage()
        bw_images = preprocess.bw_image()
        """


        output = []
        for idx, image in enumerate(tqdm(self.img_array,
                                         desc='Preprocess: Convert to BW & High Contrast Image')):
            if not idx in masked_img or masked_img is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=2)
                img_gray = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                output.append(img_gray)

        output = np.asarray(output)
        if self.include_original:
            output = np.concatenate((self.img_array, output), axis=0)
        return output
    


    def segment_white(self, 
                      masked_img: Union[np.ndarray, list, None], 
                      img: Union[np.ndarray, None]) -> np.ndarray:
        
        """
        Segment images based on white regions.

        Parameters:
        - masked_img (ndarray | list | None): Masked images to exclude from segmentation.
        - img (ndarray | None): Image to segment (optional).

        Returns:
        - output (ndarray): The segmented images.

        This method segments the images based on white regions. It converts the images to the HSV color space, sets the saturation channel to zero, converts them back to the BGR color space, and increases the contrast. It then applies a binary threshold to isolate white regions and performs additional operations to enhance the segmentation.

        The `masked_img` parameter allows excluding specific images from segmentation if provided. The `img` parameter can be used to segment a single image instead of the entire dataset.

        Example usage:
        preprocess = PreprocessImage()
        segmented_images = preprocess.segment_white()
        """


        def segment_white_wrapper(image: np.ndarray) -> np.ndarray:
            """
            Perform white region segmentation on an image.

            Parameters:
            - image (ndarray): The input image.

            Returns:
            - result (ndarray): The segmented image.

            This function applies white region segmentation to the input image. It converts the image to the HSV color space, sets the saturation channel to zero, and converts the image back to the BGR color space. It then increases the contrast using cv2.convertScaleAbs() with specified contrast and brightness factors.

            The function converts the resulting image to grayscale, threshold it to obtain a binary image, and then performs additional operations to enhance the white regions. Finally, it converts the image back to RGB color space.

            Example usage:
            image = cv2.imread('image.jpg')
            segmented_image = segment_white_wrapper(image)
            """


            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Set the saturation channel (index 1) to zero
            hsv[:, :, 1] = 0

            # Convert the image back to BGR color space
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Increase the contrast using cv2.convertScaleAbs()
            alpha = 1.5  # Contrast factor
            beta = 0  # Brightness offset
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
            
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            
            gray_ = np.where(result > 200, 255, 0).astype(np.uint8)

            for i in range(3):
                gray = cv2.bitwise_and(gray_, result)
                gray = cv2.add(gray_, gray)
                result = gray

            gray = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            return gray

        if img is not None:
            return segment_white_wrapper(img)
        
        output = []
        for idx, image in enumerate(tqdm(self.img_array, 
                                    desc='Preprocess: BW Segmented Image')):
            if not idx in masked_img or masked_img is not None:
                gray = segment_white_wrapper(image)
                output.append(gray)

        output = np.asarray(output)
        if self.include_original:
            output = np.concatenate((self.img_array, output), axis=0)
        return output
    


    def segment_number_image(self, 
                             masked_img: Union[np.ndarray, list, None]) -> np.ndarray:
        """
        Segment images containing numbers.

        Parameters:
        - masked_img (ndarray | list | None): Masked images to exclude from segmentation.

        Returns:
        - output (ndarray): The segmented images.

        This method segments the images containing numbers. It first applies the white region segmentation using the `segment_white` method and then applies additional operations such as contour detection and masking to isolate the number regions.

        The `masked_img` parameter allows excluding specific images from segmentation if provided.

        Example usage:
        preprocess = PreprocessImage()
        segmented_number_images = preprocess.segment_number_image()
        """


        output = []
        for idx, image in enumerate(tqdm(self.img_array, 
                                    desc='Preprocess: Segment Number Image')):
            if not idx in masked_img or masked_img is not None:
                image = self.segment_white(img = image)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Create a mask of the same shape as the image
                mask = np.zeros_like(gray)

                # Iterate through the contours and draw filled contours on the mask
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Adjust the minimum area threshold as needed
                        cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                # Apply the mask to the image
                result = cv2.bitwise_and(image, image, mask=mask)
                output.append(result)

        output = np.asarray(output)
        if self.include_original:
            output = np.concatenate((self.img_array, output), axis=0)
        return output

 
    
    def gray_image(self, masked_img: Union[np.ndarray, list, None]) -> np.ndarray:
        """
        Convert images to grayscale.

        Parameters:
        - masked_img (ndarray | list | None): Masked images to exclude from conversion.

        Returns:
        - output (ndarray): The converted images.

        This method converts the images to grayscale. It uses the cv2.cvtColor function to convert the images from BGR to grayscale and applies contrast adjustments using the cv2.convertScaleAbs function.

        The `masked_img` parameter allows excluding specific images from the conversion if provided.

        Example usage:
        preprocess = PreprocessImage()
        gray_images = preprocess.gray_image()
        """


        output = []
        for idx, image in enumerate(tqdm(self.img_array, 
                                    desc='Preprocess: Convert to Grayscale Image')):
            if not idx in masked_img or masked_img is not None:
                result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                alpha = 1.2  # Contrast factor
                beta = 0  # Brightness offset
                result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                output.append(result)
        output = np.asarray(output)
        if self.include_original:
            output = np.concatenate((self.img_array, output), axis=0)
        return output


            

