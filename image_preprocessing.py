from PIL import Image, ImageEnhance
import cv2
import numpy as np

class ImagePreprocessor:
    """
    Class responsible for pre-processing images before OCR.
    Follows the Single Responsibility Principle.
    """

    def to_grayscale(self, image_path):
        """
        Converts the image to grayscale.
        
        :param image_path: Path to the image file
        :return: Grayscale image
        """
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

    def denoise(self, gray_image):
        """
        Applies noise reduction to the image.
        
        :param gray_image: Grayscale image
        :return: Denoised image
        """
        denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)
        return denoised_image

    def threshold(self, denoised_image):
        """
        Applies binary thresholding to the image.
        
        :param denoised_image: Denoised grayscale image
        :return: Thresholded (binary) image
        """
        _, thresholded_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded_image

    def enhance_contrast(self, image_path):
        """
        Enhances the contrast of the image using Pillow.
        
        :param image_path: Path to the image file
        :return: Image with enhanced contrast
        """
        image = Image.open(image_path)
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(2)  # Adjust the contrast factor here
        return enhanced_image
