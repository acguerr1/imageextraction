# utils.py - Utility Functions and Helpers

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
This module contains various utility functions and helper classes used across the project.

Author: Krishna Kamath
Date: 31st July 2023
"""



def is_page_empty(image_path):
    # Checks if image is all white pixels (no image contained in page)

    # Open the image using Pillow
    image = Image.open(image_path)

    # Convert the image to grayscale for comparison with (255, 255, 255) white color
    image_gray = image.convert("L")

    # Check if all pixels are white (255)
    width, height = image_gray.size
    all_white = all(image_gray.getpixel((x, y)) == 255 for x in range(width) for y in range(height))

    return all_white


def file_exists(file_path):
    # Given a file path, checks whether it exists
    return os.path.exists(file_path)

def get_image_from_path(image_path):
    # Load the image from the file path
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Failed to load image from {image_path}. Check if the file exists or the image format is supported.")
        return None

    return image



def get_horizontal_profile(image):
    # Get horizontal profile of image
    return np.sum(image, axis=1)



def get_vertical_profile(image):
    # Get vertical profile of image
    return np.sum(image, axis=0)


def plot_profiles(image):
    ## Plotting the horizontal and vertical profiles

    horizontal_profile = get_horizontal_profile(image)
    vertical_profile = get_vertical_profile(image)
    plt.subplot(2, 1, 1)
    plt.plot(horizontal_profile)
    plt.title('Horizontal Profile')

    plt.subplot(2, 1, 2)
    plt.plot(vertical_profile)
    plt.title('Vertical Profile')

    plt.tight_layout()
    plt.show()



def crop_images(image_path, output_path):
    # given images with whitespace, crops the images
    # Currently works for 1 image. Add for multiple images

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    th, threshed = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    
    # to add: multiple boxes
    x, y, w, h = cv2.boundingRect(cnt)
    dst = img[y:y + h, x:x + w]
    cv2.imwrite(output_path, dst)