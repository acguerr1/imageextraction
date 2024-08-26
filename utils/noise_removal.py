# noise_removal.py
import cv2
import numpy as np
from pyzbar import pyzbar

def remove_borders(image, white_threshold=170):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    left_border, right_border = 0, w - 1
    top_border, bottom_border = 0, h - 1

    # Find left border
    for i in range(w):
        col = gray[:, i]
        if np.mean(col) > white_threshold:
            left_border = i
            break

    # Find right border
    for i in range(w - 1, -1, -1):
        col = gray[:, i]
        if np.mean(col) > white_threshold:
            right_border = i
            break

    # Find top border
    for i in range(h):
        row = gray[i, :]
        if np.mean(row) > white_threshold:
            top_border = i
            break

    # Find bottom border
    for i in range(h - 1, -1, -1):
        row = gray[i, :]
        if np.mean(row) > white_threshold:
            bottom_border = i
            break

    cropped_image = image[top_border:bottom_border, left_border:right_border]
    return cropped_image

def detect_barcodes(image):
    """
    Detects barcodes in an image.

    :param image: Input image in OpenCV format (numpy array).
    :return: Boolean if barcode was detected
    """
    barcode_detected = False
    barcodes = pyzbar.decode(image)

    if barcodes:
        barcode_detected = True

    return barcode_detected

