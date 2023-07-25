import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_horizontal_profile(image):
    return np.sum(image, axis=1)

def get_vertical_profile(image):
    return np.sum(image, axis=0)




# Load the image using OpenCV
image_path = "out1.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print(f"Failed to load image from {image_path}")
else:
    # Calculate horizontal and vertical profiles
    horizontal_profile = get_horizontal_profile(image)
    vertical_profile = get_vertical_profile(image)

    # Plotting the horizontal and vertical profiles
    plt.subplot(2, 1, 1)
    plt.plot(horizontal_profile)
    plt.title('Horizontal Profile')

    plt.subplot(2, 1, 2)
    plt.plot(vertical_profile)
    plt.title('Vertical Profile')

    plt.tight_layout()
    plt.show()
