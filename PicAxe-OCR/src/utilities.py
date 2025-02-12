# utilitpies.py
import os  # For file path operations, checking existence, and creating directories
import sys  # For system-specific parameters and functions
import shutil  # For high-level file operations
import argparse  # For parsing command-line arguments
import subprocess  # For running external commands
import time  # For time-related functions
import gc  # For garbage collection

import cv2 as cv  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations and array manipulations
import math  # For mathematical functions
from PIL import Image  # For image operations
from pdf2image import convert_from_path  # For converting PDFs to images
from matplotlib import pyplot as plt  # For plotting images and graphs
import re  # For regular expressions
import statistics  # For statistical calculations
import logging  # For logging messages

# Ensure the correct path is added for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2 as cv
import numpy as np

def is_image_file(filename):
    # Check if the file is an image based on its extension and if it's a valid file
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    return os.path.isfile(filename) and any(filename.lower().endswith(ext) for ext in valid_extensions)

# Check if a file is already processed
# This function checks if a file has already been processed by looking for the output file in the directory.
def is_file_already_processed(base_name, directory):
    expected_filename = f"{base_name}.png"
    return os.path.isfile(os.path.join(directory, expected_filename))

def filter_image_files(pdf_img_files_all, output_dir, ext='.png', keywords=''):
    """Filter image files based on keywords or return all if no keywords are specified."""
    pdf_img_files = []
    for image_path in pdf_img_files_all:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_file_path = f'{output_dir}/{os.path.splitext(base_name)[0]}{ext}'

        # Check if file has already been processed
        if os.path.exists(output_file_path) or not is_image_file(image_path):
            continue

        # Check if any of the keywords are in the base name or if no keywords are provided
        if not keywords or any(keyword.lower() in base_name.lower() for keyword in keywords):
            pdf_img_files.append(image_path)

    return pdf_img_files

def open_horizontally(image, iters=3):
    """Apply horizontal morphological opening to remove horizontal lines."""
    bw = image.copy()
    if len(image.shape) != 2:
        bw = binarize_img(image)  # Binarize image if not already binary

    kernel_size = (45, 1)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)  # Create a horizontal kernel
    bw_closed = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel, iterations=iters)  # Apply morphological opening
    return bw_closed

def open_vertically(image, iters=3):
    """Apply vertical morphological opening to remove vertical lines."""
    bw = image.copy()
    if len(image.shape) != 2:
        bw = binarize_img(image)  # Binarize image if not already binary

    kernel_size = (1, 45)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)  # Create a vertical kernel
    bw_closed = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel, iterations=iters)  # Apply morphological opening
    return bw_closed

def rm_h_lines_margin(image):
    """Remove horizontal lines near the top and bottom margins of the image."""
    imgcp = image.copy()
    bw_closed_h = open_horizontally(imgcp)  # Open image horizontally to find horizontal lines
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (45, 2))  # Define a kernel for dilation
    contours, _ = cv.findContours(cv.bitwise_not(bw_closed_h), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = [cnt for cnt in contours if (cv.boundingRect(cnt)[2] / cv.boundingRect(cnt)[3]) >= 2.0]  # Filter contours by aspect ratio

    # Initialize mask
    height, width = imgcp.shape[:2]
    if len(imgcp.shape) == 2:  # Grayscale image
        mask = np.zeros_like(imgcp)  # Initialize mask for grayscale image
    else:
        mask = np.zeros((height, width), dtype=np.uint8)  # Initialize mask for color image
    
    # Calculate margins
    top_margin = get_closest_object_distance(bw_closed_h, 'top')  # Get top margin distance
    bottom_margin = get_closest_object_distance(bw_closed_h, 'bottom')  # Get bottom margin distance
    margin_h = max(int(0.12 * height), top_margin, bottom_margin)  # Determine margin height
    margin_cnts = []
    
    # Identify top and bottom horizontal lines
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if (y < margin_h or y + h > height - margin_h):  # Check if contour is within margin
            margin_cnts.append(cnt)
    
    # Sort contours by the y-coordinate
    margin_cnts = sorted(margin_cnts, key=lambda cnt: cv.boundingRect(cnt)[1])  # Sort contours by y-coordinate
    if margin_cnts:
        topmost_cnt = margin_cnts[0]  # Identify topmost contour
        bottommost_cnt = margin_cnts[-1] if len(margin_cnts) > 1 else None  # Identify bottommost contour
        # Ensure they are distinct
        margin_cnts = [topmost_cnt] if topmost_cnt is bottommost_cnt else [topmost_cnt, bottommost_cnt]  # Keep distinct contours
        margin_cnts = [cnt for cnt in margin_cnts if cnt is not None]  # Filter out None
    
    # Create mask
    cv.drawContours(mask, margin_cnts, -1, 255, thickness=cv.FILLED)  # Draw contours on the mask
    mask = cv.dilate(mask, kernel, iterations=15)  # Dilate the mask to cover the lines
    
    # Remove horizontal lines by setting those areas to white
    if len(imgcp.shape) == 2:  # Grayscale image
        imgcp[mask == 255] = 255  # Set mask areas to white
    else:  # Color image
        imgcp[mask == 255] = [255, 255, 255]  # Set mask areas to white in color image
    
    return imgcp, mask  # Return processed image and mask

def rm_v_lines_margin(image):
    """Remove vertical lines near the left and right margins of the image."""
    imgcp = image.copy()
    bw_closed_v = open_vertically(imgcp)  # Open image vertically to find vertical lines
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 45))  # Define a kernel for dilation
    contours, _ = cv.findContours(bw_closed_v, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = [cnt for cnt in contours if (cv.boundingRect(cnt)[3] / cv.boundingRect(cnt)[2]) >= 3.0]  # Filter contours by aspect ratio

    # Initialize mask
    height, width = imgcp.shape[:2]
    if len(imgcp.shape) == 2:  # Grayscale image
        mask = np.zeros_like(imgcp)  # Initialize mask for grayscale image
    else:
        mask = np.zeros((height, width), dtype=np.uint8)  # Initialize mask for color image
    
    # Calculate margins
    left_margin = get_closest_object_distance(bw_closed_v, 'left')  # Get left margin distance
    right_margin = get_closest_object_distance(bw_closed_v, 'right')  # Get right margin distance
    margin_w = max(int(0.12 * width), left_margin, right_margin)  # Determine margin width
    
    # Identify left and right vertical lines
    margin_cnts = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if (x < margin_w or x + w > width - margin_w):  # Check if contour is within margin
            margin_cnts.append(cnt)
    
    # Sort contours by the x-coordinate
    margin_cnts = sorted(margin_cnts, key=lambda cnt: cv.boundingRect(cnt)[0])  # Sort contours by x-coordinate
    if margin_cnts:
        leftmost_cnt = margin_cnts[0]  # Identify leftmost contour
        rightmost_cnt = margin_cnts[-1] if len(margin_cnts) > 1 else None  # Identify rightmost contour
        # Ensure they are distinct
        margin_cnts = [leftmost_cnt] if leftmost_cnt is rightmost_cnt else [leftmost_cnt, rightmost_cnt]  # Keep distinct contours
        margin_cnts = [cnt for cnt in margin_cnts if cnt is not None]  # Filter out None
    
    # Create mask
    cv.drawContours(mask, margin_cnts, -1, 255, thickness=cv.FILLED)  # Draw contours on the mask
    mask = cv.dilate(mask, kernel, iterations=15)  # Dilate the mask to cover the lines
    
    # Remove vertical lines by setting those areas to white
    if len(imgcp.shape) == 2:  # Grayscale image
        imgcp[mask == 255] = 255  # Set mask areas to white
    else:  # Color image
        imgcp[mask == 255] = [255, 255, 255]  # Set mask areas to white in color image
    
    return imgcp, mask  # Return processed image and mask

def get_closest_object_distance(bw, from_edge):
    """Calculate the distance from the specified edge to the closest black pixel in a binary image."""
    height, width = bw.shape
    if from_edge == 'top':
        for i in range(height):
            if np.any(bw[i, :] == 0):
                return i  # Return distance from the top edge
    elif from_edge == 'bottom':
        for i in range(height-1, -1, -1):
            if np.any(bw[i, :] == 0):
                return height - i  # Return distance from the bottom edge
    elif from_edge == 'left':
        for i in range(width):
            if np.any(bw[:, i] == 0):
                return i  # Return distance from the left edge
    elif from_edge == 'right':
        for i in range(width-1, -1, -1):
            if np.any(bw[:, i] == 0):
                return width - i  # Return distance from the right edge
    return 0  # Default return if no black pixels are found

def exclude_full_image_bbox(bboxes, image_shape):
    """Exclude bounding boxes that cover the entire image."""
    img_height, img_width = image_shape[:2]
    # Filter out bounding boxes that cover the entire image
    filtered_bboxes = [bbox for bbox in bboxes if not (
        bbox[0] <= 1 and bbox[1] <= 1 and
        bbox[2] >= img_width - 1 and bbox[3] >= img_height - 1)]
    return filtered_bboxes

def clean_corners(bw):
    """Clean the corners of a binary image by masking out the corner areas."""
    height, width = bw.shape
    mask = np.ones_like(bw) * 255  # Initialize a white mask

    # Calculate margins from each edge
    top_margin = get_closest_object_distance(bw, 'top')
    bottom_margin = get_closest_object_distance(bw, 'bottom')
    left_margin = get_closest_object_distance(bw, 'left')
    right_margin = get_closest_object_distance(bw, 'right')

    # Determine the size of the corner areas to mask
    corner_height = max(int(0.12 * height), top_margin, bottom_margin)
    corner_width = max(int(0.15 * width), left_margin, right_margin)

    # Apply black masks to the corners
    mask[:corner_height, :corner_width] = 0  # Top-left corner
    mask[:corner_height, width - corner_width:] = 0  # Top-right corner
    mask[height - corner_height:, :corner_width] = 0  # Bottom-left corner
    mask[height - corner_height:, width - corner_width:] = 0  # Bottom-right corner

    # Clean corners by applying the mask
    cleaned = cv.bitwise_or(bw, cv.bitwise_not(mask))
    return cleaned, mask  # Return the cleaned image and the mask used

def file_exists(filepath):
    """Check if a file exists at the specified path."""
    return os.path.isfile(filepath)  # Return True if file exists, else False

def pdf_to_images(pdf_path):
    """Convert a PDF document to a list of images, each representing a page."""
    images = convert_from_path(pdf_path, 500)  # Convert PDF to images at 500 DPI
    # Convert each image from RGB to BGR format (for OpenCV) and return the list
    return [cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR) for image in images]

def display_n_images(images, titles=None, save_dir=None):
    """Display multiple images in a grid and optionally save them."""
    num_images = len(images)
    cols = 2
    rows = (num_images + cols - 1) // cols  # Calculate the number of rows needed
    plt.figure(figsize=(15, 5 * rows))  # Set the figure size based on the number of rows

    if titles is None:
        titles = [f'Image {i+1}' for i in range(num_images)]  # Generate default titles if none provided

    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)  # Create a subplot for each image
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)  # Display the image, using grayscale if 2D
        plt.title(titles[i])  # Set the title for each image
        plt.axis('off')  # Hide the axis

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
            save_path = os.path.join(save_dir, f"{titles[i].replace(' ', '_')}.png")  # Construct the save path
            cv.imwrite(save_path, cv.cvtColor(image, cv.COLOR_BGR2RGB))  # Save the image in RGB format
            print(f"Saved {save_path}")  # Print the save path

    plt.tight_layout()  # Adjust subplots to fit into figure area
    plt.show()  # Display the figure
    return images  # Return the list of images

def display_2_images(image, final):
    """Display two images side by side for comparison."""
    plt.figure(figsize=(10, 5))  # Set the figure size
    titles = ['Original', 'Final']  # Titles for the images
    images_to_plot = [image, final]  # List of images to display

    for j in range(2):
        plt.subplot(1, 2, j+1)  # Create subplots
        plt.imshow(images_to_plot[j], cmap='gray' if len(images_to_plot[j].shape) == 2 else None)  # Display each image
        plt.title(titles[j])  # Set title
        plt.axis('off')  # Hide the axis

    plt.tight_layout()  # Adjust subplots to fit into figure area
    plt.show()  # Display the figure
    return None  # Return None as no images are returned

def draw_cnts(contours, image):
    """Draw contours on a white image."""
    contour_image = np.full_like(image, 255)  # Create a white image of the same size as the input
    cv.drawContours(contour_image, contours, -1, (0, 255, 0, 255), 2)  # Draw contours in green
    return contour_image  # Return the image with contours

def get_2DConv_kernel(ks=7):
    """Create a 2D convolution kernel with a given size."""
    kernel = np.ones((ks, ks), np.float32) / 9  # Create an averaging kernel
    return kernel  # Return the kernel

def get_kernel(ks=3, ks2=3):
    """Create a rectangular structuring element for morphological operations."""
    return np.ones((ks, ks2), np.uint8)  # Return a kernel filled with ones

def clean_image_filter2D(img):
    """Apply a 2D convolution filter to clean the image."""
    kernel = get_2DConv_kernel(ks=9)  # Get the convolution kernel
    img2d = cv.filter2D(img, -1, kernel)  # Apply the filter to the image
    return img2d  # Return the filtered image

def clean_image_morph(img):
    """Clean the image using morphological operations."""
    kernel = get_2DConv_kernel(ks=5)  # Get the convolution kernel
    img_close = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=5)  # Apply closing operation
    img_open = cv.morphologyEx(img_close, cv.MORPH_OPEN, kernel, iterations=5)  # Apply opening operation
    return img_open  # Return the morphologically cleaned image

def cvsm(image, title=None):
    """Display an image using Matplotlib, converting BGR to RGB if needed."""
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))  # Convert BGR image to RGB for display
    plt.axis('off')  # Hide the axis
    if title:
        plt.title(title)  # Set the title if provided
    plt.show()  # Display the image

def is_alphanumeric(text):
    """Check if a string contains any alphanumeric characters."""
    return bool(re.search(r'[A-Za-z0-9]', text))  # Return True if any alphanumeric characters are found

def calculate_statistics(data_dict):
    """Calculate mean, median, standard deviation, and a threshold for a dictionary of data."""
    statistics_dict = {}
    for key, values in data_dict.items():
        if len(values) < 2:
            # Handle cases with fewer than 2 values
            mean_val = statistics.mean(values) if values else 0
            median_val = statistics.median(values) if values else 0
            statistics_dict[key] = (mean_val, median_val, 0, 0)  # Store mean, median, std, and threshold
            continue
        # Calculate statistics for data with more than 2 values
        mean_val, std_val = statistics.mean(values), statistics.stdev(values)
        median_val, threshold = statistics.median(values), (1.9 * std_val)  # Set threshold as 1.9 times the std deviation
        statistics_dict[key] = (mean_val, median_val, std_val, threshold)  # Store mean, median, std, and threshold
    return statistics_dict  # Return the dictionary with calculated statistics

def get_encompassing_bbox(bbox1, bbox2):
    """Calculate the smallest bounding box that encompasses two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x_min = min(x1, x2)  # Determine the minimum x-coordinate
    y_min = min(y1, y2)  # Determine the minimum y-coordinate
    x_max = max(x1 + w1, x2 + w2)  # Determine the maximum x-coordinate
    y_max = max(y1 + h1, y2 + h2)  # Determine the maximum y-coordinate
    return (x_min, y_min, x_max - x_min, y_max - y_min)  # Return the encompassing bounding box

def get_encompassing_bbox_all(bboxes):
    """Calculate the smallest bounding box that encompasses a list of bounding boxes."""
    if not bboxes:
        return None
    x_min = min([bbox[0][0] for bbox in bboxes])  # Determine the minimum x-coordinate
    y_min = min([bbox[0][1] for bbox in bboxes])  # Determine the minimum y-coordinate
    x_max = max([bbox[2][0] for bbox in bboxes])  # Determine the maximum x-coordinate
    y_max = max([bbox[2][1] for bbox in bboxes])  # Determine the maximum y-coordinate
    return (x_min, y_min, x_max - x_min, y_max - y_min)  # Return the encompassing bounding box

def draw_bounding_box(image, bbox, thickness=5, color=(0, 255, 0, 255)):
    """Draw a bounding box on an image with a specified thickness and color."""
    image = image.copy()  # Copy the image to avoid modifying the original
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGRA)  # Convert grayscale image to BGRA if needed
    x, y, w, h = bbox
    cv.rectangle(image, (x, y), (x+w, y+h), color, thickness=thickness)  # Draw the rectangle
    return image  # Return the image with the bounding box

def binarize_img(image, threshold=128):
    """Convert an image to binary using a threshold value and Otsu's method."""
    gray = image
    if len(image.shape) != 2:
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)  # Convert image to grayscale if not already
    _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Apply threshold
    return binary  # Return the binary image

def rsz(image, fx=0.4, fy=0.4, inter=cv.INTER_LINEAR):
    """Resize the image using the specified scale factors."""
    resized = cv.resize(image, None, fx=fx, fy=fy, interpolation=inter)  # Resize the image
    return resized  # Return the resized image

def rel_path(file_path):
    """Get the relative path of a file from the project directory."""
    project_dir = '/Users/brunofelalaga/projects/ocr/picaxe_paddleocr_github'
    return os.path.relpath(file_path, start=project_dir)  # Return the relative path

def whiten_outside_bbx(image, bbox):
    """Whiten the area outside the given bounding box in an image."""
    white_image = np.ones_like(image) * 255  # Create a white image of the same size
    x, y, w, h = bbox
    white_image[y:y+h, x:x+w] = image[y:y+h, x:x+w]  # Copy the original image inside the bounding box
    return white_image  # Return the image with whitened outside area

def whiten_pixels(image, x, y, w, h):
    """Whiten a specific rectangular region in an image."""
    if w <= 0 or h <= 0:
        return image  # Skip if the width or height is invalid
    if x < 0:
        w += x  # Adjust width if x is negative
        x = 0  # Set x to 0
    if y < 0:
        h += y  # Adjust height if y is negative
        y = 0  # Set y to 0
    white_rect = 255 * np.ones(shape=[h, w, 3], dtype=np.uint8)  # Create a white rectangle
    image[y:y+h, x:x+w] = white_rect  # Whiten the specified rectangle
    return image  # Return the image with the whitened rectangle

def white_out_text_areas(image, text_mask):
    """Whiten the areas of an image corresponding to the text mask."""
    _, binary_mask = cv.threshold(text_mask, 128, 255, cv.THRESH_BINARY)  # Convert mask to binary
    white_image = np.ones_like(image) * 255  # Create a white image of the same size

    # Ensure binary_mask is 3D to match the shape of the color image
    if len(image.shape) == 2:  # Grayscale image
        text_whitened_image = np.where(binary_mask == 255, white_image, image)  # Apply the mask to whiten text areas
    else:  # Color image
        binary_mask_3d = np.stack([binary_mask]*3, axis=-1)  # Expand binary_mask to 3D
        text_whitened_image = np.where(binary_mask_3d == 255, white_image, image)  # Apply the mask to whiten text areas

    return text_whitened_image  # Return the image with whitened text areas

def calculate_bounding_box(text_mask):
    """Calculate the bounding box that encompasses all contours in a text mask."""
    contours, _ = cv.findContours(text_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Find contours
    if not contours:
        return None  # Return None if no contours are found
    # Calculate the bounding box that encompasses all contours
    x_min = min([cv.boundingRect(cnt)[0] for cnt in contours])
    y_min = min([cv.boundingRect(cnt)[1] for cnt in contours])
    x_max = max([cv.boundingRect(cnt)[0] + cv.boundingRect(cnt)[2] for cnt in contours])
    y_max = max([cv.boundingRect(cnt)[1] + cv.boundingRect(cnt)[3] for cnt in contours])
    return (x_min, y_min, x_max - x_min, y_max - y_min)  # Return the bounding box

def ndarr2pil(image):
    """Convert a NumPy array (OpenCV image) to a PIL Image."""
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image)  # Convert grayscale image
    else:
        pil_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))  # Convert BGR to RGB and then to PIL
    dpi = pil_image.info.get('dpi', (500, 500))[0]  # Get DPI information or default to 500
    return pil_image, dpi  # Return the PIL image and its DPI

def pil2ndarr(image):
    """Convert a PIL Image to a NumPy array (OpenCV image)."""
    if isinstance(image, np.ndarray):
        return image  # If already a NumPy array, return it
    image_array = np.array(image)  # Convert PIL image to NumPy array
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB image
        image_array = cv.cvtColor(image_array, cv.COLOR_RGB2BGR)  # Convert RGB to BGR
    elif len(image_array.shape) == 4 and image_array.shape[2] == 4:  # RGBA image
        image_array = cv.cvtColor(image_array, cv.COLOR_RGBA2BGRA)  # Convert RGBA to BGRA
    return image_array  # Return the NumPy array

def inpaint_text(image, text_mask):
    """Inpaint text areas in an image using the text mask."""
    inpaint_mask = cv.bitwise_not(text_mask)  # Invert the text mask
    inpainted_image = cv.inpaint(image, inpaint_mask, 3, cv.INPAINT_TELEA)  # Inpaint the masked areas
    return inpainted_image  # Return the inpainted image

def remove_black_borders(image_path):
    """Remove black borders from an image by finding the largest contour and cropping it."""
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  # Read the image in grayscale
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Find contours
    if len(contours) == 0:
        print(f"No contours found in {image_path}")
        return None  # Return None if no contours are found
    largest_contour = max(contours, key=cv.contourArea)  # Get the largest contour
    mask = np.zeros_like(img)  # Create a mask for the largest contour
    cv.drawContours(mask, [largest_contour], 0, 255, thickness=cv.FILLED)  # Draw the contour on the mask
    result = cv.bitwise_and(img, mask)  # Apply the mask to the image
    x, y, w, h = cv.boundingRect(largest_contour)  # Get the bounding rectangle of the contour
    cropped_image = result[y:y + h, x:x + w]  # Crop the image to the bounding rectangle
    return cropped_image  # Return the cropped image

def is_file_already_processed(base_name, directory):
    """Check if a file has already been processed and saved in the specified directory."""
    expected_filename = f"{base_name}.png"  # Construct the expected file name
    return os.path.isfile(os.path.join(directory, expected_filename))  # Check if the file exists

def pre_process(i, dir, dilation_iterations, image):
    """Pre-process an image for contour detection by applying blurring and thresholding."""
    if image is None:
        image_path = os.path.join(dir, f'pg{i}.png')  # Construct the image path
        image = cv.imread(image_path)  # Read the image

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert the image to grayscale
    blur = cv.GaussianBlur(gray, (9, 9), 10)  # Apply Gaussian blur
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]  # Apply thresholding
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))  # Create a structuring element
    dilate = cv.dilate(thresh, kernel, iterations=dilation_iterations)  # Apply dilation
    return cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Find and return contours

def count_black_and_total_pixels(image):
    """Count the number of black pixels and total pixels in an image."""
    total_pixels = image.size  # Get the total number of pixels
    black_pixels = np.sum(image == 0)  # Count the number of black pixels
    return black_pixels, total_pixels  # Return the counts

def resize(image, fx=0.5, fy=0.5, inter=cv.INTER_AREA):
    """Resize an image using the specified scale factors."""
    if fx <= 0 or fy <= 0:
        raise ValueError("Scaling factors must be positive numbers.")  # Validate scale factors

    # Rescale image using fx and fy for scaling factors
    rescaled = cv.resize(image, (0, 0), fx=fx, fy=fy, interpolation=inter)
    return rescaled  # Return the resized image

def filter_cnts(proc_img_cnts, image, height=100, aspr=10):
    """Filter contours based on height and aspect ratio."""
    filtered_contours = []
    imh, imw = image.shape[:2]  # Get image dimensions
    for contour in proc_img_cnts:
        x, y, w, h = cv.boundingRect(contour)  # Get bounding rectangle of the contour
        aspr_actual = float(w/h)  # Calculate the aspect ratio
        if h < height and w < imw / 2:  # Filter contours by height and width
            filtered_contours.append(contour)  # Add the contour to the filtered list
    return filtered_contours  # Return the filtered contours

def get_pdf_file_paths(corpus='exp'):
    """Get the file paths of PDF files based on the specified corpus."""
    file_paths = []
    if corpus == 'exp':
        sample_papers_files = [os.path.join('sample_papers', f) for f in os.listdir('sample_papers')]  # List sample papers
        problematic_pdfs_files = [os.path.join('problematic_pdfs', f) for f in os.listdir('problematic_pdfs')]  # List problematic PDFs
        file_paths = sample_papers_files + problematic_pdfs_files  # Combine the file lists
    elif corpus == 'biofilm':
        file_paths = [os.path.join('biofilm_corpus_pdfs', f) for f in os.listdir('biofilm_corpus_pdfs')]  # List biofilm corpus PDFs
    elif corpus == 'anthro':
        file_paths = [os.path.join('anthropocene', f) for f in os.listdir('anthropocene')]  # List anthropocene PDFs
    file_paths.sort()  # Sort the file paths
    return file_paths  # Return the sorted file paths

def preprocess_ocr(image):
    """Pre-process an image for OCR by binarizing and applying morphological operations."""
    gray = image
    if len(image.shape) != 2:
        gray = binarize_img(image)  # Binarize image if not already binary

    gray = cv.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    gray = apply_opening(gray, kernel_size=(2, 2),  iterations=1)  # Apply opening
    gray = apply_closing(gray, kernel_size=(2, 2),  iterations=1)  # Apply closing
    return gray  # Return the pre-processed image

def compute_dynamic_margin(bounding_boxes):
    """Compute a dynamic margin based on the average height of bounding boxes."""
    if not bounding_boxes:
        return 0  # Return 0 if no bounding boxes are provided
    heights = [max([point[1] for point in box]) - min([point[1] for point in box]) for box in bounding_boxes]  # Calculate heights
    avg_height = np.mean(heights)  # Calculate average height
    dynamic_margin = int(avg_height * 0.1)  # Set dynamic margin as 10% of the average height
    return dynamic_margin  # Return the dynamic margin

def draw_bounding_boxes(image, bounding_boxes, margin=0, ptxt=False):
    """Draw bounding boxes on an image with optional margin and text."""
    if len(image.shape) == 2:  # Grayscale image
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # Convert to BGR
    elif image.shape[2] == 4:  # RGBA image
        image = cv.cvtColor(image, cv.COLOR_RGBA2BGR)  # Convert to BGR

    # Ensure bounding_boxes is a list of tuples
    if isinstance(bounding_boxes, tuple):
        bounding_boxes = [bounding_boxes]  # Convert to list if a single tuple
    elif isinstance(bounding_boxes, list):
        if isinstance(bounding_boxes[0], list):
            bounding_boxes = [tuple(box) for box in bounding_boxes]  # Convert inner lists to tuples

    for box in bounding_boxes:
        # Check if the box is in the expected format
        if isinstance(box[0], (list, tuple)) and len(box) == 4:
            x1, y1 = box[0]
            x2, y2 = box[2]
            w = x2 - x1
            h = y2 - y1
        elif len(box) == 4:
            x, y, w, h = box
            x1, y1 = x - margin, y - margin
            x2, y2 = x + w + margin, y + h + margin
        else:
            print(f"Unexpected box format: {box}")
            continue

        # Ensure the coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)

        cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0, 255), 2)  # Draw the bounding box
        if ptxt:
            text = f"{abs(y1-y2)}, {w/h:.2f}"
            cv.putText(image, text, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv.LINE_AA)  # Add text
    return image  # Return the image with bounding boxes

# Mask creation based on bounding boxes
def get_text_mask(mask, bounding_boxes, margin=0):
    """Create a mask from bounding boxes with optional margin."""
    for box in bounding_boxes:
        x_coords = [int(point[0]) for point in box]
        y_coords = [int(point[1]) for point in box]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)

        # Adjust margins for bounding boxes, ensuring they don't go out of image boundaries
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin-2)
        x2, y2 = min(mask.shape[1] - 1, x2 + margin), min(mask.shape[0] - 1, y2 + margin+2)

        # Fill the rectangle on the mask
        mask[y1:y2, x1:x2] = 255
    return mask

# OCR text removal function using PaddleOCR
logging.basicConfig(level=logging.WARNING)
logging.getLogger('paddleocr').setLevel(logging.WARNING)
logging.getLogger('ppocr').setLevel(logging.WARNING)
def remove_txt_paddle(image, ocrm=None, margin=None, max_retries=1, delay=0):
    """Remove text from an image using OCR with retries and optional margin."""
    if ocrm is None:
        print("OCR model not provided")
        return image, None

    retries = 0
    image = preprocess_ocr(image)  # Preprocess the image for OCR
    original_image = image.copy()
    last_result = None
    all_boxes = []

    while retries < max_retries:
        result = ocrm.ocr(image)
        if result[0] and result[0][0]:  # Check if there is at least one detected text box
            boxes = [res[0] for res in result[0]]

            if margin is None:
                margin = 0  # Set a default margin if not provided

            text_mask = np.zeros_like(image, dtype=np.uint8)
            text_mask = get_text_mask(text_mask, boxes, margin=margin)
            imgwhite = white_out_text_areas(image, text_mask)

            last_result = imgwhite
            image = imgwhite  # Update image to the one with text areas whitened

            # Collect all bounding boxes
            all_boxes.extend(boxes)
        else:
            break  # No text detected, exit the loop
        
        # Explicitly clear memory
        del boxes, text_mask, imgwhite
        gc.collect()
        
        retries += 1
        time.sleep(delay)  # Give a small delay before the next attempt

    if last_result is None:
        last_result = original_image

    return last_result, None

# Corner cleaning and margin line removal
def nip_corners_and_remove_margin_lines(image):
    """Remove corners and clean up margin lines from the image."""
    imgcp = image.copy()
    imgcp, _ = clean_corners(imgcp)  # Clean image corners
    imgcp_h, _ = rm_h_lines_margin(imgcp)  # Remove horizontal margin lines
    imgcp_vh, _ = rm_v_lines_margin(imgcp_h)  # Remove vertical margin lines
    return imgcp_vh

# Apply morphological operations to a binary image
def apply_morphology(binary_image, operation, kernel_size=(3, 3), iterations=1):
    """Apply a specified morphological operation to a binary image."""
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)  # Create structuring element
    result_image = cv.morphologyEx(binary_image, operation, kernel, iterations=iterations)  # Apply morphology
    return result_image

# Apply opening (erosion followed by dilation)
def apply_opening(binary_image, kernel_size=(3, 3), iterations=1):
    """Apply morphological opening to a binary image."""
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)  # Create structuring element
    opened_image = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel, iterations=iterations)  # Apply opening
    return opened_image

# Apply closing (dilation followed by erosion)
def apply_closing(binary_image, kernel_size=(3, 3), iterations=1):
    """Apply morphological closing to a binary image."""
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)  # Create structuring element
    closed_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel, iterations=iterations)  # Apply closing
    return closed_image

# Calculate statistics for a dictionary of values
def calculate_statistics(data_dict):
    """Calculate mean, median, standard deviation, and threshold for dictionary values."""
    statistics_dict = {}
    for key, values in data_dict.items():
        if len(values) < 2:
            mean_val = statistics.mean(values) if values else 0
            median_val = statistics.median(values) if values else 0
            statistics_dict[key] = (mean_val, median_val, 0, 0)
            continue
        mean_val, std_val = statistics.mean(values), statistics.stdev(values)
        median_val, threshold = statistics.median(values), (1.9 * std_val)
        statistics_dict[key] = (mean_val, median_val, std_val, threshold)
    return statistics_dict

# Rescale an image by a given ratio
def rescale(image, ratio):
    """Rescale an image by a given ratio."""
    new_width = int(image.shape[1] * ratio)
    new_height = int(image.shape[0] * ratio)
    dim = (new_width, new_height)
    rescaled = cv.resize(image, dim, interpolation=cv.INTER_AREA)  # Resize the image
    return rescaled

# Binarize an image using a threshold
def binarize_img(image, threshold=128):
    """Convert an image to binary using a threshold."""
    gray = image
    if len(image.shape) != 2:
      gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert to grayscale if not already
    _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Apply thresholding
    return binary

# Generate a rectangular kernel
def get_kernel(ks=3, ks2=3):
    """Create a rectangular kernel with specified dimensions."""
    return np.ones((ks, ks2), np.uint8)

# Filter contours to retain large ones
def filter_for_large_cnts(proc_img_cnts, img_shape, height=100, aspr=10):
    """Filter contours to keep only those larger than a specified height."""
    filtered_contours = []
    imh, imw = img_shape
    for contour in proc_img_cnts:
        x, y, w, h = cv.boundingRect(contour)  # Get bounding rectangle of the contour
        aspr_actual = float(w/h)  # Calculate the aspect ratio
        if h > height:  # Filter contours by height
            filtered_contours.append(contour)  # Add the contour to the filtered list
    return filtered_contours

# Calculate distance between bounding box edges
def calculate_edge_distance(bbox1, bbox2):
    """Calculate the distance between edges of two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate the distances between the edges
    left = x2 - (x1 + w1)  # Distance from the right edge of bbox1 to the left edge of bbox2
    right = x1 - (x2 + w2)  # Distance from the left edge of bbox1 to the right edge of bbox2
    top = y2 - (y1 + h1)  # Distance from the bottom edge of bbox1 to the top edge of bbox2
    bottom = y1 - (y2 + h2)  # Distance from the top edge of bbox1 to the bottom edge of bbox2

    # Choose the minimum non-negative distance
    horizontal_distance = max(left, right, 0)
    vertical_distance = max(top, bottom, 0)

    # Return the maximum distance between the two minimum distances
    return max(horizontal_distance, vertical_distance)

# Get the bounding box that encompasses two boxes
def get_encompassing_bbox(bbox1, bbox2):
    """Calculate the encompassing bounding box for two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)
    return (x_min, y_min, x_max - x_min, y_max - y_min)

# Calculate the center of a bounding box
def calculate_center(bbox):
    """Calculate the center point of a bounding box."""
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)

# Calculate the Euclidean distance between two points
def calculate_distance(center1, center2):
    """Calculate the Euclidean distance between two center points."""
    return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

# Merge bounding boxes that are close to each other based on center-to-center distance
def merge_close_bboxes_center_2_center(bboxes, threshold=100):
    """Merge bounding boxes that are close based on their center-to-center distance."""
    while True:
        merged = False
        i = 0
        while i < len(bboxes):
            j = i + 1
            while j < len(bboxes):
                center1 = calculate_center(bboxes[i])
                center2 = calculate_center(bboxes[j])
                distance = calculate_distance(center1, center2)

                if distance < threshold:
                    new_bbox = get_encompassing_bbox(bboxes[i], bboxes[j])
                    bboxes.pop(j)  # Remove jth bounding box
                    bboxes.pop(i)  # Remove ith bounding box (note: i < j, so removing j first is safe)
                    bboxes.append(new_bbox)  # Add the new merged bounding box
                    merged = True
                    break  # Break the inner loop to start over
                j += 1
            if merged:
                break  # Break the outer loop to start over
            i += 1
        if not merged:
            break  # Exit the while loop if no more merges occur
    return bboxes

# Check if two bounding boxes overlap
def overlap(bbox1, bbox2):
    """Determine if two bounding boxes overlap."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

# Merge overlapping bounding boxes into a single bounding box
def merge_overlapping_bboxes(bboxes):
    """Merge overlapping bounding boxes into a single bounding box."""
    merged_bboxes = []
    while bboxes:
        bbox = bboxes.pop(0)
        i = 0
        while i < len(bboxes):
            if overlap(bbox, bboxes[i]):
                bbox = get_encompassing_bbox(bbox, bboxes.pop(i))  # Merge the overlapping bounding boxes
            else:
                i += 1
        merged_bboxes.append(bbox)  # Add the merged bounding box to the result list
    return merged_bboxes

# Calculate the Intersection over Union (IoU) of two bounding boxes
def calculate_iou(bbox1, bbox2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Determine the coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height  # Calculate the area of the intersection

    # Calculate the union area
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area  # Calculate the area of the union

    # Compute the IoU
    iou = inter_area / union_area if union_area != 0 else 0  # Handle division by zero
    return iou

def merge_bboxes_iou(bboxes_input, iou_threshold=0.5):
    """Merge bounding boxes based on Intersection over Union (IoU) threshold."""
    bboxes = list(bboxes_input)  # Create a shallow copy of the input list
    merged_bboxes = []
    while bboxes:
        bbox = bboxes.pop(0)
        i = 0
        while i < len(bboxes):
            if calculate_iou(bbox, bboxes[i]) >= iou_threshold:
                bbox = get_encompassing_bbox(bbox, bboxes.pop(i))  # Merge if IoU exceeds the threshold
            else:
                i += 1
        merged_bboxes.append(bbox)
    return merged_bboxes

def merge_close_bboxes(bboxes, threshold=100):
    """Merge bounding boxes that are close to each other based on edge distance."""
    while True:
        merged = False
        i = 0
        while i < len(bboxes):
            j = i + 1
            while j < len(bboxes):
                distance = calculate_edge_distance(bboxes[i], bboxes[j])
                if distance < threshold:
                    new_bbox = get_encompassing_bbox(bboxes[i], bboxes[j])
                    bboxes.pop(j)  # Remove jth bounding box
                    bboxes.pop(i)  # Remove ith bounding box (note: i < j, so removing j first is safe)
                    bboxes.append(new_bbox)  # Add the new merged bounding box
                    merged = True
                    break  # Break the inner loop to start over
                j += 1
            if merged:
                break  # Break the outer loop to start over
            i += 1
        if not merged:
            break  # Exit the while loop if no more merges occur
    return bboxes

def is_file_already_processed(base_name, directory):
    """Check if a file has already been processed and exists in the given directory."""
    expected_filename = f"{base_name}.png"
    return os.path.isfile(os.path.join(directory, expected_filename))

def draw_contours_(image, contours):
    """Draw contours on an image."""
    if len(image.shape) == 2:
        contour_image = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGRA)  # Convert to BGRA if needed
    else:
        contour_image = cv.cvtColor(image.copy(), cv.COLOR_RGBA2BGRA)
    cv.drawContours(contour_image, contours, -1, (0, 255, 0, 255), 2)
    return contour_image

def extract_and_place_regions(image, bboxes):
    """Extract regions defined by bounding boxes and place them on a white background."""
    white_image = np.ones_like(image) * 255
    for bbox in bboxes:
        x, y, w, h = bbox
        region = image[y:y+h, x:x+w]
        white_image[y:y+h, x:x+w] = region
    return white_image

def white_out_text_areas__(image, text_mask):
    """Whiten out areas defined by the text mask."""
    print(f"here {image.shape}")
    _, binary_mask = cv.threshold(text_mask, 128, 255, cv.THRESH_BINARY)
    white_image = np.ones_like(image) * 255
    if len(image.shape) == 2:  # Grayscale image
        text_whitened_image = np.where(binary_mask == 255, white_image, image)
    else:  # Color image
        text_whitened_image = np.where(binary_mask[..., np.newaxis] == 255, white_image, image)
    return text_whitened_image

def filter_text_boxes(image, bounding_boxes, stats):
    """Filter text boxes based on statistical thresholds."""
    mean_density, median_density, std_density, threshD = stats['text_densities']
    mn_arat, md_arat, std_arat, threshARATS = stats['arats']
    mn_area, md_area, std_area, threshAREA = stats['box_areas']
    filtered_boxes = []

    # Convert image to grayscale if it is colored
    gray = image
    if len(image.shape) > 2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    for (x, y, w, h) in bounding_boxes:
        if w == 0 or h == 0:
            continue
        text_box = gray[y:y+h, x:x+w]
        num_text_pixels = cv.countNonZero(text_box)
        bbox_area = w * h
        arat = h / w
        text_density = num_text_pixels / bbox_area

        dev_density = abs(text_density - mean_density)
        dev_arat = abs(arat - mn_arat)
        dev_area = abs(bbox_area - mn_area)

        if (threshARATS and dev_arat > threshARATS) or (threshAREA and dev_area > threshAREA):
            print(f"here1{text_density}")
            continue
        if dev_density > threshD or text_density < 0.1:
            print("here2")
            continue
        print("here")
        filtered_boxes.append((x, y, w, h))
    return filtered_boxes

# Define Sobel kernels
sobel_kernels = {
    'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64),
    'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64),
    'sobel_45': np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float64),
    'sobel_135': np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float64),
    'sobel_22.5': np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype=np.float64),
    'sobel_67.5': np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], dtype=np.float64),
    'sobel_112.5': np.array([[-1, -1, 0], [1, 0, -1], [0, 1, 1]], dtype=np.float64),
    'sobel_157.5': np.array([[0, -1, -1], [-1, 0, 1], [1, 1, 0]], dtype=np.float64)
}

def calculate_edge_distance(bbox1, bbox2):
    """Calculate the minimum distance between the edges of two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate the distances between the edges
    left = x2 - (x1 + w1)  # Distance from the right edge of bbox1 to the left edge of bbox2
    right = x1 - (x2 + w2)  # Distance from the left edge of bbox1 to the right edge of bbox2
    top = y2 - (y1 + h1)  # Distance from the bottom edge of bbox1 to the top edge of bbox2
    bottom = y1 - (y2 + h2)  # Distance from the top edge of bbox1 to the bottom edge of bbox2

    # Choose the minimum non-negative distance
    horizontal_distance = max(left, right, 0)
    vertical_distance = max(top, bottom, 0)

    # Return the maximum distance between the two minimum distances
    return max(horizontal_distance, vertical_distance)

def merge_close_bboxes(bboxes, threshold=100):
    """Merge bounding boxes that are close to each other based on edge distance."""
    while True:
        merged = False
        i = 0
        while i < len(bboxes):
            j = i + 1
            while j < len(bboxes):
                distance = calculate_edge_distance(bboxes[i], bboxes[j])
                if distance < threshold:
                    new_bbox = get_encompassing_bbox(bboxes[i], bboxes[j])
                    bboxes.pop(j)  # Remove jth bounding box
                    bboxes.pop(i)  # Remove ith bounding box (note: i < j, so removing j first is safe)
                    bboxes.append(new_bbox)  # Add the new merged bounding box
                    merged = True
                    break  # Break the inner loop to start over
                j += 1
            if merged:
                break  # Break the outer loop to start over
            i += 1
        if not merged:
            break  # Exit the while loop if no more merges occur
    return bboxes

def get_encompassing_bbox(bbox1, bbox2):
    """Return a bounding box that encompasses both bbox1 and bbox2."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def binarize_img(image, threshold=128):
    """Convert an image to a binary image using a threshold."""
    gray = image
    if len(image.shape) != 2:
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)  # Convert to grayscale if necessary
    _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return binary

def get_kernel(ks=3, ks2=3):
    """Return a rectangular kernel of size (ks, ks2)."""
    return np.ones((ks, ks2), np.uint8)

def gaussian(img):
    """Apply Gaussian blur to the image."""
    return cv.GaussianBlur(img, (5, 5), 0)

def add_padding_to_bbox(bbox, padding, image_shape):
    """Add padding to a bounding box while ensuring it stays within image bounds."""
    x, y, w, h = bbox
    x_new = max(0, x - padding)
    y_new = max(0, y - padding)
    w_new = min(image_shape[1] - x_new, w + 2 * padding)  # Ensure width doesn't go out of image bounds
    h_new = min(image_shape[0] - y_new, h + 2 * padding)  # Ensure height doesn't go out of image bounds
    return (x_new, y_new, w_new, h_new)

def apply_padding_to_bboxes(bboxes, padding, image_shape):
    """Apply padding to a list of bounding boxes."""
    return [add_padding_to_bbox(bbox, padding, image_shape) for bbox in bboxes]

def remove_horizontal_lines(image, line_width_threshold=50, aspect_ratio_threshold=5):
    """Remove horizontal lines from an image based on width and aspect ratio thresholds."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if len(image.shape) == 3 else image  # Convert to grayscale if necessary
    _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)  # Binarize the image

    # Morphological operations to highlight horizontal lines
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 5))
    dilated = cv.dilate(binary, horizontal_kernel, iterations=1)
    eroded = cv.erode(dilated, horizontal_kernel, iterations=1)

    # Find contours
    contours, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw the filtered contours
    mask = np.ones_like(binary) * 255

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        if w > line_width_threshold and aspect_ratio > aspect_ratio_threshold:
            cv.drawContours(mask, [contour], -1, 0, -1)  # Draw the contour on the mask

    cleaned = cv.bitwise_and(binary, binary, mask=mask)  # Apply the mask to the binary image
    cleaned = cv.bitwise_not(cleaned)  # Invert the cleaned image

    # Convert back to BGR if the original image was in color
    if len(image.shape) == 3:
        cleaned = cv.cvtColor(cleaned, cv.COLOR_GRAY2BGR)
    return cleaned

def remove_vertical_lines(image, line_height_threshold=50, aspect_ratio_threshold=2):
    """Remove vertical lines from an image based on height and aspect ratio thresholds."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if len(image.shape) == 3 else image  # Convert to grayscale if necessary
    _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)  # Binarize the image

    # Morphological operations to highlight vertical lines
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 25))
    dilated = cv.dilate(binary, vertical_kernel, iterations=1)
    eroded = cv.erode(dilated, vertical_kernel, iterations=1)
    contours, _ = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw the filtered contours
    mask = np.ones_like(binary) * 255

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = float(h) / w if w > 0 else 0
        if h > line_height_threshold and aspect_ratio > aspect_ratio_threshold:
            cv.drawContours(mask, [contour], -1, 0, -1)  # Draw the contour on the mask

    cleaned = cv.bitwise_and(binary, binary, mask=mask)  # Apply the mask to the binary image
    cleaned = cv.bitwise_not(cleaned)  # Invert the cleaned image

    # Convert back to BGR if the original image was in color
    if len(image.shape) == 3:
        cleaned = cv.cvtColor(cleaned, cv.COLOR_GRAY2BGR)
    return cleaned

def filter_contours_by_area(contours, image_shape, area_threshold):
    """Filter contours based on their area relative to the image size."""
    img_height, img_width = image_shape[:2]
    filtered_contours = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w * h >= area_threshold and not (w >= img_width - 1 and h >= img_height - 1):
            filtered_contours.append((x, y, w, h))
    return filtered_contours
    
def filter_bbxs_by_area(bbxs, area_threshold):
    """Filter bounding boxes by a minimum area threshold."""
    filtered_bboxes = []
    for bbx in bbxs:
        x, y, w, h = bbx
        area = w * h
        if area >= area_threshold:
            filtered_bboxes.append(bbx)
    return filtered_bboxes

def clean_corners(bw):
    """Clean corners of the image based on margins and corner sizes."""
    height, width = bw.shape
    mask = np.ones_like(bw) * 255

    # Calculate margins from the edges
    top_margin = get_closest_object_distance(bw, 'top')
    bottom_margin = get_closest_object_distance(bw, 'bottom')
    left_margin = get_closest_object_distance(bw, 'left')
    right_margin = get_closest_object_distance(bw, 'right')

    # Define the corner size based on image size and margins
    corner_height = max(int(0.12 * height), top_margin, bottom_margin)
    corner_width = max(int(0.15 * width), left_margin, right_margin)

    # Mask the corners to clean them
    mask[:corner_height, :corner_width] = 0  # Top-left corner
    mask[:corner_height, width - corner_width:] = 0  # Top-right corner
    mask[height - corner_height:, :corner_width] = 0  # Bottom-left corner
    mask[height - corner_height:, width - corner_width:] = 0  # Bottom-right corner

    # Apply the mask to clean the corners
    cleaned = cv.bitwise_or(bw, cv.bitwise_not(mask))
    return cleaned, mask

def clean_corners_in_bounding_box(bw):
    """Clean corners within the largest bounding box in the image."""
    # Find contours and bounding boxes
    contours, _ = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    bboxes = [cv.boundingRect(c) for c in contours]

    # Exclude the bounding box that covers the full image
    bboxes = exclude_full_image_bbox(bboxes, bw.shape)

    if not bboxes:
        return bw, None
        raise ValueError("No valid bounding boxes found after excluding full image bbox.")

    # Find the largest bounding box by area
    largest_bbox = max(bboxes, key=lambda bbox: bbox[2] * bbox[3])
    x, y, w, h = largest_bbox

    # Create a mask for the bounding box area
    mask = np.ones_like(bw) * 255

    # Calculate margins within the bounding box
    top_margin = get_closest_object_distance(bw[y:y+h, x:x+w], 'top')
    bottom_margin = get_closest_object_distance(bw[y:y+h, x:x+w], 'bottom')
    left_margin = get_closest_object_distance(bw[y:y+h, x:x+w], 'left')
    right_margin = get_closest_object_distance(bw[y:y+h, x:x+w], 'right')

    # Define the corner size within the bounding box
    corner_height = max(int(0.20 * h), top_margin, bottom_margin)
    corner_width = max(int(0.20 * w), left_margin, right_margin)

    # Mask the corners within the bounding box
    mask[y:y+corner_height, x:x+corner_width] = 0  # Top-left corner
    mask[y:y+corner_height, x+w-corner_width:x+w] = 0  # Top-right corner
    mask[y+h-corner_height:y+h, x:x+corner_width] = 0  # Bottom-left corner
    mask[y+h-corner_height:y+h, x+w-corner_width:x+w] = 0  # Bottom-right corner

    # Apply the mask to clean the corners within the bounding box
    cleaned = bw.copy()
    cleaned[mask == 0] = 255

    return cleaned, mask

def convert_bbxs_2points(bounding_boxes):
    """Convert bounding boxes from (x, y, w, h) to [(x1, y1), (x2, y2)] format."""
    converted_bboxes = []
    for box in bounding_boxes:
        x1, y1, w, h = box
        converted_bboxes.append([(x1, y1), (x1+w, y1+h)])
    return converted_bboxes

def load_and_convert_image(image_path):
    """Load an image and convert it to RGB format if necessary."""
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')  # Convert RGBA to RGB
    return np.array(image)

def draw_contours(image, contours):
    """Draw contours on the image."""
    if len(image.shape) == 2:
        contour_image = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)  # Convert grayscale to BGR if needed
    cv.drawContours(contour_image, contours, -1, (0, 255, 0, 255), 2)  # Draw contours in green
    return contour_image

def save_page_image_with_bounding_boxes(cleaned_image, filtered_bboxes, page_output_file_path):
    """Draw bounding boxes on the cleaned image and save the result."""
    # Draw bounding boxes on a copy of the cleaned image
    page_image = draw_bounding_boxes(cleaned_image.copy(), convert_bbxs_2points(filtered_bboxes), ptxt=True)

    # Save the processed image with bounding boxes
    cv.imwrite(page_output_file_path, page_image)
    print(f'{page_output_file_path} saved successfully')
    return page_image

def draw_bounding_boxes(image, bounding_boxes, margin=0, ptxt=False):
    """Draw bounding boxes on the image, with optional text annotations."""
    if len(image.shape) == 2:  # Convert grayscale to BGR
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # Convert RGBA to BGR
        image = cv.cvtColor(image, cv.COLOR_RGBA2BGR)

    if not bounding_boxes:  # Handle empty bounding boxes list
        print("Bounding boxes list is empty")
        return image

    # Convert bounding boxes to the appropriate format
    if isinstance(bounding_boxes, tuple):
        bounding_boxes = [bounding_boxes]
    elif isinstance(bounding_boxes, list):
        if isinstance(bounding_boxes[0], list):
            bounding_boxes = [tuple(box) for box in bounding_boxes]

    for box in bounding_boxes:
        # Handle different formats of bounding boxes
        if isinstance(box[0], (list, tuple)) and len(box) == 4:
            x1, y1 = box[0]
            x2, y2 = box[2]
            w = x2 - x1
            h = y2 - y1
        elif isinstance(box[0], tuple) and len(box) == 2:
            x1, y1 = box[0]
            x2, y2 = box[1]
            w = x2 - x1
            h = y2 - y1
        elif len(box) == 4:
            x, y, w, h = box
            x1, y1 = x - margin, y - margin
            x2, y2 = x + w + margin, y + h + margin
        else:
            print(f"Unexpected box format: {box}")
            continue

        # Ensure bounding box coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)

        # Draw the bounding box on the image
        cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0, 255), 2)
        if ptxt:
            text = f"{abs(y1-y2)}, {w/h:.2f}"
            cv.putText(image, text, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv.LINE_AA)
    return image

def is_file_already_processed(base_name, directory):
    """Check if a file with a given base name has already been processed and saved in the directory."""
    expected_filename = f"{base_name}.png"
    return os.path.isfile(os.path.join(directory, expected_filename))

def filter_for_large_cnts(proc_img_cnts, img_shape, height=100, aspr=10):
    """Filter contours based on height and aspect ratio."""
    filtered_contours = []
    imh, imw = img_shape
    for contour in proc_img_cnts:
        x, y, w, h = cv.boundingRect(contour)
        aspr_actual = float(w/h)
        if h > height:  # Filter based on height and optionally aspect ratio
            filtered_contours.append(contour)
    return filtered_contours

def clear_memory(var_list):
    """Clear memory by deleting variables and forcing garbage collection."""
    for var in var_list:
        del var
    gc.collect()  # Collect garbage to free memory
    time.sleep(1)  # Pause briefly to ensure memory is freed
    return
