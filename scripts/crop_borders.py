import os
import sys
import cv2 as cv
import numpy as np

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')
sys.path.append(src_path)

# Import utilities and configuration
from utilities import rel_path, cvsm, is_image_file
from config import bounding_boxes_dir, pdf_imgs_dir as input_dir, cropped_dir as output_dir

def remove_borders(image, white_threshold=170):
    """Remove borders from the image based on a white threshold."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
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

    # Crop the image and return it along with its bounding box
    cropped_image = image[top_border:bottom_border, left_border:right_border]
    return cropped_image, (top_border, left_border, bottom_border, right_border)

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

def place_on_white_background(image, cropped_image, bbox):
    """Place the cropped image on a white background using the bounding box coordinates."""
    top_border, left_border, bottom_border, right_border = bbox
    h, w, _ = image.shape

    # Create a white image of the same size as the original
    white_image = np.full((h, w, 3), 255, dtype=np.uint8)

    # Place the cropped image on the white background
    white_image[top_border:bottom_border, left_border:right_border] = cropped_image
    return white_image

def main():
        """Main function to crop borders from images and save them with white backgrounds."""
        # Get a list of image files
        pdf_img_files_all = sorted([f"{input_dir}/{filename}" for filename in os.listdir(input_dir)])

        # Get only files not yet processed
        pdf_img_files = filter_image_files(pdf_img_files_all, output_dir)
        print(f"\nCropping {len(pdf_img_files)} images")

        # Process each image: crop borders and place on white background and save to output
        for filename in pdf_img_files[:]:
            file_path = f"{input_dir}/{os.path.basename(filename)}"
            img = cv.imread(file_path)
            
            # Exclude files in dir that are not images: .DS_Store
            if img is None or img.size == 0:
                continue
            img_cropped, bounding_box = remove_borders(img)
            cropped_img_on_white_bg = place_on_white_background(img, img_cropped, bounding_box)
            cv.imwrite(f"{output_dir}/{os.path.basename(filename)}", cropped_img_on_white_bg)

        print(f"\nCropping {len(pdf_img_files)} images completed")
        return 


if __name__ == "__main__":
    main()
