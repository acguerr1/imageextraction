import os
import sys
import json
import time
import cv2 as cv
from pathlib import Path

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')
sys.path.append(src_path)

# Import necessary functions and variables
from init_layoutparser_model import initialize_model
from utilities import rel_path, is_image_file
from config import project_dir, output_tables_dir, bounding_boxes_dir as output_dir, cropped_dir as input_dir

def save_bounding_boxes(image_name, bounding_boxes):
    """Save detected bounding boxes to a JSON file."""
    bbox_file_path = f'{output_dir}/{os.path.splitext(image_name)[0]}.json'
    os.makedirs(os.path.dirname(bbox_file_path), exist_ok=True)
    with open(bbox_file_path, 'w') as f:
        json.dump({"bounding_boxes": bounding_boxes, "file_name": os.path.splitext(image_name)[0]}, f, indent=4)
    return None

def get_image_table(pdf_img_path):
    """Detect and extract tables from an image."""
    global model
    if 'model' not in globals():
        initialize_model()

    # Check if the image file exists
    if not Path(pdf_img_path).exists():
        os.path.relpath(pdf_img_path, start=project_dir)
        print(f"Input file doesn't exist: {rel_path(pdf_img_path)}")
        return None

    # Load the image
    pdf_img = cv.imread(pdf_img_path)
    if pdf_img is None:
        print(f"Failed to read image at this path: {rel_path(pdf_img_path)}")
        return None

    pdf_img_copy = pdf_img.copy()
    height, width = pdf_img.shape[:2]

    # Detect layout and extract bounding boxes for tables
    layout = model.detect(pdf_img)
    bounding_boxes = []
    table_number = -1

    for l in layout:
        if l.type == 'Table':
            table_number += 1
            x1, x2 = int(l.block.x_1), int(l.block.x_2)
            y1, y2 = int(l.block.y_1), int(l.block.y_2)

            # Apply padding to the bounding box
            x_pad, y_pad = 10, 10
            x1_padded, x2_padded = max(0, x1 - x_pad), min(width, x2 + x_pad)
            y1_padded, y2_padded = max(0, y1 - y_pad), min(height, y2 + y_pad)

            # Extract the table image
            img_tb = pdf_img[y1_padded:y2_padded, x1_padded:x2_padded]
            pdf_img_copy[y1_padded:y2_padded, x1_padded:x2_padded] = 255

            # Save the extracted table image
            output_path = f"{output_tables_dir}/{os.path.basename(pdf_img_path).split('.')[0]}_table{table_number}.png"
            cv.imwrite(output_path, img_tb)

            # Store the bounding box information
            bounding_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    # Save bounding boxes as JSON
    save_bounding_boxes(os.path.basename(pdf_img_path), bounding_boxes)
    time.sleep(0.1)
    return None

def filter_image_files(pdf_img_files_all, output_dir, ext='.png', keywords=''):
    """Filter image files based on keywords or skip if already processed."""
    pdf_img_files = []
    for image_path in pdf_img_files_all:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_file_path = f'{output_dir}/{os.path.splitext(base_name)[0]}{ext}'

        # Skip files that have already been processed
        if os.path.exists(output_file_path) or not is_image_file(image_path):
            continue

        # Filter by keywords if provided, otherwise add all files
        if not keywords or any(keyword.lower() in base_name.lower() for keyword in keywords):
            pdf_img_files.append(image_path)
        
    return pdf_img_files

def main():
        """Main execution block to remove tables from images."""
        # Get a sorted list of all image files in the directory
        pdf_img_files_all = sorted([f"{input_dir}/{filename}" for filename in os.listdir(input_dir)])

        # Get only files not yet processed
        pdf_img_files = filter_image_files(pdf_img_files_all, output_dir, ext='.json')
        print(f"\nRemoving tables from {len(pdf_img_files)} images")

        # Exit if no files need processing
        if len(pdf_img_files) == 0:
            print("\nAll files already processed")
            
            return None
        
        # Process each file to detect and remove tables
        for filename in pdf_img_files:
            get_image_table(filename)
        
        print(f"\nTable removal from {len(pdf_img_files)} completed")
        return None

if __name__ == '__main__':

    # Initialize the model
    model = initialize_model()

    main()

