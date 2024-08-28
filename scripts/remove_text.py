# remove_text.py
# This script is designed for high RAM usage and primarily focuses on text removal from images using PaddleOCR.

import os
import gc
import sys
import json
import time
import shutil
import random
import logging
import cv2 as cv
from paddleocr import PaddleOCR

# Add the src directory to the Python path for importing utilities and configurations
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')
sys.path.append(src_path)

# Import utilities and configuration settings
from utilities import remove_txt_paddle, nip_corners_and_remove_margin_lines, clean_image_filter2D, binarize_img, rel_path, is_image_file
from config import pages_no_tables_dir, cropped_dir, processed_files_log, bounding_boxes_dir as input_dir, text_removed_dir as output_dir

# Load the log of processed files to avoid reprocessing the same files
def load_processed_files():
    """Load the log of files that have already been processed."""
    if os.path.exists(processed_files_log):
        with open(processed_files_log, 'r') as f:
            return set(json.load(f))
    else:
        return set()

# Save the log of processed files after each batch to track progress
def save_processed_files(processed_files):
    """Save the log of files that have been processed."""
    with open(processed_files_log, 'w') as f:
        json.dump(list(processed_files), f)

# Function to remove tables from the image using bounding boxes and return the cleaned image path
def get_no_tb_img(img_path):
    """
    Remove tables from the image based on pre-detected bounding boxes stored in a JSON file.
    Returns the path of the cleaned image.
    """
    base_name = f'{os.path.basename(img_path).split(".")[0]}.json'
    file_path = os.path.join(input_dir, base_name)

    with open(file_path, 'r') as file:
        try:
            content = json.load(file)  # Load bounding box data from JSON
            bbxs = content.get("bounding_boxes")
            pdf_img_file = f'{cropped_dir}/{content.get("file_name")}.png'

            img = cv.imread(pdf_img_file)

            if img is None:
                print(f"\n./{rel_path(pdf_img_file)} is Nonetype!\n")
                return None

            imgc = img.copy()  # Create a copy of the image to modify

            if not bbxs:  # If no bounding boxes, save the image as is
                save_path = f'{pages_no_tables_dir}/{content.get("file_name")}.png'
                cv.imwrite(save_path, imgc)
                del img, content, bbxs
                gc.collect()
                return save_path

            # Iterate over bounding boxes and whiten the table areas
            for bbox in bbxs:
                x1, y1, x2, y2 = bbox.get('x1'), bbox.get('y1'), bbox.get('x2'), bbox.get('y2')
                if x1 is None or y1 is None or x2 is None or y2 is None:
                    print(f"Invalid bounding box coordinates in {rel_path(file_path)}")
                    continue
                imgc[y1:y2, x1:x2] = 255  # Whiten the area inside the bounding box

            # Save the cleaned image
            save_path = f'{pages_no_tables_dir}/{content.get("file_name")}.png'
            cv.imwrite(save_path, imgc)

            del imgc, img, content, bbxs
            gc.collect()
            return save_path

        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {rel_path(file_path)}")

    print(f"Error: {rel_path(img_path)}")
    return None

# Generator function to yield batches of file paths for processing
def batch_generator(file_paths, batch_size):
    """
    Yield batches of file paths for processing.
    This helps in managing memory usage by processing images in smaller chunks.
    """
    for i in range(0, len(file_paths), batch_size):
        yield file_paths[i:i + batch_size]

# Function to retrieve the list of image JSON files for processing
def get_img_json_files():
    """
    Retrieve the list of image JSON files that contain bounding box information.
    These files are used to identify areas of images that need to be processed.
    """
    input_files = []
    for json_file in os.listdir(input_dir):
        if '.png' in json_file:
            continue
        input_files.append(f"{input_dir}/{json_file}")
    gc.collect()
    return input_files

# First part of preprocessing: Border removal and margin cleaning
def preprocess_borders_part1(img_path):
    """
    Preprocess the image by removing borders and cleaning margins.
    This is the first step in preparing the image for text removal.
    """
    no_tb_img_path = get_no_tb_img(img_path)
    img = cv.imread(no_tb_img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        return None

    img_bw = binarize_img(img)  # Convert image to binary
    mask = clean_image_filter2D(img_bw)  # Apply cleaning filter
    img_bw_flt = cv.bitwise_and(img_bw, img_bw, mask=mask)
    img_bw_flt[mask == 0] = 255
    img_bw_flt_nipped = nip_corners_and_remove_margin_lines(img_bw_flt)  # Remove corners and margins

    # Save the intermediate processed image
    intermediate_path = f'{output_dir}/intermediate/{os.path.basename(img_path).split(".")[0]}_nipped.png'
    os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
    cv.imwrite(intermediate_path, img_bw_flt_nipped)

    del img, img_bw, mask, img_bw_flt, img_bw_flt_nipped
    gc.collect()
    return intermediate_path

# Second part of preprocessing: Text removal and final processing
def preprocess_borders_part2(intermediate_path, img_path):
    """
    Finalize preprocessing by removing text from the image.
    This is the second step after initial border and margin cleaning.
    """
    if not intermediate_path or not os.path.isfile(intermediate_path):
        return None
    img_bw_flt_nipped = cv.imread(intermediate_path, cv.IMREAD_GRAYSCALE)
    if img_bw_flt_nipped is None:
        print(f"Failed to read intermediate file ./{rel_path(intermediate_path)}")
        return None

    rand_sleep = random.randint(2, 20)
    time.sleep(rand_sleep)
    no_txt_img, _ = remove_txt_paddle(img_bw_flt_nipped.copy(), ocrm=ocrm, margin=1)

    try:
        blurred_image = cv.bilateralFilter(no_txt_img, 11, 75, 75)  # Apply bilateral filter to smoothen the image
        no_txt_img = cv.addWeighted(blurred_image, 1 + 1.5, blurred_image, -1.5, 0)  # Sharpen the image
        del blurred_image
        gc.collect()

    except Exception as e:
        print(f"Error processing {rel_path(img_path)}: {e}")
        return None

    save_path = f'{output_dir}/{os.path.basename(img_path).split(".")[0]}.png'
    cv.imwrite(save_path, no_txt_img)  # Save the final processed image

    del no_txt_img, img_bw_flt_nipped
    gc.collect()
    rand_sleep = random.randint(1, 2)
    time.sleep(rand_sleep)
    return save_path

# Main function to run the batch processing mode
def run_bulk_mode(dona=[]):
    """
    Run the batch mode for processing multiple images.
    The function processes images in batches to optimize memory usage and speed.
    """
    image_paths_all = sorted(list(get_img_json_files()))

    # Filter out already processed files to avoid reprocessing
    image_paths = []
    processed_files = load_processed_files()
    for image_path in image_paths_all[:1]:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if base_name in processed_files or is_image_file(f"{output_dir}/{base_name}.png"):
            continue

        image_paths.append(image_path)

    print(f"\nRemoving text from {len(image_paths)} images")

    batch_size = 4  # Adjust based on available memory
    total_batches = len(image_paths) // batch_size + (1 if len(image_paths) % batch_size != 0 else 0)

    for batch_index, batch_paths in enumerate(batch_generator(image_paths, batch_size), start=1):

        # First part: Process and save intermediate files
        intermediate_paths = []
        for batch_path in batch_paths:
            intermediate_path = preprocess_borders_part1(batch_path)
            intermediate_paths.append(intermediate_path)
        gc.collect()
        time.sleep(0.01)

        # Second part: Process using intermediate files and clean up
        for intermediate_path, batch_path in zip(intermediate_paths, batch_paths):
            preprocess_borders_part2(intermediate_path, batch_path)

        # Remove the intermediate directory if it exists
        intermediate_dir = f"{output_dir}/intermediate"
        if os.path.exists(intermediate_dir):
            shutil.rmtree(intermediate_dir)  # Remove the directory and its contents
                
        # Update the processed files log
        processed_files.update(os.path.basename(path) for path in batch_paths)
        save_processed_files(processed_files)

        # Clear memory after each batch
        gc.collect()
        sleep_time = 0.1
        time.sleep(sleep_time)  # Pause between batches

    print(f"\nRemoved text from {len(image_paths)} images successfully")
    return dona

# Main execution block
if __name__ == '__main__':
    gc.collect()
    start = time.time()

    # Initialize the OCR model
    global ocrm  # Declare ocrm as global to be used across different processes
    logging.getLogger('ppocr').setLevel(logging.WARNING)
    ocrm = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize PaddleOCR model with English language support
    
    # Run the bulk mode function to process images
    dona = run_bulk_mode()
    end = time.time()
    # Print the runtime of the process (currently commented out)
    # print(f"\nRuntime: {(end - start) / 60:.2f} minutes")

