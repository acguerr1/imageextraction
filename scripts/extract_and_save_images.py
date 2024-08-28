import os
import sys
import time
import json
import logging
import cv2 as cv
import numpy as np
from paddleocr import PaddleOCR

# Ensure that the custom utilities module can be imported
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')
sys.path.append(src_path)

# Import utilities and configuration
from utilities import apply_padding_to_bboxes, filter_bbxs_by_area, merge_close_bboxes, \
                merge_bboxes_iou, exclude_full_image_bbox, remove_horizontal_lines, remove_vertical_lines, binarize_img, \
                remove_txt_paddle, rel_path, filter_for_large_cnts, get_kernel, is_image_file
from config import pdf_imgs_dir, log_dir, target_images as input_dir, extracted_images as output_dir, valid_image_extensions

# Initialize processed files set and log file path
processed_files = set()  # Using a set for faster lookup
log_file_path = os.path.join(log_dir, "processing_log.json")

def load_processed_files():
    """Load the processed files log."""
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            return set(json.load(log_file))
    return set()

def update_log(processed_files):
    """Update the processed files log."""
    with open(log_file_path, "w") as log_file:
        json.dump(list(processed_files), log_file)

def save_extracted_images(filtered_bboxes, image_path, output_dir, pdf_imgs_dir):
    """Save extracted images from bounding boxes."""
    for idx, (x, y, w, h) in enumerate(filtered_bboxes):
        orig_img_path = os.path.join(pdf_imgs_dir, os.path.basename(image_path))

        if not os.path.exists(orig_img_path):
            print(f"Original image not found: {orig_img_path}")
            continue

        orig_img = cv.imread(orig_img_path)
        if orig_img is None:
            print(f"Failed to read original image: {orig_img_path}")
            continue

        ximg_orig = orig_img[y:y+h, x:x+w]
        part1 = "_".join(os.path.basename(image_path).split('.')[0].split('_')[:-1])
        basename = f"{part1}_{os.path.basename(image_path).split('.')[0].split('_')[-1]}_image{idx}.png"
        pcx_img_save_path = f"{output_dir}/{basename}"
        cv.imwrite(pcx_img_save_path, ximg_orig)

def process_and_find_contours(image, image_path, pg_images_dir, ocrm=None):
    """Process image to find contours and extract regions."""
    if ocrm == None:
        print("\nEnter ocrm model")
        return

    # Binarize, invert, erode, detect edges, and dilate image to connect broken edges
    imgbw = binarize_img(image)
    imgbw = cv.bitwise_not(cv.Canny(imgbw, 100, 200))
    imgbw = cv.erode(imgbw, get_kernel(7, 7), iterations=1)
    imgcan = cv.Canny(imgbw, 100, 200)
    imgcan = cv.dilate(imgcan, get_kernel(7, 7), iterations=15)

    # Find contours
    cnts, _ = cv.findContours(imgcan, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts_img = cv.cvtColor(imgcan, cv.COLOR_GRAY2BGR)
    cnts_img = cv.drawContours(np.zeros_like(cnts_img), cnts, -1, (0, 255, 0, 255), 2)

    # Filter contours for only large contours
    cnts_flt = filter_for_large_cnts(cnts, cnts_img.shape[:2]) 
    cnts_img = cv.drawContours(np.zeros_like(cnts_img), cnts_flt, -1, (0, 255, 0, 255), 2)

    ximg_orig = np.ones_like(cnts_img) * 255

    for idx, cnt in enumerate(cnts_flt):
        x, y, w, h = cv.boundingRect(cnt)

        orig_img_path = os.path.join(pg_images_dir, os.path.basename(image_path))
        orig_img = cv.imread(orig_img_path)
        if orig_img is None:
            print(f"Failed to read original image: ./{rel_path(orig_img_path)}") 
            continue
        ximg_orig[y:y+h, x:x+w] = orig_img[y:y+h, x:x+w]

    # Binarize and remove text from image
    ximg_orig = binarize_img(ximg_orig)
    ximg_orig, _ = remove_txt_paddle(ximg_orig, ocrm=ocrm) 
    return cnts_flt, cnts_img, ximg_orig

def process_image_last(image_path):
    """Process the image by removing lines, finding contours, and saving extracted regions."""
    try:
        image = cv.imread(image_path)
        bw = binarize_img(image) 
        image = bw
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

    # Remove vertical and horizontal lines from the image
    cleaned_image = remove_vertical_lines(image, aspect_ratio_threshold=5)
    cleaned_image = remove_horizontal_lines(cleaned_image, aspect_ratio_threshold=10) 

    # Apply Gaussian blur and unsharp masking to enhance edges
    cleaned_image2 = cv.GaussianBlur(cleaned_image, (9, 9), 1)
    cleaned_image2 = cv.addWeighted(cleaned_image2, 1 + 1.5, cleaned_image2, -1.5, 0)
    cleaned_image = cv.cvtColor(cleaned_image, cv.COLOR_BGR2GRAY) if len(cleaned_image) == 3 else cleaned_image

    # Find contours and filter by area
    contours, _, _ = process_and_find_contours(cleaned_image, image_path, pdf_imgs_dir, ocrm=ocrm)

    # Get contour bbxs and filter by area and aspect ratio (for v, h lines)
    bbxs = [cv.boundingRect(cnt) for cnt in contours]
    bbxs = exclude_full_image_bbox(bbxs, cleaned_image.shape) 
    bbxs = [bbx for bbx in bbxs if ((bbx[2] * bbx[3] > 10000) or (bbx[3] > 0 and bbx[2] / bbx[3] >= 3) or (bbx[2] > 0 and bbx[3] / bbx[2] >= 3))]

    # Merge overlapping bounding boxes by IOU and distance
    merged_bboxes = merge_bboxes_iou(bbxs, iou_threshold=0.001)
    merged_bboxes = merge_close_bboxes(merged_bboxes, threshold=100)

    # Filter bbxs again and save extracted images
    filtered_bboxes = filter_bbxs_by_area(merged_bboxes, 500000)
    filtered_bboxes = apply_padding_to_bboxes(filtered_bboxes, 10, image.shape) 
    if filtered_bboxes:
        save_extracted_images(filtered_bboxes, image_path, output_dir, pdf_imgs_dir)

    # Return the image path to indicate it was processed successfully
    return image_path

def main():
    """Main function to process images in the input directory."""
    processed_files = load_processed_files()
    png_files = sorted(os.listdir(input_dir))[:]

    image_paths = sorted([f"{input_dir}/{fname}" for fname in png_files \
                           if is_image_file(os.path.join(input_dir, fname)) ])# \
                        #    and os.path.join(input_dir, fname) not in processed_files ])
                            
                            
    print(f"\nExtracting images from {len(image_paths)} pages")
    # print(image_paths)

    # Single-process version
    processed_images = []
    for file in image_paths:
        result = process_image_last(file)
        processed_images.append(result)

    # Filter out None values and update the log
    processed_files.update(filter(None, processed_images))
    update_log(processed_files)

    print(f"\nExtracted a total of {len(os.listdir(output_dir))} images")

if __name__ == "__main__":
    # Entry point of the script
    start = time.time()

    # Initialize the OCR model
    global ocrm  # Declare ocrm as global
    logging.getLogger('ppocr').setLevel(logging.WARNING)  # Set logging level to WARNING or ERROR
    ocrm = PaddleOCR(use_angle_cls=True, lang='en')

    main()

    end = time.time()
    # print(f"\nRuntime: {(end - start) / 60:.2f} minutes")
