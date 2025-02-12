import os
import gc
import sys
import time
import shutil
import cv2 as cv
import numpy as np

# Ensure that the custom utilities module can be imported
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')
sys.path.append(src_path)

# Import utilities and configuration
from utilities import rel_path, extract_and_place_regions, merge_close_bboxes, is_image_file, \
        merge_bboxes_iou, exclude_full_image_bbox, get_kernel, binarize_img, sobel_kernels
from config import config
output_dir = config.target_images
input_dir = config.text_removed_dir

# Main image processing function
# This function processes a single image to detect regions of interest.
def process_image_main_2(pdf_img_path):
    output_file_path = os.path.join(output_dir, os.path.basename(pdf_img_path))

    try:
        # Load and preprocess the image
        page = cv.imread(pdf_img_path)
        image = cv.cvtColor(page, cv.COLOR_BGR2GRAY)
        image = (binarize_img(image, 254)).copy()
    except Exception as e:
        print(f"Error processing ./{rel_path(pdf_img_path)}: {e}")
        return None

    # Apply Sobel filters and convert results to uint8
    sobel_results = {}
    for direction, kernel in sobel_kernels.items(): 
        filtered = cv.filter2D(image, cv.CV_64F, kernel)
        sobel_results[direction] = cv.convertScaleAbs(filtered)

    # Combine all Sobel results to highlight edges in different directions
    sobel_combined = np.zeros_like(sobel_results['sobel_x'])
    for result in sobel_results.values():
        sobel_combined = cv.addWeighted(sobel_combined, 1, result, 0.125, 0)

    sobel_combined = cv.bitwise_not(sobel_combined)
    imgsb = binarize_img(sobel_combined)

    # Erode and detect edges in the image to isolate regions of interest
    imgsb = cv.erode(imgsb, get_kernel(7, 7), iterations=1)
    imgsb = cv.bitwise_not(cv.Canny(imgsb, 100, 200))
    imgsb = cv.erode(imgsb, get_kernel(7, 7), iterations=1)

    # Find and draw contours around the detected regions
    cnts, _ = cv.findContours(imgsb, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Filter contours by height to remove small elements
    cnts_flt_h = filter_cnts(cnts, image)

    # Get bounding boxes and exclude any bounding box that covers the full image
    bboxes = [cv.boundingRect(c) for c in cnts_flt_h]
    bboxes = exclude_full_image_bbox(bboxes, image.shape)

    # Merge overlapping bounding boxes using Intersection over Union (IoU) and distance threshold
    merged_bboxes = merge_bboxes_iou(bboxes, iou_threshold=0.001)
    merged_bboxes = merge_close_bboxes(merged_bboxes, threshold=1000)
    merged_bx_img = extract_and_place_regions(image.copy(), merged_bboxes)

    # Save the processed image with the merged bounding boxes
    cv.imwrite(output_file_path, merged_bx_img)

    # Clear variables to free memory
    del page, image, sobel_results, sobel_combined, imgsb, cnts, cnts_flt_h, bboxes, merged_bboxes, merged_bx_img
    gc.collect()
    time.sleep(5)
    return None

# Filter contours based on height and aspect ratio
# This function filters out contours that do not meet certain height and aspect ratio criteria.
def filter_cnts(proc_img_cnts, image, height=100, aspr=10):
    filtered_contours = []
    _, imw = image.shape[:2]
    for contour in proc_img_cnts:
        _, _, w, h = cv.boundingRect(contour)
        if h < height and w < imw / 2:  # Filter by height and width
            filtered_contours.append(contour)
    return filtered_contours

# Main function to manage image processing
def main():
    png_files = os.listdir(input_dir)
    # Get unprocessed files
    file_paths = sorted([f"{input_dir}/{f}" for f in png_files \
                            if is_image_file(f"{input_dir}/{f}") and  # the file in input_dir should be image files
                                not is_image_file(f"{output_dir}/{f}") ]) # and the file shouldnt have output_dir images yet

    print(f"\nSelecting target images in {len(file_paths)} pages")

    for file in file_paths:
        process_image_main_2(file)

    shutil.rmtree(config.text_removed_dir) # delete bounding boxes dir since they are not needed anymore
    print(f"\nSelecting target images in {len(file_paths)} pages completed")

# Entry point of the script
# This is the entry point of the script. It initializes the processing and tracks the runtime.
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    # print(f"\nRuntime: {(end - start) / 60:.2f} minutes")
