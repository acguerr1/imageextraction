# remove_scan_borders.py
import os
import gc  # Import garbage collection module
import sys
import time
import logging
import cv2 as cv
import numpy as np
from paddleocr import PaddleOCR

# Ensure that the custom utilities module can be imported
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')
sys.path.append(src_path)

# Import utilities and configuration
from utilities import rel_path, remove_txt_paddle, binarize_img, clean_image_filter2D, exclude_full_image_bbox, is_file_already_processed
from config import text_removed_dir as input_dir, masking_imgs_dir as output_dir, pdf_imgs_dir, cropped_dir

# Function to get the original image from the PDF images directory
def get_orig(image_path):
    img = cv.imread(f'{cropped_dir}/{os.path.basename(image_path)}')
    # img = cv.imread(f'{pdf_imgs_dir}/{os.path.basename(image_path)}')
    return img

# Function to process a single image: remove scan borders and apply OCR
def process_image(image_path):
    global ocrm  # Access the global ocrm variable
    
    try:
        # Load and process the image
        print(f"\nWorking on ./{rel_path(image_path)}")
        img = cv.imread(image_path)

        if img is None:
            print(f"Failed to read image file: ./{rel_path(image_path)}")
            return None
        
        # Binarize the image and erode to enhance the structures
        imgray = binarize_img(img.copy())
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        imgray = cv.erode(imgray, kernel, iterations=20)

        # Find contours and filter by area
        contours, _ = cv.findContours(imgray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv.contourArea(cnt) > 15000]

        # Get bounding boxes and exclude the full-image bounding box
        bbxs = [cv.boundingRect(cnt) for cnt in contours]
        bbxs = exclude_full_image_bbox(bbxs, img.shape)

        # Draw bounding boxes on the original image
        imgcnt = img.copy()
        for x, y, w, h in bbxs:
            cv.rectangle(imgcnt, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.putText(imgcnt, str(w*h), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Create a mask to preserve the original image content within the bounding boxes
        mask = np.ones_like(img) * 255
        for x, y, w, h in bbxs:
            orig = get_orig(image_path)
            mask[y:y+h, x:x+w] = orig[y:y+h, x:x+w]
            del orig  # Free memory for the original image
            gc.collect()

        # Binarize and clean the mask
        mask = binarize_img(mask)
        mask2 = clean_image_filter2D(mask)
        img_bw_flt = cv.bitwise_and(mask, mask, mask=mask2)
        img_bw_flt[mask2 == 0] = 255
        img_bw_flt = binarize_img(img_bw_flt, 250)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        img_bw_flt = cv.erode(img_bw_flt, kernel, iterations=1)

        # Remove text from the processed image using OCR
        no_txt_img, ocr_result = remove_txt_paddle(img_bw_flt.copy(), ocrm=ocrm, margin=1)
        
        # Save the processed image
        save_path = f"{output_dir}/{os.path.basename(image_path)}"
        cv.imwrite(save_path, no_txt_img)
        print(f"Saved ./{rel_path(save_path)} successfully")

        # Perform garbage collection to free memory
        del img, imgray, imgcnt, mask, img_bw_flt, no_txt_img, mask2, bbxs, contours
        gc.collect()
        time.sleep(1)

    except Exception as e:
        print(f"Error processing ./{rel_path(image_path)}: {e}")
        gc.collect()  # Ensure any memory is cleared even on failure
    
    return None

# Main function to process all images in the input directory
def main():

    # Build the list of image paths, excluding those that are already processed
    # image_paths = sorted([f"{input_dir}/{fname}" for fname in os.listdir(input_dir)])
    image_paths = sorted([f"{input_dir}/{fname}" for fname in os.listdir(input_dir)\
                   if not is_file_already_processed(os.path.splitext(fname)[0], output_dir) \
                    or 'Conn_1932_page4'.lower() in os.path.splitext(fname)[0].lower() ])
    print(f"Total unprocessed images: {len(image_paths)}")

    if not image_paths:
        print("All images have been processed.")
        return None

    for f in image_paths[:]:
        process_image(f)

    print("\nInput processed successfully!!!")
    gc.collect()  # Final garbage collection after all processing
    return None

# Main execution block
if __name__ == '__main__':

    start = time.time()
    # Initialize the OCR model
    global ocrm  # Declare ocrm as global
    # Set logging level to WARNING or ERROR
    logging.getLogger('ppocr').setLevel(logging.WARNING)
    ocrm = PaddleOCR(use_angle_cls=True, lang='en')
    main()

    end = time.time()
    # print(f"\nRuntime: {(end - start) / 60:.2f} minutes")
