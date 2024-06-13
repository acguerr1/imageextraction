import cv2
import numpy as np
import pytesseract
import os
import argparse
import dask.dataframe as dd
from PIL import Image
from io import BytesIO
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import EXTRACTED_IMAGES_DIR, IMAGE_BOUNDARIES, delete_and_recreate_dir

def remove_lines_and_borders(image):
    """
    Removes long vertical and horizontal lines and scan borders from the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
    cnts, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w / h > 5:  # Long horizontal lines
            cv2.drawContours(image, [c], -1, (255, 255, 255), 3)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detected_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
    cnts, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h / w > 5:  # Long vertical lines
            cv2.drawContours(image, [c], -1, (255, 255, 255), 3)

    return image

def decode_image(image_data):
    """
    Decodes image bytes into an OpenCV image format.
    """
    im_pil = Image.open(BytesIO(image_data))
    im_cv = np.array(im_pil)
    return cv2.cvtColor(im_cv, cv2.COLOR_RGB2BGR)

def pre_process(image, dilation_iterations):
    """
    Pre-processes an image for contour detection.
    """
    image = remove_lines_and_borders(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_image = cv2.dilate(thresholded_image, kernel, iterations=dilation_iterations)

    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_images(image, dilation_iterations=10, debug_mode=False, output_dir=EXTRACTED_IMAGES_DIR):
    """
    Finds and extracts images from the provided image and stores bounding box coordinates.
    """
    contours = pre_process(image, dilation_iterations)
    output_image = image.copy()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    bounding_boxes = []
    img_cnt = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 500 and h > 500:
            bounding_boxes.append((x, y, x + w, y + h))
            cropped_image = output_image[y:y + h, x:x + w]
            output_image_path = os.path.join(output_dir, f'image_{img_cnt + 1}.png')
            img_cnt += 1
            cv2.imwrite(output_image_path, cropped_image)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (36, 255, 12), 2)

    if debug_mode:
        if not os.path.exists(IMAGE_BOUNDARIES):
            os.makedirs(IMAGE_BOUNDARIES)
        debug_output_path = os.path.join(IMAGE_BOUNDARIES, 'image.png')
        cv2.imwrite(debug_output_path, output_image)

    return bounding_boxes

def remove_small_noises(image, dilation_iterations):
    """
    Removes small noises from the image.
    """
    contours = pre_process(image, dilation_iterations)
    output_image = image.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 500 and h < 500:
            output_image = whiten_pixels(output_image, x, y, w, h)

    return output_image

def remove_text(image, confidence_cutoff, debug_mode, output_dir=EXTRACTED_IMAGES_DIR):
    """
    Removes text from an image using OCR and saves the processed image.
    """
    data = pytesseract.image_to_data(image)
    rows = data.splitlines()

    for row in rows[1:]:
        values = row.split('\t')
        if len(values) < 12 or values[-2] == '-1' or float(values[-2]) < confidence_cutoff or not values[-1].strip():
            continue

        x, y, w, h = int(values[6]) - 10, int(values[7]) - 10, int(values[8]) + 35, int(values[9]) + 35
        image = whiten_pixels(image, x, y, w, h)

        if debug_mode:
            cv2.imshow('Processed Image', image)
            cv2.waitKey(0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_image_path = os.path.join(output_dir, 'image.png')
    cv2.imwrite(output_image_path, image)

def process_image_row(index, row, output_dir='extracted_images'):
    """
    Processes a single row from a DataFrame containing image data.
    """
    try:
        image_data = row['image']['bytes']
        image = decode_image(image_data)
        image = remove_lines_and_borders(image)
        bounding_boxes = find_images(image, output_dir=output_dir)
        row['predicted_bounding_box'] = bounding_boxes
        row['predicted_label'] = 'image'
        return row
    except Exception as e:
        print(f"Error processing row at index {index}: {e}")
        return None

def process_image_data(parquet_path, output_parquet_path, output_dir='extracted_images'):
    """
    Process images from a Parquet file and save the results.
    """
    delete_and_recreate_dir(output_dir)
    
    # Read the Parquet file into a DataFrame
    dataframe = pd.read_parquet(parquet_path,engine='pyarrow')
    dataframe = dataframe.compute()

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:  # Using 8 workers
        futures = [
            executor.submit(process_image_row, index, row, output_dir)
            for index, row in dataframe.iterrows()
        ]
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    # Create a new DataFrame from the results
    updated_dataframe = pd.DataFrame(results)
    # Save the updated DataFrame back to a Parquet file
    updated_dataframe.to_parquet(output_parquet_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from PDF pages or process image data from Parquet.")
    parser.add_argument('--parquet', type=str, help='Path to the Parquet file containing image data.')
    parser.add_argument('--output_parquet', type=str, help='Path to save the updated Parquet file.')

    args = parser.parse_args()

    if args.parquet and args.output_parquet:
        print(f'Processing image data from {args.parquet}')
        process_image_data(args.parquet, args.output_parquet)
    else:
        print("Please provide a valid Parquet file path and output path.")

