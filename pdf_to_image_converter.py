from pdf2image import convert_from_path
import os
import sys
from utils import file_exists, PDF_PAGES, PDF_PATH, CROPPED_PAGES, BINARY_PAGES
import argparse
import cv2
import numpy as np


def remove_black_borders(image_path):

    # Removes the dark background which sometimes outlines scanned pages. 
    # Useful when image hasn't been scanned properly

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print(f"No contours found in {image_path}")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
    result = cv2.bitwise_and(img, mask)

    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = result[y:y + h, x:x + w]

    return cropped_image


def process_images_in_folder(input_folder=BINARY_PAGES, output_folder='cropped_images'):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            input_image_path = os.path.join(input_folder, filename)
            cropped_image = remove_black_borders(input_image_path)
            if cropped_image is not None:
                output_image_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_image_path, cropped_image)


def convert_to_binary():
    if not os.path.exists('binary_images'):
        os.makedirs('binary_images')
    for item in os.listdir(PDF_PAGES):
        image = cv2.imread(os.path.join(PDF_PAGES, item))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        negative_thresh = 255 - thresh
        cv2.imwrite(os.path.join('binary_images', item), negative_thresh)


def convert_pdf_to_images(pdf_path, output_dir = PDF_PAGES,  dpi=500):
    print("Converting pdf pages to pngs...")
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pages = convert_from_path(pdf_path, dpi)

    for count, page in enumerate(pages):
        page.save(os.path.join(output_dir, f'pg{count}.png'), 'PNG')

if __name__ == "__main__":
    # Parsing command line argument
    parser = argparse.ArgumentParser(description="PDF to Image Converter")
    parser.add_argument('--file', dest='file_name', required=True, help='The name of the PDF file to convert.')
    args = parser.parse_args()
    
    # Getting path of file
    pdf_path = os.path.join(PDF_PATH, args.file_name)
    if not file_exists(pdf_path):
        print(f'{pdf_path} not found in directory.')
        sys.exit(1)

    # TODO: check if pdf is newer pdf. You can get images of the pdf by using pdf attributes.

    convert_pdf_to_images(pdf_path)
    convert_to_binary()
    process_images_in_folder()