from pdf2image import convert_from_path
import os
import sys
from utils import file_exists, PDF_PAGES, PDF_PATH
import argparse
from PIL import Image
import io
import cv2

def convert_to_binary():
    if not os.path.exists('binary_images'):
        os.makedirs('binary_images')
    for item in os.listdir(PDF_PAGES):
        image = cv2.imread(os.path.join(PDF_PAGES, item))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        negative_thresh = 255 - thresh
        # cv2.imshow('binary_output.png', gray)
        cv2.imwrite(os.path.join('binary_images', item), negative_thresh)
        # cv2.waitKey(0)



def convert_pdf_to_images(pdf_path, output_dir = PDF_PAGES,  dpi=500):
    print("Converting pdf pages to jpegs...")
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
    # convert_to_binary()
