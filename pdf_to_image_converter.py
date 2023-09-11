from pdf2image import convert_from_path
import os
import sys
from utils import file_exists, PDF_PAGES, PDF_PATH
import argparse
import fitz
from PIL import Image
import io



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
