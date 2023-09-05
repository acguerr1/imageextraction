from pdf2image import convert_from_path
import os
import sys
from utils import file_exists, INPUT_DIR_STEP_1
import argparse


def convert_pdf_to_images(pdf_path, output_dir = INPUT_DIR_STEP_1,  dpi=500):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pages = convert_from_path(pdf_path, 500)

    for count, page in enumerate(pages):
        page.save(os.path.join(output_dir, f'out{count}.png'), 'PNG')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF to Image Converter")
    parser.add_argument('--file', dest='file_name', required=True, help='The name of the PDF file to convert.')
    args = parser.parse_args()
    pdf_path = os.path.join('sample_papers', args.file_name)
    if not file_exists(pdf_path):
        print(f'{pdf_path} not found in directory.')
        sys.exit(1)
    convert_pdf_to_images(pdf_path)
