# convert_pdfs_too_images.py
import os
import sys
import argparse
import cv2 as cv
from pdf2image import convert_from_path

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src')
sys.path.append(src_path)

# Import utilities and configuration
from utilities import pil2ndarr
from config import config

input_dir = config.pdf_files
output_dir = config.pdf_imgs_dir


def convert_pdf_to_images(pdf_path, dpi=500):
    """Convert PDF pages to images and save them as PNGs."""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the base name of the PDF file without the extension
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi)

    for count, page in enumerate(pages):
        # Generate the output filename and path
        output_filename = f"{base_name}_page{count}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Convert the page to a NumPy array and save as PNG
        page = pil2ndarr(page)
        cv.imwrite(output_path, page)

def main(args):
    """Main function to convert selected PDF files to images."""
    if args.file_name:
        # If a specific file is provided, convert only that file
        files = [os.path.join(input_dir, args.file_name)]
        if not os.path.isfile(files[0]):
            print(f'{files[0]} not found in directory.')
            sys.exit(1)
    else:
        # Otherwise, convert all PDF files in the input directory
        files = sorted([os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith('.pdf')])
    
    # Convert PDF files to images
    print(f"Converting {len(files)} pdf pages to pngs...")
    for file in files:
        convert_pdf_to_images(file)
    
    print(f"Converting {len(files)} pdf pages to pngs completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF files to PNG images.")
    parser.add_argument('--file', dest='file_name', required=False, help='The name of the PDF file to convert.')
    args = parser.parse_args()

    main(args)  # Execute the main function if the script is run directly
