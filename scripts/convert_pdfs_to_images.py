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
from config import pdf_imgs_dir as output_dir, pdf_files as input_dir

def convert_pdf_to_images(pdf_path, dpi=500):
    """Convert PDF pages to images and save them as PNGs."""
    print("Converting pdf pages to pngs...")
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

def main():
    """Main function to convert selected PDF files to images."""
    # Get list of PDF files in the input PDF directory
    files = sorted([f"{input_dir}/{filename}" for filename in os.listdir(input_dir)])
    
    # Convert the first four PDF files to images
    for file in files[:4]:
        convert_pdf_to_images(file)

if __name__ == "__main__":
    main()  # Execute the main function if the script is run directly
