import sys
import os
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

# Import functions from utility and image processing scripts
from utils import (
    delete_and_recreate_dir, PDF_PATH, PDF_PAGES, PAGES_WO_TEXT_DIR, EXTRACTED_IMAGES_DIR, 
    BULK_PATH, BINARY_PAGES, file_exists, delete_dir, CROPPED_PAGES
)
from pdf_to_image_converter import convert_pdf_to_images, convert_to_binary, process_images_in_folder

# Path configuration
# BULK_PATH = BULK_PATH + "/dataset/biofilm_corpus_pdfs"

# Function to delete and recreate directories
def delete_all_temporary_dirs():
    delete_and_recreate_dir(PDF_PAGES)
    delete_and_recreate_dir(BINARY_PAGES)
    delete_and_recreate_dir(CROPPED_PAGES)
    delete_and_recreate_dir(PAGES_WO_TEXT_DIR)

def process_pdf(pdf_path):
    try:
        print(f'Working on {pdf_path}')
        delete_all_temporary_dirs()

        convert_pdf_to_images(pdf_path)
        convert_to_binary()
        process_images_in_folder()

        num_pages = len(os.listdir(PDF_PAGES))
        print(f"Extracting images from {num_pages} pages")

        command = ["python3", "extract_images.py", "--pages", str(num_pages)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()
        return_code = process.returncode
        if return_code == 0:
            print("Image extraction successful.")
        else:
            print(f"Command failed with return code {return_code}. Error message:\n{stderr.decode()}")

        delete_dir(PDF_PAGES)
        delete_dir(PAGES_WO_TEXT_DIR)
        delete_dir(BINARY_PAGES)
        delete_dir(CROPPED_PAGES)

        # extract images from extracted_images to an output folder
        for item in os.listdir(EXTRACTED_IMAGES_DIR):
            item_path = os.path.join(EXTRACTED_IMAGES_DIR, item)
            if not os.path.exists('final_output'):
                os.makedirs('final_output')
            output_item_path = os.path.join('final_output', os.path.splitext(pdf_path.split('/')[-1])[0] + '_' + item)
            if os.path.isfile(item_path):
                shutil.copy2(item_path, output_item_path)
            elif os.path.isdir(item_path):
                raise Exception("It can't be a directory. Something went wrong")

        delete_dir(EXTRACTED_IMAGES_DIR)
    except Exception as e:
        print(f"An error occurred while processing {pdf_path}: {e}")

def bulk_mode():
    file_paths = [os.path.join(BULK_PATH, f) for f in os.listdir(BULK_PATH) if os.path.isfile(os.path.join(BULK_PATH, f)) and not f.startswith('.')]
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_pdf, pdf_path): pdf_path for pdf_path in file_paths}
        for future in as_completed(futures):
            pdf_path = futures[future]
            try:
                future.result()
                print(f"Finished processing {pdf_path}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

# Main function to handle command-line arguments and process PDFs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Extraction of PDFs")
    parser.add_argument('--file', dest='file_name', required=False, help='The name of the PDF file to convert.')
    parser.add_argument('--bulk', action='store_true', help='Enable bulk processing mode')
    args = parser.parse_args()

    if not args.file_name and not args.bulk:
        parser.error("Either --file or --bulk must be provided.")
        sys.exit(1)

    if args.file_name:
        pdf_path = os.path.join(PDF_PATH, args.file_name)
        delete_all_temporary_dirs()

        convert_pdf_to_images(pdf_path)
        convert_to_binary()
        process_images_in_folder()

        num_pages = len(os.listdir(PDF_PAGES))
        print(f"Extracting images from {num_pages} pages")

        command = ["python3", "extract_images.py", "--pages", str(num_pages)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()
        return_code = process.returncode
        if return_code == 0:
            print("Image extraction successful.")
        else:
            print(f"Command failed with return code {return_code}. Error message:\n{stderr.decode()}")

        delete_dir(PDF_PAGES)
        delete_dir(PAGES_WO_TEXT_DIR)
        delete_dir(BINARY_PAGES)
        delete_dir(CROPPED_PAGES)

        # extract images from extracted_images to an output folder
        for item in os.listdir(EXTRACTED_IMAGES_DIR):
            item_path = os.path.join(EXTRACTED_IMAGES_DIR, item)
            if not os.path.exists('final_output'):
                os.makedirs('final_output')
            output_item_path = os.path.join('final_output', os.path.splitext(args.file_name)[0] + '_' + item)
            if os.path.isfile(item_path):
                shutil.copy2(item_path, output_item_path)
            elif os.path.isdir(item_path):
                raise Exception("It can't be a directory. Something went wrong")

        delete_dir(EXTRACTED_IMAGES_DIR)
    else:
        bulk_mode()
