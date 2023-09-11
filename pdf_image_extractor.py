import sys
import os
import argparse
from utils import delete_and_recreate_dir, PDF_PAGES, PAGES_WO_TEXT_DIR, PDF_PATH, EXTRACTED_IMAGES_DIR, file_exists, delete_dir
from pdf_to_image_converter import convert_pdf_to_images
import shutil
import subprocess


#  check that directories are deleted and recreated

def delete_all_temporary_dirs():
    delete_and_recreate_dir(PDF_PAGES)
    delete_and_recreate_dir(PAGES_WO_TEXT_DIR)

# extract the images from extracted_images/ folder and add to new folder

# delete all temporary folders created.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Extraction of pdfs")
    parser.add_argument('--file', dest='file_name', required=True, help='The name of the PDF file to convert.')
    args = parser.parse_args()
    file_name = args.file_name
    pdf_path = os.path.join(PDF_PATH, args.file_name)
    if not file_exists(pdf_path):
        print(f'{pdf_path} not found in directory.')
        sys.exit(1) 
    delete_all_temporary_dirs()
   
    convert_pdf_to_images(pdf_path)

    # pages will be equal to number of elements in the directory where we are individually storing pages
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

    # delete_dir(PDF_PAGES)
    # delete_dir(PAGES_WO_TEXT_DIR)
    
    # extract images from extracted_images to an output folder
    for item in os.listdir(EXTRACTED_IMAGES_DIR):
        item_path = os.path.join(EXTRACTED_IMAGES_DIR, item)
        if not os.path.exists('final_output'):
            os.makedirs('final_output')
        output_item_path = os.path.join('final_output', os.path.splitext(file_name)[0] + '_' + item)
        if os.path.isfile(item_path):
            shutil.copy2(item_path, output_item_path)
        elif os.path.isdir(item_path):
            raise Exception("It can't be a directory. Something went wrong")
