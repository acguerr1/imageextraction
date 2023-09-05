import sys
import os
import argparse
from utils import delete_and_recreate_dir, INPUT_DIR_STEP_1, INPUT_DIR_STEP_2, INPUT_DIR_STEP_3, file_exists, delete_dir
from pdf_to_image_converter import convert_pdf_to_images
import shutil
import subprocess


#  check that directories are deleted and recreated

def delete_all_temporary_dirs():
    delete_and_recreate_dir(INPUT_DIR_STEP_1)
    delete_and_recreate_dir(INPUT_DIR_STEP_2)
    delete_and_recreate_dir(INPUT_DIR_STEP_3)




# extract the images from extracted_images/ folder and add to new folder

# delete all temporary folders created.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Extraction of pdfs")
    parser.add_argument('--file', dest='file_name', required=True, help='The name of the PDF file to convert.')
    args = parser.parse_args()
    file_name = args.file_name
    pdf_path = os.path.join('sample_papers', args.file_name)
    if not file_exists(pdf_path):
        print(f'{pdf_path} not found in directory.')
        sys.exit(1) 
    delete_all_temporary_dirs()
    convert_pdf_to_images(pdf_path)

    # pages will be equal to number of elements in the directory where we are individually storing pages
    num_pages = len(os.listdir(INPUT_DIR_STEP_1))

    print(f"Converting {num_pages} pages")

    command = ["python3", "extract_images_2.py", "--pages", str(num_pages)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()
    
    # delete tmep folders
    return_code = process.returncode
    if return_code == 0:
        print("Command executed successfully.")
    else:
        print(f"Command failed with return code {return_code}. Error message:\n{stderr.decode()}")

    delete_dir(INPUT_DIR_STEP_1)
    delete_dir(INPUT_DIR_STEP_2)
    delete_dir('bounded_images')
    # extract images from extracted_images to an output folder
    # some issue here. Correct this
    for item in os.listdir(INPUT_DIR_STEP_3):
        item_path = os.path.join(INPUT_DIR_STEP_3, item)
        print(item_path)
        if not os.path.exists('final_output'):
            os.makedirs('final_output')
        output_item_path = os.path.join('final_output', file_name + item)
        if os.path.isfile(item_path):
            shutil.copy2(item_path, output_item_path)
        elif os.path.isdir(item_path):
            raise Exception("It can't be a directory. Something went wrong")

    # delete extracted_images as well
    delete_dir(INPUT_DIR_STEP_3)