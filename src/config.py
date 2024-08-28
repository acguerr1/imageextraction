# config.py
import os

# Base project directory
project_dir = os.getcwd()

# Default PDF files directory
pdf_files = os.path.join(project_dir, 'data/pdf_files')

# Sample and bulk directories
sample_papers_dir  = os.path.join(project_dir, 'data/pdf_files/sample_papers')
bulk_papers_dir = os.path.join(project_dir, 'data/pdf_files/bulk_papers')

# Script directories
scripts_dir = os.path.join(project_dir, 'scripts')  # Directory for scripts

# Image directories for various processing stages
output_tables_dir = os.path.join(project_dir, 'data/images/tables')  # Output directory for tables
pages_no_tables_dir = os.path.join(project_dir, 'data/images/pdf_images_no_tables')  # Directory for images without tables
pdf_imgs_dir = os.path.join(project_dir, 'data/images/pdf_page_images_2')  # Directory for PDF page images

# Bounding box and text removal directories
bounding_boxes_dir = os.path.join(project_dir, 'data/images/bounding_boxes')  # Duplicate for bounding box data
text_removed_dir = os.path.join(project_dir, 'data/images/text_removed')  # Directory for text-removed images

# Masking, cropped and output image directories
masking_imgs_dir = os.path.join(project_dir, 'data/images/masking_imgs')  # Directory for masking images
# run_1_3_masking_imgs_dir = os.path.join(project_dir, 'data/images/run_1_3_masking_imgs')  # Another reference to masking images
target_images = os.path.join(project_dir, 'data/images/target_images')  # Directory for masked images from run 2.0
extracted_images = os.path.join(project_dir, 'data/images/extracted_images')  # Directory for images from run 3.0
page_output_dir = os.path.join(project_dir, 'data/images/page_output_dir')  # Directory for final page outputs
cropped_dir = os.path.join(project_dir, 'data/images/cropped_images') # border cropped dir

# Log directory
log_dir = os.path.join(project_dir, 'data/logs')  # Directory for log files

# Ensure all directories exist
directories = [
    scripts_dir, output_tables_dir, pages_no_tables_dir, pdf_imgs_dir, 
    bounding_boxes_dir, text_removed_dir, masking_imgs_dir, 
    target_images, extracted_images, page_output_dir, log_dir, cropped_dir, 
    pdf_files, sample_papers_dir, bulk_papers_dir
    ]

for directory in directories:
    os.makedirs(directory, exist_ok=True)  # Create the directories if they don't exist

# Log file path
processed_files_log = os.path.join(log_dir, 'run_1_2_processed_files_log.json')  # Path for processed files log

# LayoutParser Model configuration path
model_config_path = 'lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config'  # Path to the model configuration


# Valid image extensions
valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


