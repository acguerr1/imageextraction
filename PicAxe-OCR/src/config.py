import os

class Config:
    def __init__(self):
        # Base project directory
        self.project_dir = os.getenv('PROJECT_DIR', os.getcwd())

        # Default PDF files directory
        self.pdf_files = os.getenv('PDF_FILES', os.path.join(self.project_dir, 'data/pdf_files'))

        # Sample and bulk directories
        self.sample_papers_dir = os.getenv('SAMPLE_PAPERS_DIR', os.path.join(self.project_dir, 'data/pdf_files/sample_papers'))
        self.bulk_papers_dir = os.getenv('BULK_PAPERS_DIR', os.path.join(self.project_dir, 'data/pdf_files/bulk_papers'))

        # Script directories
        self.scripts_dir = os.getenv('SCRIPTS_DIR', os.path.join(self.project_dir, 'scripts'))

        # Image directories for various processing stages
        self.output_tables_dir = os.getenv('OUTPUT_TABLES_DIR', os.path.join(self.project_dir, 'data/images/tables'))
        self.pages_no_tables_dir = os.getenv('PAGES_NO_TABLES_DIR', os.path.join(self.project_dir, 'data/images/pdf_images_no_tables'))
        self.pdf_imgs_dir = os.getenv('PDF_IMGS_DIR', os.path.join(self.project_dir, 'data/images/pdf_page_images'))

        # Bounding box and text removal directories
        self.bounding_boxes_dir = os.getenv('BOUNDING_BOXES_DIR', os.path.join(self.project_dir, 'data/images/bounding_boxes'))
        self.text_removed_dir = os.getenv('TEXT_REMOVED_DIR', os.path.join(self.project_dir, 'data/images/text_removed'))

        # Masking, cropped and output image directories
        self.masking_imgs_dir = os.getenv('MASKING_IMGS_DIR', os.path.join(self.project_dir, 'data/images/masking_imgs'))
        self.target_images = os.getenv('TARGET_IMAGES', os.path.join(self.project_dir, 'data/images/target_images'))
        self.extracted_images = os.getenv('EXTRACTED_IMAGES', os.path.join(self.project_dir, 'data/images/extracted_images'))
        self.page_output_dir = os.getenv('PAGE_OUTPUT_DIR', os.path.join(self.project_dir, 'data/images/page_output_dir'))
        self.cropped_dir = os.getenv('CROPPED_DIR', os.path.join(self.project_dir, 'data/images/cropped_images'))

        # Log directory
        self.log_dir = os.getenv('LOG_DIR', os.path.join(self.project_dir, 'data/logs'))

        # Ensure all directories exist
        directories = [
            self.scripts_dir, self.output_tables_dir, self.pages_no_tables_dir, self.pdf_imgs_dir, 
            self.bounding_boxes_dir, self.text_removed_dir, self.masking_imgs_dir, 
            self.target_images, self.extracted_images, self.page_output_dir, self.log_dir, self.cropped_dir, 
            self.pdf_files, self.sample_papers_dir, self.bulk_papers_dir
            ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        # Log file path
        self.processed_files_log = os.getenv('PROCESSED_FILES_LOG', os.path.join(self.log_dir, 'run_1_2_processed_files_log.json'))

        # LayoutParser Model configuration path
        self.model_config_path = os.getenv('MODEL_CONFIG_PATH', 'lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config')

        # Valid image extensions
        self.valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

config = Config()


# # config.py
# import os

# # Base project directory
# project_dir = os.getcwd()

# # Default PDF files directory
# pdf_files = os.path.join(project_dir, 'data/pdf_files')

# # Sample and bulk directories
# sample_papers_dir  = os.path.join(project_dir, 'data/pdf_files/sample_papers')
# bulk_papers_dir = os.path.join(project_dir, 'data/pdf_files/bulk_papers')

# # Script directories
# scripts_dir = os.path.join(project_dir, 'scripts')  # Directory for scripts

# # Image directories for various processing stages
# output_tables_dir = os.path.join(project_dir, 'data/images/tables')  # Output directory for tables
# pages_no_tables_dir = os.path.join(project_dir, 'data/images/pdf_images_no_tables')  # Directory for images without tables
# pdf_imgs_dir = os.path.join(project_dir, 'data/images/pdf_page_images_2')  # Directory for PDF page images

# # Bounding box and text removal directories
# bounding_boxes_dir = os.path.join(project_dir, 'data/images/bounding_boxes')  # Duplicate for bounding box data
# text_removed_dir = os.path.join(project_dir, 'data/images/text_removed')  # Directory for text-removed images

# # Masking, cropped and output image directories
# masking_imgs_dir = os.path.join(project_dir, 'data/images/masking_imgs')  # Directory for masking images
# # run_1_3_masking_imgs_dir = os.path.join(project_dir, 'data/images/run_1_3_masking_imgs')  # Another reference to masking images
# target_images = os.path.join(project_dir, 'data/images/target_images')  # Directory for masked images from run 2.0
# extracted_images = os.path.join(project_dir, 'data/images/extracted_images')  # Directory for images from run 3.0
# page_output_dir = os.path.join(project_dir, 'data/images/page_output_dir')  # Directory for final page outputs
# cropped_dir = os.path.join(project_dir, 'data/images/cropped_images') # border cropped dir

# # Log directory
# log_dir = os.path.join(project_dir, 'data/logs')  # Directory for log files

# # Ensure all directories exist
# directories = [
#     scripts_dir, output_tables_dir, pages_no_tables_dir, pdf_imgs_dir, 
#     bounding_boxes_dir, text_removed_dir, masking_imgs_dir, 
#     target_images, extracted_images, page_output_dir, log_dir, cropped_dir, 
#     pdf_files, sample_papers_dir, bulk_papers_dir
#     ]

# for directory in directories:
#     os.makedirs(directory, exist_ok=True)  # Create the directories if they don't exist

# # Log file path
# processed_files_log = os.path.join(log_dir, 'run_1_2_processed_files_log.json')  # Path for processed files log

# # LayoutParser Model configuration path
# model_config_path = 'lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config'  # Path to the model configuration


# # Valid image extensions
# valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


