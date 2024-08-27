# Picaxe PaddleOCR GitHub Project

## Overview

This project is designed to process and analyze PDF images using PaddleOCR, LayoutParser, and other image processing libraries. It includes a pipeline of scripts to remove tables, extract text, and clean up images.

## Setup Instructions

### 1. Clone the Repository

git clone <repository-url>
cd picaxe_paddleocr_github


### 2. Set Up the Virtual Environment
Create and activate a virtual environment:

python3 -m venv padenv
source padenv/bin/activate  # On macOS/Linux
.\padenv\Scripts\activate  # On Windows

### 3. Install Dependencies
pip install -r requirements.txt


### 4. Install Additional Dependencies
#### Poppler (required for pdf2image):

brew install poppler
LayoutParser with PaddleDetection:

pip install "layoutparser[paddledetection]"

## Project Structure

picaxe_paddleocr_github/
│
├── README.md                       # Overview and instructions for the project
├── requirements.txt                # List of dependencies
├── setup.py                        # Setup script (optional)
├── padenv/                         # Virtual environment (can be outside the project)
│
├── main.py                         # Main script to coordinate the entire pipeline
│
├── scripts/                        # Directory for main scripts
│   ├── remove_tables.py            # Step 1 script: Remove tables
│   ├── remove_text.py              # Step 2 script: Remove text
│   ├── remove_scan_borders.py      # Step 3 script: Remove scan borders
│   ├── select_target_images.py     # Step 4 script: Select target images
│   └── extract_and_save_images.py  # Step 5 script: Extract and save images
│
├── src/                            # Directory for source code (utilities and model-related code)
│   ├── utilities.py                # Common utility functions
│   ├── init_layoutparser_model.py  # Model initialization and layout detection code
│   └── other_modules.py            # Any additional reusable code modules
│
├── data/                           # Directory for input data
│   ├── images/                     # Images used in processing
│   ├── bounding_boxes/             # Bounding box data
│   ├── pdf_images_no_tables/       # Processed images with no tables
│   ├── run_1_2_text_removed/       # Images with text removed
│   ├── run_1_3_masking_imgs/       # Masked images from run 1.3
│   └── ...                         # Other data directories as needed
│
├── results/                        # Directory for output results
│   ├── tables/                     # Output tables
│   ├── processed_images/           # Final processed images
│   └── ...                         # Other result files
│
└── logs/                           # Directory for logs
    ├── run_logs/                   # Logs for different runs
    └── processed_files_log.json    # Log for processed files


## Running the Pipeline
To run the entire processing pipeline, use the following command:

python main.py

This will execute the scripts in the following order:

remove_tables.py - Removes tables from the images.
remove_text.py - Removes text from the images.
remove_scan_borders.py - Removes scan borders from the images.
select_target_images.py - Selects target images for further processing.
extract_and_save_images.py - Extracts and saves images based on the processed results.


## Notes
Ensure that the virtual environment is activated before running any scripts.
Modify paths and configurations as needed in the config.py file located in the src/ directory.

## License
[Include license information if applicable.] 