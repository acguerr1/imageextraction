# Picaxe PaddleOCR GitHub Project

## Overview

This software tool is designed to process and analyze PDF images using PaddleOCR, LayoutParser, and other image processing libraries. It includes a pipeline of scripts to remove tables, extract text, and clean up images.

## Running

```bash
pip install -r requirements.txt

python main.py --file example_file.pdf
python main.py --bulk
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd picaxe_paddleocr
```

### 2. Set Up the Virtual Environment

Create and activate a virtual environment:

```bash
python3 -m venv padenv
source padenv/bin/activate  # On macOS/Linux
.\padenv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

or run 

```bash
python ./src/install_pkgs.py
```

### 4. Install Additional Dependencies

#### Poppler (required for pdf2image):

**Linux and macOS:**

1. Install Homebrew:
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
2. Verify Homebrew Version:
    ```bash
    brew --version
    ```
3. Now install Poppler with Homebrew:
    ```bash
    brew install poppler
    ```

**Windows:**

1. Install Chocolatey from this website:  
   [https://chocolatey.org/](https://chocolatey.org/)
2. Run:
    ```bash
    choco install poppler
    ```

## Running the Pipeline

To run the entire processing pipeline, use the following command:

```bash
python main.py
```

This will execute the scripts in the following order:

1. **convert_pdfs_to_images.py** - Converts PDF file pages to PNG images.
2. **crop_borders.py** - Crops the borders of the pages if they have scan borders.
3. **remove_tables.py** - Removes tables from the images.
4. **remove_text.py** - Removes text from the images.
5. **select_target_images.py** - Selects target images for further processing.
6. **extract_and_save_images.py** - Extracts and saves images based on the processed results.

## Notes

- Ensure that the virtual environment is activated before running any scripts.
- Modify paths and configurations as needed in the `config.py` file located in the `src/` directory.

## License

[Include license information if applicable.]
