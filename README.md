# Picaxe-OCR

## Overview

This software tool is designed to process and analyze PDF images using PaddleOCR, LayoutParser, and other image processing libraries. It includes a pipeline of scripts to remove tables, extract text, and clean up images.

## Running
Installing dependencies
```bash
pip install -r requirements.txt
brew install poppler
```

Running main.py with options
```bash
python main.py --file example_file.pdf
python main.py --bulk
python main.py --sample
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>  <local-destination-name>
cd <local-destination-name>
```

### 2. Set Up the Virtual Environment

Create and activate a virtual environment:

```bash
pip install virtualenv
virtualenv <venv-name>
python -m venv padenv
source <venv-name>/bin/activate  # On macOS/Linux
.\<venv-name>\Scripts\activate  # On Windows
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

1. Install Homebrew if you dont have it already:
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
python main.py --bulk
python main.py --sample
python main.py --file filename
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

[......]
