# Image Extraction Using Object Detection

This branch focuses on image extraction from PDFs using object detection models. The repository provides tools for detecting and extracting figures and tables from documents using YOLO models and other related preprocessing and postprocessing techniques to center detection region and resolve overlapping detection boxes.

## Installation 🖥️

To get started with this project, follow these installation steps:

### 1. Clone the Repository and Checkout the Branch

You can clone the repository and directly switch to the `PicAxe_YOLO` branch:

```bash
git clone -b PicAxe_YOLO https://github.com/acguerr1/imageextraction.git
cd imageextraction
```

### 2. Install Dependencies 

**Install the required Python packages and system dependencies:**
```bash
pip install -r requirements.txt
```

For macOS: Use Homebrew to install poppler-utils and zbar:
```bash
brew install poppler
brew install zbar
```

For Windows:
1. Download and install Poppler from Poppler for Windows. After extraction, add the bin/ folder to your system's PATH.
2. Download and install ZBar from ZBar project.

**Pretrained YOLOv8 Weights**:

Users should download our weights from the following URL:

[Download YOLOv8 Weights](https://drive.google.com/drive/folders/1PiPbbhUsw95kdpfAmKlm6Xq1RfcIuu3p?usp=sharing)

After downloading, please place the weights file in the `detection_weights` folder in your project directory

## Quick Start 🌟

Option 1. Users can run `demo.ipynb` on Google Colab.
Option 2. Docker image: 
#### 2.1 Pull the Docker Image
First, pull the Docker image using the following command:

```bash
docker pull qilinzhou56/yolo-image-extractor:tagname
```
#### 2.2 Navigate to Your Project Folder (e.g., image extraction)
```bash
cd /path/to/your/project_folder
```
#### 2.3 Run the Docker Container
```bash
docker run -v $(pwd)/detection_weights:/app/detection_weights \
    -v $(pwd)/input_dir:/app/input_dir \
    -v $(pwd)/output_dir:/app/output_dir \
    qilinzhou56/yolo-image-extractor:tagname --input_dir /app/input_dir --output_dir /app/output_dir --combined
```

## Run the Extraction Script 🏃
```bash
python yolo_detection.py --input_dir <input_directory> --output_dir <output_directory> --combined
```
You can select one of the following detection modes by using the appropriate flag:
- `--figure_sensitive` for enhanced figure detection
- `--table_sensitive` for optimized table detection
- `--combined` for maximizing recall for both figures and tables

#### Other Optional Parameters
- `--use_segmentation`
  - **Description**: Use Eynollah segmentation models after border removal for enhanced noise removal.
  - **Usage**: Include this flag to apply additional noise removal using a segmentation model.
  - **Range**: This is a boolean flag, meaning it can be either included (True) or omitted (False).

- `--debug`
  - **Description**: Keep temporary files for debugging purposes.
  - **Usage**: Include this flag if you want to retain intermediate files for debugging, including border_removed image and detection boxes.
  - **Range**: This is a boolean flag, meaning it can be either included (True) or omitted (False).

- `--batch_size` (default: `5`)
  - **Description**: Number of PDFs to process in each batch.
  - **Usage**: Specify the number of PDFs to be processed together in a single batch.
  - **Suggested Range**: `1` to `50`
  - **Meaning**: Lower values (closer to 1) process fewer PDFs at a time, reducing memory usage. Higher values (closer to 50) process more PDFs simultaneously but require more memory.

- `--threshold` (default: `0.25`)
  - **Description**: Confidence threshold for YOLO model detections. Detections with confidence below this value will be ignored.
  - **Usage**: Adjust this value to filter out less confident detections.
  - **Range**: `0.0` to `1.0`
  - **Meaning**: A threshold closer to 0.0 includes all detections, even low-confidence ones. A threshold closer to 1.0 only includes very high-confidence detections.

- `--dilation` (default: `5`)
  - **Description**: Dilation parameter to group nearby detected figures into larger regions.
  - **Usage**: Specify the number of pixels to expand detected regions for grouping nearby elements.
  - **Suggested Range**: `0` to `25`
  - **Meaning**: Lower values (closer to 1) result in minimal grouping of detected regions. Higher values (closer to 20) group more elements together, which can be useful for detecting closely spaced figures.

- `--border_threshold` (default: `140`)
  - **Description**: Pixel intensity threshold for border removal during preprocessing.
  - **Usage**: Set this value to control the detection and removal of borders based on pixel intensity.
  - **Range**: `0` to `255`
  - **Meaning**: Lower values (closer to 0) detect darker borders, while higher values (closer to 255) detect lighter borders.

- `--crop_proportion_threshold` (default: `0.65`)
  - **Description**: Minimum proportion of the original image that should be retained after margin cropping.
  - **Usage**: Adjust this value to ensure that the cropped image retains a certain percentage of the original image size.
  - **Suggested Range**: `0.2` to `0.9`
  - **Meaning**: A lower threshold (closer to 0.5) retains at least half of the original image size. A higher threshold (closer to 0.9) ensures that most of the image is retained after cropping. Highly skewed or poorly-scanned documents should have a lower threshold as it often has a large border to crop.

- `--remove_barcodes`
  - **Description**: Enables detection and removal of barcodes in cropped images with Pyzbar.
  - **Usage**: Particularly useful for scanned documents where barcodes might be present and detected as figures, which should be excluded from further processing.
  - **Range**: This is a boolean flag, meaning it can be either included (True) or omitted (False).

## Quality Control 🧑‍✈️
**Current Achievements:**
- **Lightweight Detection:** Utilizes the lightweight YOLOv8 model to quickly pinpoint the positions of figures and tables within scanned historical documents.
- **Preprocessing & Postprocessing:** Adds preprocessing steps for noise removal with an option for main region crop (referenced from **Eynollah**) and postprocessing steps with a choice to group or divide nearby figures, which is crucial for handling the complexities of these documents.
- **Adaptability:** Trained on "shabby" documents, making the model flexible and capable of detecting figures and tables even in poorly scanned or degraded quality documents.
- **High Figure Recall** 

**Areas for Improvement (Mainly for table extraction):**
- **Single-Class Training Limitation:** The current YOLOv8 model is trained for single-class detection, which may not fully capture the complexity of tables and sometimes results in over-extraction of table-like content, such as equations.
- **Future Enhancements:** Future researchers could improve upon this by using the pre-trained weights for multi-class training, allowing the model to distinguish more accurately between tables and other similar content.
