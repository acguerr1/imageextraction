# Figure Extraction Using Deep Learning

This branch focuses on figure extraction from PDFs using deep learning models. The repository provides tools for detecting and extracting figures and tables from documents using YOLO models and other related techniques.

## Installation

To get started with this project, follow these installation steps:

### 1. Clone the Repository and Checkout the Branch

You can clone the repository and directly switch to the `dlmodel` branch:

```bash
git clone -b dlmodel https://github.com/acguerr1/imageextraction.git
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

### 3. Run the Extraction Script
```bash
python yolo_detection.py --input_dir <input_directory> --output_dir <output_directory> --combined
```
You can select one of the following detection modes by using the appropriate flag:
- `--figure_sensitive` for enhanced figure detection
- `--table_sensitive` for optimized table detection
- `--combined` for maximizing recall for both figures and tables

#### Other Optional Parameters
- `--use_segmentation`: 
  - **Description**: Use a segmentation model after border removal for enhanced noise removal.
  - **Usage**: Include this flag to apply additional noise removal using a segmentation model.

- `--debug`: 
  - **Description**: Keep temporary files for debugging purposes.
  - **Usage**: Include this flag if you want to retain intermediate files for debugging, including border_removed image and detection boxes.

- `--batch_size` (default: `5`): 
  - **Description**: Number of PDFs to process in each batch.
  - **Usage**: Specify the number of PDFs to be processed together in a single batch.

- `--threshold` (default: `0.25`): 
  - **Description**: Confidence threshold for YOLO model detections. Detections with confidence below this value will be ignored.
  - **Usage**: Adjust this value to filter out less confident detections.

- `--dilation` (default: `5`): 
  - **Description**: Dilation parameter to group nearby detected figures into larger regions.
  - **Usage**: Specify the number of pixels to expand detected regions for grouping nearby elements.

- `--border_threshold` (default: `140`): 
  - **Description**: Pixel intensity threshold for border removal during preprocessing.
  - **Usage**: Set this value to control the detection and removal of borders based on pixel intensity.

- `--crop_proportion_threshold` (default: `0.65`): 
  - **Description**: Minimum proportion of the original image that should be retained after margin cropping.
  - **Usage**: Adjust this value to ensure that the cropped image retains a certain percentage of the original image size.

### 4. Quality Control
**Current Achievements:**
- **Lightweight Detection:** Utilizes the lightweight YOLOv8 model to quickly pinpoint the positions of figures and tables within scanned historical documents.
- **Preprocessing & Postprocessing:** Adds preprocessing steps for noise removal with an option for main region crop and postprocessing steps with a choice to group or divide nearby figures, which is crucial for handling the complexities of these documents.
- **Adaptability:** Trained on "shabby" documents, making the model flexible and capable of detecting figures and tables even in poorly scanned or degraded quality documents.
- **High Figure Recall** 

**Areas for Improvement (Mainly for table extraction):**
- **Single-Class Training Limitation:** The current YOLOv8 model is trained for single-class detection, which may not fully capture the complexity of tables and sometimes results in over-extraction of table-like content, such as equations.
- **Future Enhancements:** Future researchers could improve upon this by using the pre-trained weights for multi-class training, allowing the model to distinguish more accurately between tables and other similar content.
