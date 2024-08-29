# init_layoutparser_model.py
import os
import sys
import cv2 as cv
import layoutparser as lp
from paddleocr import PaddleOCR

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Now you can import the functions from utilities.py
from utilities import binarize_img
from config import config 
model_config_path = config.model_config_path

# Initialize the OCR model
ocrm = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize the PaddleOCR model

def initialize_model():
    """Initialize the layout detection model with specific configuration."""
    global model
    model = lp.PaddleDetectionLayoutModel(
        config_path=model_config_path,
        extra_config={"MODEL.ROI_HEADS.SCORE_THRESH_TEST": 0.8},
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )
    return model

def detect_layout(image_path):
    """Detect layout elements in an image and draw bounding boxes."""
    image = cv.imread(image_path)
    image = binarize_img(image)
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    layout = model.detect(image)
    
    # Custom drawing of bounding boxes
    for element in layout:
        x1, y1, x2, y2 = map(int, element.coordinates)
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
        label = f"{element.type}, {element.score:.2f}"
        print(f"Element Type: {label}, with score: {element.score}, Coordinates: ({x1}, {y1}), ({x2}, {y2})")
        cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    return image, layout

# Uncomment the following lines for testing and debugging
# initialize_model()