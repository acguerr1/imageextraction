import os
from pdf2image import convert_from_path
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
import cv2
import torch

# Initialize the processor and detector with the DETR model
# General Purpose Detector
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)

def convert_pdf_to_images(pdf_path, dpi=500):
    try:
        return convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def non_max_suppression(boxes, scores, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

def process_image(image, image_index, output_folder, annotation_file, threshold=0.3, iou_threshold=0.5):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = detector(**inputs)

    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

    print(f"Processing page {image_index}, detected {len(boxes)} visual elements.")

    if len(boxes) == 0:
        print(f"No visual elements detected on page {image_index}")
        return

    boxes_np = [box.tolist() for box in boxes]
    scores_np = scores.tolist()
    indices = non_max_suppression(boxes_np, scores_np, iou_threshold)

    for idx in indices:
        box = boxes_np[idx]
        score = scores_np[idx]
        label = labels[idx].item()

        x1, y1, x2, y2 = map(int, box)
        cropped_image = image_np[max(0, y1):min(image_np.shape[0], y2), max(0, x1):min(image_np.shape[1], x2)]

        if cropped_image.size > 0:
            cropped_image_path = os.path.join(output_folder, f"cropped_page_{image_index}_{idx}.jpg")
            cv2.imwrite(cropped_image_path, cropped_image)
            with open(annotation_file, 'a') as f:
                f.write(f"{cropped_image_path}, {label}, {score:.2f}, {box}\n")

def main(pdf_file):
    base_name = os.path.splitext(os.path.basename(pdf_file))[0]
    final_output_dir = os.path.join("DETR_Output", base_name)
    ensure_directory_exists(final_output_dir)

    annotation_file = os.path.join(final_output_dir, "annotations.txt")
    if os.path.exists(annotation_file):
        os.remove(annotation_file)

    images = convert_pdf_to_images(pdf_file)
    for idx, image in enumerate(images):
        process_image(image, idx, final_output_dir, annotation_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect and crop visual elements in PDF files using DETR")
    parser.add_argument('--file', required=True, help='The path to the PDF file to process.')
    args = parser.parse_args()
    main(args.file)
