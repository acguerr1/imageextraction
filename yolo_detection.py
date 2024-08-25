import os
from pdf2image import convert_from_path
import cv2
import argparse
from border_removal import remove_borders
from box_merging import merge_boxes
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def process_pdf_batch(pdf_batch, input_dir, output_dir, threshold=0.1, debug=False):
    os.makedirs(output_dir, exist_ok=True)
    # Define the weights directory relative to the current working directory
    weights_dir = os.path.join(os.getcwd(), "weights")

    # Ensure the model paths point to the correct directory
    model_paths = [
        os.path.join(weights_dir, "yolov8(aug 23).pt"),
        os.path.join(weights_dir, "best (yolov8).pt")
    ]

    # Create temporary directory only if debugging is enabled
    temporary = None
    if debug:
        temporary = os.path.join(output_dir, 'temporary')
        os.makedirs(temporary, exist_ok=True)

    for filename in pdf_batch:
        if filename.endswith('.pdf'):
            filepath = os.path.join(input_dir, filename)
            print(f'Processing file: {filename}')
            try:
                # Convert PDF pages to images
                images = convert_from_path(filepath, dpi=400)

                for page_num, image in enumerate(images):
                    try:
                        # Convert PIL image to OpenCV format
                        original_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                        # Remove borders
                        border_removed_image = remove_borders(original_image.copy())

                        # Save border-removed image if debugging
                        if debug:
                            border_removed_image_path = os.path.join(temporary, f'{filename.split(".")[0]}_page_{page_num + 1}_border_removed.png')
                            cv2.imwrite(border_removed_image_path, border_removed_image)
                            print(f'Border-removed image saved: {border_removed_image_path}')

                        # Run YOLO models in parallel using the in-memory image
                        def run_yolo(model):
                            return model(border_removed_image, imgsz=640, iou=0.6)

                        with ThreadPoolExecutor(max_workers=len(model_paths)) as executor:
                            results = list(executor.map(run_yolo, [YOLO(model_path) for model_path in model_paths]))

                        # Process results from both models
                        boxes = []
                        for i, model_results in enumerate(results):
                            for result in model_results:
                                if result.boxes is not None:
                                    for box in result.boxes:
                                        confidence = box.conf.item()
                                        if confidence > threshold:
                                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                            boxes.append((x1, y1, x2, y2))

                                            # Annotate the image with bounding boxes for debugging purposes
                                            if debug:
                                                color = (255, 0, 0) if i == 0 else (0, 0, 255)  # Blue for first model, Red for second model
                                                cv2.rectangle(border_removed_image, (x1, y1), (x2, y2), color, 6)
                                                cv2.putText(border_removed_image, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 6)

                        # Save the annotated image with bounding boxes and confidence scores (for debugging)
                        if debug:
                            result_image_path = os.path.join(temporary, f"{filename.split('.')[0]}_page_{page_num + 1}_result.png")
                            cv2.imwrite(result_image_path, border_removed_image)
                            print(f'Result saved: {result_image_path}')

                        # Merge boxes and crop images from the original border-removed image
                        merged_boxes = merge_boxes(boxes)
                        image_count = 0
                        for x1, y1, x2, y2 in merged_boxes:
                            width = x2 - x1
                            height = y2 - y1
                            aspect_ratio = height / width
                            area = width * height

                            if aspect_ratio <= 6 and (1/aspect_ratio) <= 6 and area >= 60000:
                                imc = border_removed_image[y1:y2, x1:x2]
                                image_count += 1
                                image_name = f"{filename.split('.')[0]}_page_{page_num + 1}_image_{image_count}.png"
                                cv2.imwrite(os.path.join(output_dir, image_name), imc)
                                print(f'Detected and saved: {image_name}')
                            else:
                                print(f'Skipped cropping for box with aspect ratio {aspect_ratio:.2f} or area {area} on page {page_num + 1}')
                    except Exception as e:
                        print(f'ERROR: Cannot process page {page_num + 1} of file {filename}. Reason: {e}')
            except Exception as e:
                print(f'ERROR: Cannot process file {filename}. Reason: {e}')


def process_pdfs_in_batches(input_dir, output_dir, batch_size=5, threshold=0.1, debug=False):
    files = os.listdir(input_dir)
    for i in range(0, len(files), batch_size):
        pdf_batch = files[i:i + batch_size]
        process_pdf_batch(pdf_batch, input_dir, output_dir, threshold=threshold, debug=debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDFs with YOLO models.")
    parser.add_argument("--input_dir", required=True, help="Directory containing PDF files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save cropped images.")
    parser.add_argument("--debug", action="store_true", help="Keep temporary files for debugging.")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of PDFs to process in each batch.")

    args = parser.parse_args()
    process_pdfs_in_batches(args.input_dir, args.output_dir, batch_size=args.batch_size, threshold=0.1, debug=args.debug)
