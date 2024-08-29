import os
from pdf2image import convert_from_path
import cv2
import argparse
from utils.noise_removal import remove_borders, detect_barcodes
from utils.box_merging import merge_boxes
from utils.segmentation import combine_predictions_and_crop, predict_layout, predict_table_regions, extract_and_scale_table_regions
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm

# Lazy loading for segmentation models
_layout_model = None
_table_model = None

def get_layout_model():
    global _layout_model
    if _layout_model is None:
        print("Loading full region segmentation model...")
        from huggingface_hub import from_pretrained_keras
        _layout_model = from_pretrained_keras("SBB/eynollah-full-regions-1column")
    return _layout_model

def get_table_model():
    global _table_model
    if _table_model is None:
        print("Loading table segmentation model...")
        from huggingface_hub import from_pretrained_keras
        _table_model = from_pretrained_keras("SBB/eynollah-tables")
    return _table_model

def process_pdf_batch(pdf_batch, input_dir, output_dir, model_paths, use_segmentation=False, threshold=0.25, dilation=5, border_threshold=140, crop_proportion_threshold=0.65, debug=False, remove_barcodes=False):
    os.makedirs(output_dir, exist_ok=True)
    
    temporary = None
    if debug:
        temporary = os.path.join(output_dir, 'temporary')
        os.makedirs(temporary, exist_ok=True)

    # Load YOLO models
    models = [YOLO(model_path) for model_path in model_paths]

    for filename in tqdm(pdf_batch, desc="Processing PDFs", unit="file"):
        if filename.endswith('.pdf'):
            filepath = os.path.join(input_dir, filename)
            print(f'Processing file: {filename}')

            # Adjust crop_proportion_threshold if filename ends with "split"
            adjusted_threshold = crop_proportion_threshold
            if filename.lower().endswith('split.pdf'):
                adjusted_threshold += 0.15

            try:
                images = convert_from_path(filepath, dpi=400)

                for page_num, image in enumerate(images):
                    try:
                        original_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                        # Step 1: Border removal
                        border_removed_image = remove_borders(original_image.copy(), white_threshold=border_threshold)

                        if use_segmentation:
                            # Load segmentation models only if required
                            layout_model = get_layout_model()
                            table_model = get_table_model()

                            # Step 2: Further noise removal using segmentation model
                            layout_prediction = predict_layout(border_removed_image, layout_model)
                            table_prediction, original_height, original_width = predict_table_regions(border_removed_image, table_model)
                            table_boxes = extract_and_scale_table_regions(table_prediction, original_height, original_width)
                            border_removed_image, _, _ = combine_predictions_and_crop(border_removed_image, layout_prediction, table_boxes, adjusted_threshold)

                        if debug:
                            # Save the border-removed (and possibly segmentation-enhanced) image
                            border_removed_image_path = os.path.join(temporary, f'{filename.split(".")[0]}_page_{page_num + 1}_border_removed.png')
                            cv2.imwrite(border_removed_image_path, border_removed_image)
                            print(f'Border-removed image saved: {border_removed_image_path}')

                        detection_image = border_removed_image.copy()

                        # Initialize the boxes list to store bounding boxes for the current page
                        boxes = []

                        if len(models) > 1:
                            # Parallel processing if combined mode is selected
                            def run_yolo(model):
                                verbose_level = True if debug else False
                                return model(border_removed_image, imgsz=640, iou=0.6, verbose=verbose_level)

                            with ThreadPoolExecutor(max_workers=len(models)) as executor:
                                results = list(executor.map(run_yolo, models))

                            for i, model_results in enumerate(results):
                                for result in model_results:
                                    if result.boxes is not None:
                                        for box in result.boxes:
                                            confidence = box.conf.item()
                                            if confidence > threshold:
                                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                                width = x2 - x1
                                                height = y2 - y1 
                                                # Assign color based on the model
                                                color = (0, 0, 255) if model_paths[i].endswith('figure_sensitive.pt') else (255, 0, 0)

                                                if debug:                               
                                                    print(f"Detected box dimensions - Width: {width}, Height: {height} with {confidence}")  
                                                    cv2.rectangle(detection_image, (x1, y1), (x2, y2), color, 6)
                                                    cv2.putText(detection_image, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 6)

                                                boxes.append((x1, y1, x2, y2))

                        else:
                            # Single model processing
                            verbose_level = True if debug else False
                            results = models[0](border_removed_image, imgsz=640, iou=0.6, verbose=verbose_level)

                            for result in results:
                                if result.boxes is not None:
                                    for box in result.boxes:
                                        confidence = box.conf.item()
                                        if confidence > threshold:
                                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())   
                                            # Assign color based on the single model
                                            # Red for figure-sensitive, blue for table-sensitive
                                            color = (0, 0, 255) if model_paths[0].endswith('figure_sensitive.pt') else (255, 0, 0)

                                            if debug:
                                                print(f"Detected box dimensions - Width: {width}, Height: {height} with {confidence}")  
                                                cv2.rectangle(detection_image, (x1, y1), (x2, y2), color, 6)
                                                cv2.putText(detection_image, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 6)

                                            boxes.append((x1, y1, x2, y2))

                        if debug and detection_image is not None:
                            result_image_path = os.path.join(temporary, f"{filename.split('.')[0]}_page_{page_num + 1}_result.png")
                            cv2.imwrite(result_image_path, detection_image)
                            print(f'Result saved: {result_image_path}')

                        # Merge overlapping boxes and process further
                        merged_boxes = merge_boxes(boxes, dilation_iter=dilation)
                   
                        
                        area_threshold = 160000 if use_segmentation else 80000
                        image_count = 0
                        # Add the remaining merged boxes that don't have a large aspect ratio
                        for x1, y1, x2, y2 in merged_boxes:
                            width = x2 - x1
                            height = y2 - y1
                            inv_aspect_ratio = height / width
                            area = width * height

                            if inv_aspect_ratio <= 6 and area >= area_threshold:
                                imc = border_removed_image[y1:y2, x1:x2]
                                
                                # Detect barcode in the cropped image
                                if remove_barcodes and detect_barcodes(imc):
                                    print(f"Barcode detected in cropped image on page {page_num + 1} of file {filename}. Skipping saving.")
                                    continue  # Skip saving the image if barcode is detected

                                image_count += 1
                                image_name = f"{filename.split('.')[0]}_page_{page_num + 1}_image_{image_count}.png"
                                cv2.imwrite(os.path.join(output_dir, image_name), imc)
                                print(f'Detected and saved: {image_name}')
                            else:
                                print(f'Skipped cropping for box with aspect ratio {inv_aspect_ratio:.2f} or area {area} on page {page_num + 1}')
                    except Exception as e:
                        print(f'ERROR: Cannot process page {page_num + 1} of file {filename}. Reason: {e}')
            except Exception as e:
                print(f'ERROR: Cannot process file {filename}. Reason: {e}')

def process_pdfs_in_batches(input_dir, output_dir, model_paths, use_segmentation=False, batch_size=5, threshold=0.2, dilation=5, border_threshold=140, crop_proportion_threshold=0.55, debug=False):
    files = os.listdir(input_dir)
    for i in range(0, len(files), batch_size):
        pdf_batch = files[i:i + batch_size]
        process_pdf_batch(pdf_batch, input_dir, output_dir, model_paths, use_segmentation=use_segmentation, threshold=threshold, dilation=dilation, border_threshold=border_threshold, crop_proportion_threshold=crop_proportion_threshold, debug=debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDFs with YOLO models.")
    parser.add_argument("--input_dir", required=True, help="Directory containing PDF files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save cropped images.")
    parser.add_argument("--figure_sensitive", action="store_true", help="Use figure_sensitive.pt model for figure detection.")
    parser.add_argument("--table_sensitive", action="store_true", help="Use table_sensitive.pt model for table detection.")
    parser.add_argument("--combined", action="store_true", help="Use both figure_sensitive.pt and table_sensitive.pt models for combined detection.")
    parser.add_argument("--use_segmentation", action="store_true", help="Use segmentation model after border removal for enhanced noise removal.")
    parser.add_argument("--debug", action="store_true", help="Keep temporary files for debugging.")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of PDFs to process in each batch.")
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold for YOLO model.")
    parser.add_argument("--dilation", type=int, default=5, help="Dilation parameter for grouping nearby figures.")
    parser.add_argument("--border_threshold", type=int, default=140, help="Pixel intensity threshold for border removal.")
    parser.add_argument("--crop_proportion_threshold", type=float, default=0.65, help="Minimum proportion of original image kept after margin crop")
    parser.add_argument("--remove_barcodes", action="store_true", help="Enable barcode detection and removal in cropped images.")

    args = parser.parse_args()
    
    # Determine which model(s) to use based on user input
    weights_dir = os.path.join(os.getcwd(), "detection_weights")
    model_paths = []
    if args.combined:
        model_paths = [os.path.join(weights_dir, "figure_sensitive.pt"), os.path.join(weights_dir, "table_sensitive.pt")]
    elif args.figure_sensitive:
        model_paths = [os.path.join(weights_dir, "figure_sensitive.pt")]
    elif args.table_sensitive:
        model_paths = [os.path.join(weights_dir, "table_sensitive.pt")]
    else:
        raise ValueError("You must select at least one mode: --figure_sensitive, --table_sensitive, or --combined.")
    
    # Process using the selected model(s), parallel processing only if combined mode is selected
    process_pdfs_in_batches(args.input_dir, args.output_dir, model_paths, use_segmentation=args.use_segmentation, batch_size=args.batch_size, threshold=args.threshold, dilation=args.dilation, border_threshold=args.border_threshold, crop_proportion_threshold=args.crop_proportion_threshold, debug=args.debug)
