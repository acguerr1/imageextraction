import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import easyocr
from skimage import filters
from imutils.object_detection import non_max_suppression

# Initialize EasyOCR reader
reader = easyocr.Reader(["en"])  # Add more languages if needed


def is_paragraph_text(x, y, w, h, image_width, image_height):
    aspect_ratio = w / float(h)
    if w > 0.5 * image_width and aspect_ratio > 2:
        return True
    if h < 0.1 * image_height:
        return True
    return False


def remove_text_with_easyocr(image):
    results = reader.readtext(image)
    boxes = []
    confidences = []
    image_height, image_width = image.shape[:2]

    for bbox, text, prob in results:
        if prob > 0.85:
            (tl, tr, br, bl) = bbox
            tl = tuple(map(int, tl))
            br = tuple(map(int, br))
            x, y = tl
            w, h = br[0] - x, br[1] - y
            if is_paragraph_text(x, y, w, h, image_width, image_height):
                boxes.append([x, y, x + w, y + h])
                confidences.append(prob)

    if len(boxes) == 0:
        print("No paragraph-like boxes found.")
        return image

    boxes = np.array(boxes)
    confidences = np.array(confidences)
    print(
        f"Paragraph-like boxes before NMS: {len(boxes)}, Confidences: {len(confidences)}"
    )
    nms_indices = non_max_suppression(boxes, probs=confidences, overlapThresh=0.3)
    print(f"NMS Indices: {nms_indices}, Number of indices: {len(nms_indices)}")

    if len(nms_indices) == 0 or np.max(nms_indices) >= len(boxes):
        print("NMS indices out of range or no indices returned.")
        return image

    final_boxes = boxes[nms_indices]
    enlarged_boxes = enlarge_boxes(final_boxes, image.shape, factor=1.1)

    for x1, y1, x2, y2 in enlarged_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=cv2.FILLED)

    return image


def enlarge_boxes(boxes, image_shape, factor=1.1):
    enlarged_boxes = []
    image_height, image_width = image_shape[:2]

    for x1, y1, x2, y2 in boxes:
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width // 2
        center_y = y1 + height // 2

        new_width = int(width * factor)
        new_height = int(height * factor)

        new_x1 = max(center_x - new_width // 2, 0)
        new_y1 = max(center_y - new_height // 2, 0)
        new_x2 = min(center_x + new_width // 2, image_width)
        new_y2 = min(center_y + new_height // 2, image_height)

        enlarged_boxes.append((new_x1, new_y1, new_x2, new_y2))

    return enlarged_boxes


def extract_images(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = filters.sobel(gray)
    edges_normalized = (edges * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        edges_normalized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    extracted_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:
            extracted_images.append(image[y : y + h, x : x + w])

    return extracted_images


def is_image_of_text(image, ocr_reader, text_threshold=50):
    results = ocr_reader.readtext(image)
    text_area = sum(
        (br[0] - tl[0]) * (br[1] - tl[1]) for (tl, tr, br, bl), text, prob in results
    )
    image_area = image.shape[0] * image.shape[1]
    text_density = (text_area / image_area) * 100
    return text_density > text_threshold


def process_pdf(pdf_path, output_dir):
    images = convert_from_path(pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    for page_num, image in enumerate(images, start=1):
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        text_removed_image = remove_text_with_easyocr(open_cv_image)
        extracted_images = extract_images(text_removed_image)

        for i, img in enumerate(extracted_images, start=1):
            if is_image_of_text(img, reader):
                print(f"Skipping text image on page {page_num}, image {i}")
                continue

            output_image_path = os.path.join(
                output_dir, f"page_{page_num}_image_{i}.png"
            )
            cv2.imwrite(output_image_path, img)
            print(
                f"Saved extracted image for page {page_num}, image {i} at {output_image_path}"
            )


if __name__ == "__main__":
    path = os.getcwd()
    inputdir = os.path.join(path, "sample_papers")
    outPut_dir = os.path.join(path, "output")

    os.makedirs(outPut_dir, exist_ok=True)
    files = [f for f in os.listdir(inputdir) if f.endswith(".pdf")]

    for file in files:
        fitem = os.path.join(inputdir, file)
        output_subdir = os.path.join(outPut_dir, os.path.splitext(file)[0])
        process_pdf(fitem, output_subdir)