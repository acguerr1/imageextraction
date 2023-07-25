import cv2
import numpy as np
import pytesseract
# Load image, grayscale, Gaussian blur, 

import numpy as np



# doesn't work. box_info[-1] is page number, not confidence

# def filter_boxes_by_confidence(boxes_str, confidence_threshold):
#     filtered_boxes = []
#     for box_line in boxes_str.splitlines():
#         box_info = box_line.split()
#         confidence = int(box_info[-1]) if box_info[-1].isdigit() else None
#         if confidence is not None and confidence >= confidence_threshold:
#             filtered_boxes.append(box_line)
#     return '\n'.join(filtered_boxes)

def draw_paragraph_box(image):
    # Load the image and convert to grayscale
    # image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7,7), 0)

    # Threshold the image to create a binary image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create a rectangular structuring element and dilate the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(binary, kernel, iterations=10)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box for the entire paragraph
    x, y, w, h = image.shape[1], image.shape[0], 0, 0
    area = 0
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        
        if x == 0 and y == 0 and w == image.shape[1] and h == image.shape[0]:
            # we don't draw over the whole image
            continue
        # Draw bounding box around the paragraph
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        area += w*h
    print(area)
    height, width, _ = image.shape
    print(height*width)
    print(area/(height*width))
    factor = 0.5
    if area > factor * height * width:
        return True
    return False
    # return image

def whiten_pixels(image, x, y, w, h):
    whitened_image = image.copy()
    whitened_image[y:y+h, x:x+w] = (255, 255, 255)
    return whitened_image



def get_horizontal_profile(image, bounding_box=None):
    if bounding_box is None:
        bounding_box = (0, image.shape[0] - 1)

    start_y, end_y = bounding_box
    profile = np.sum(image[start_y:end_y + 1, :], axis=1)
    bounding_boxes = [(start_y, end_y)]
    return profile, bounding_boxes

def get_vertical_profile(image, bounding_box=None):
    if bounding_box is None:
        bounding_box = (0, image.shape[1] - 1)

    start_x, end_x = bounding_box
    profile = np.sum(image[:, start_x:end_x + 1], axis=0)
    bounding_boxes = [(start_x, end_x)]
    return profile, bounding_boxes


image = cv2.imread('out2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# Create rectangular structuring element and dilate

# (5,5) matrix of 1s
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# dilate enhances the foreground (white) , suppresses background. Needs a binary image 
dilate = cv2.dilate(thresh, kernel, iterations=20)

# Find contours and draw rectangle
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
output_image = image.copy()
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    cropped_image = image[y:y+h, x:x+w]
    extracted_text = pytesseract.image_to_string(cropped_image)

    # for the cropped one, try img_to_boxes
    # if not (extracted_text and not extracted_text.isspace()):
    #     continue
    # boxes = pytesseract.image_to_boxes(cropped_image) 
    # # boxes = filter_boxes_by_confidence(boxes, confidence_threshold=50)
    # # print(boxes)
    # paragraphs = []
    # for b in boxes.splitlines():
    #     b = b.split(' ')
    #     x_min, y_min, x_max, y_max = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
    #     # Calculate smaller bounding box dimensions (adjust the padding as needed)
    #     padding = 0
    #     x_min_s, x_min_s, x_max_s, y_max_s = x_min + padding, y_min + padding, x_max - padding, y_max - padding

    #     # Draw a smaller bounding box around the recognized word
    #     # cv2.rectangle(cropped_image, (x_min_s, y_min_s), (x_max_s, y_max_s), (0, 255, 0), 2)
    # if paragraphs:
    #     x_min_all = min(x_min for x_min, _, _, _ in paragraphs)
    #     y_min_all = min(y_min for _, y_min, _, _ in paragraphs)
    #     x_max_all = max(x_max for _, _, x_max, _ in paragraphs)
    #     y_max_all = max(y_max for _, _, _, y_max in paragraphs)
    #     cv2.rectangle(cropped_image, (x_min_all, y_min_all), (x_max_all, y_max_all), (255, 0, 255), 2)
    # cv2.imshow('image', cropped_image)

    # cv2.waitKey()
    if  (extracted_text and not extracted_text.isspace()):

        # using image_to_data(), remove any blocks where confidence of text is less than 80%. There might be spaces which it identifies with a high level, but if there isn't any proper text, then skip this.


        text_x, text_y, text_w, text_h = x, y, w, h
        # cv2.rectangle(cropped_image, (text_x, text_y), (text_x + text_w, text_y + text_h), (0, 0, 255), 2)
        boxes = pytesseract.image_to_boxes(cropped_image)
        for b in boxes.splitlines():
            b = b.split(' ')
            # new_image = cv2.rectangle(cropped_image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 0, 255), 2)
        if draw_paragraph_box(cropped_image):
            # whiten the pixels
            output_image = whiten_pixels(output_image, x,y,w,h)
            # cv2.imshow('image', output_image)
            # cv2.waitKey()
cv2.imshow('image', output_image)
cv2.waitKey()
# boxes = pytesseract.image_to_boxes(image) 
# for b in boxes.splitlines():
#     b = b.split(' ')
#     image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

# cv2.imshow('image', image)
