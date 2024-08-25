# box_merging.py
import cv2
import numpy as np

def merge_boxes(boxes, dilation_iter=20):
    if not boxes:
        return []

    # Create a blank image to draw the bounding boxes
    max_x = max([box[2] for box in boxes])
    max_y = max([box[3] for box in boxes])
    blank_image = np.zeros((max_y, max_x), dtype=np.uint8)

    # Draw the bounding boxes on the blank image
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(blank_image, (x1, y1), (x2, y2), 255, -1)

    # Dilate the image to merge nearby boxes
    dilated_image = cv2.dilate(blank_image, None, iterations=dilation_iter)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    merged_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Increase the box size by 20 pixels up and down, 20 pixels left and right
        x = max(x - 20, 0)
        y = max(y - 20, 0)
        x2 = min(x + w + 40, max_x)  
        y2 = min(y + h + 40, max_y)  

        merged_boxes.append((x, y, x2, y2))

    return merged_boxes
