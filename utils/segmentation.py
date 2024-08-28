import cv2
import numpy as np
import tensorflow as tf

# Predict table regions in the input image using a pre-trained model.
def predict_table_regions(image, table_model):
    original_height, original_width = image.shape[:2]
    resized_image = tf.image.resize(image, [672, 480]) / 255.0
    input_tensor = tf.convert_to_tensor(resized_image, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, 0)
    prediction = table_model.predict(input_tensor)
    table_prediction = prediction[0, :, :, 1]
    return table_prediction, original_height, original_width

# Predict the layout of the document using a pre-trained segmentation model.
def predict_layout(image, layout_model):
    resized_image = tf.image.resize(image, [896, 896]) / 255.0
    input_tensor = tf.convert_to_tensor(resized_image, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, 0)
    prediction = layout_model.predict(input_tensor)
    return prediction[0]

# Extract and scale detected table regions to match the original image dimensions.
def extract_and_scale_table_regions(table_prediction, original_height, original_width, area_threshold=40000):
    table_mask = (table_prediction > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_boxes = []
    scale_x = original_width / 480
    scale_y = original_height / 672

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            table_boxes.append((x, y, w, h))

    return table_boxes

# Combine table regions with layout predictions, adjust the cropping box, and perform cropping.
def combine_predictions_and_crop(original_image, layout_prediction, table_boxes, crop_proportion_threshold=0.65, vertical_margin=10, horizontal_margin=10):
    class_map = np.argmax(layout_prediction, axis=-1)
    non_background_mask = (class_map != 0).astype(np.uint8)
    non_zero_indices = np.nonzero(non_background_mask)

    if len(non_zero_indices[0]) > 0:
        ymin, ymax = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
        xmin, xmax = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
        height_ratio = original_image.shape[0] / layout_prediction.shape[0]
        width_ratio = original_image.shape[1] / layout_prediction.shape[1]
        xmin, xmax = int(xmin * width_ratio), int(xmax * width_ratio)
        ymin, ymax = int(ymin * height_ratio), int(ymax * height_ratio)

        # Adjust crop boundaries to include table regions.
        for (tx, ty, tw, th) in table_boxes:
            xmin = min(xmin, tx)
            xmax = max(xmax, tx + tw)
            ymin = min(ymin, ty)
            ymax = max(ymax, ty + th)

        # Apply margins and crop the image.
        xmin = max(0, xmin - horizontal_margin)
        xmax = min(original_image.shape[1], xmax + horizontal_margin)
        ymin = max(0, ymin - vertical_margin)
        ymax = min(original_image.shape[0], ymax + vertical_margin)

        xmin, xmax = int(xmin), int(xmax)
        ymin, ymax = int(ymin), int(ymax)
        cropped_image = original_image[ymin:ymax, xmin:xmax, :]
        
        # Return the original image if the cropped image is too small.
        cropped_size = cropped_image.shape[0] * cropped_image.shape[1]
        original_size = original_image.shape[0] * original_image.shape[1]
        if cropped_size < crop_proportion_threshold * original_size:
            print(f"Cropped image size is less than {crop_proportion_threshold} of the original image size, returning original image.")
            return original_image, 0, 0

        return cropped_image, xmin, ymin
    else:
        # Return the original image if no region is detected.
        return original_image, 0, 0

