import cv2
import numpy as np
import pytesseract
import os
import numpy as np
import argparse
import sys

def is_cropped_image_text(image, area_filter, text_iterations):
    # Load the image and convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7,7), 0)

    # Threshold the image to create a binary image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create a rectangular structuring element and dilate the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(binary, kernel, iterations=text_iterations)

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
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        
        area += w*h
    height, width, _ = image.shape

    # can help in debugging
    factor = area_filter
    # print(pytesseract.image_to_data(image))
    # print((area/(height*width)) * 100)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    if area > factor * height * width:
        # textbox
        return True
    return False

def whiten_pixels(image, x, y, w, h):
    whitened_image = image.copy()
    whitened_image[y:y+h, x:x+w] = (255, 255, 255)
    return whitened_image


def perform(i, area_filter, dilation_iterations, text_iterations,  output_dir = 'output'):
    image_path = os.path.join('temp', f'out{i}.png')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Create rectangular structuring element and dilate

    # (5,5) matrix of 1s
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # dilate enhances the foreground (white) , suppresses background. Needs a binary image 
    dilate = cv2.dilate(thresh, kernel, iterations=dilation_iterations)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    output_image = image.copy()
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        
        # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        cropped_image = image[y:y+h, x:x+w]

        extracted_text = pytesseract.image_to_string(cropped_image)
       
        # allowed_characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_-+=[]{}|;:,.<>?'
        # custom_config = f'--psm 4 --oem 3 -c tessedit_char_whitelist={allowed_characters}'

        # extracted_text = pytesseract.image_to_string(cropped_image, config=custom_config)
        # cv2.imshow('image', cropped_image)
        # cv2.waitKey(0)
        # print(extracted_text)
        if  (extracted_text and not extracted_text.isspace()):

            # using image_to_data(), remove any blocks where confidence of text is less than 80%. There might be spaces which it identifies with a high level, but if there isn't any proper text, then skip this.
            if is_cropped_image_text(cropped_image, area_filter, text_iterations):
                # whiten the pixels
                output_image = whiten_pixels(output_image, x,y,w,h)
        # else:
        #     height, width, _ = cropped_image.shape
        #     # print(height * width)

        #     # remove small noises. 
        #     # Try to find a better way to do this 
        #     if height * width <= 5000: 
        #         output_image = whiten_pixels(output_image, x,y,w,h)
        # else:
        #     cv2.imshow('image', cropped_image)
        #     cv2.waitKey(0)
        #     print(extracted_text)
        #     print(pytesseract.image_to_data(cropped_image))
   
    output_image_path = os.path.join(output_dir, f'output{i}.png')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    cv2.imwrite(output_image_path, output_image)

if __name__ == "__main__":
    # entry point of the file
    if not os.path.exists('temp'):
        print("Input folder temp/ doesn't exist. Check if you have converted pdf to png files.")
    

    # TODO: add an arg which keeps or deletes temp/ folder. Will be useful when batching multiple
    # TODO: add an arg which can modify the kernel size of both step 
    # TODO: add a debug option which can be used to help in debugging what is happening in each step

    parser = argparse.ArgumentParser(description="Extract images from pdf page.")
    parser.add_argument('--pages', type=int, default=1, help='Number of pages present in temp.')
    parser.add_argument('--single_page',  type=int, help='If present, we parse a single page (give page number)')
    parser.add_argument('--area_filter',default=0.5, type=float, help='Ratio of area of text to ')
    parser.add_argument('--dilation_iterations',default=20, type=int, help='Number of iterations for dilations (to crop images)')
    parser.add_argument('--text_iterations',default=10, type=int, help='Number of iterations for identifying text and images in cropped image')
    args = parser.parse_args()

    
    if args.single_page is not None:
        perform(args.single_page - 1,  args.area_filter, args.dilation_iterations, args.text_iterations)
        sys.exit(0)
    else:
        for i in range(args.pages):
            perform(i, args.area_filter, args.dilation_iterations, args.text_iterations)
