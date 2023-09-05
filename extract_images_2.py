import cv2
import numpy as np
import pytesseract
import os
import numpy as np
import argparse
import sys
from utils import INPUT_DIR_STEP_1, INPUT_DIR_STEP_2, INPUT_DIR_STEP_3


def whiten_pixels(image, x, y, w, h):
    whitened_image = image.copy()
    whitened_image[y:y+h, x:x+w] = (255, 255, 255)
    return whitened_image


def remove_dash(image):
    allowed_characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_-+=[]{}|;:,.<>?'
    custom_config = f'--psm 7 --oem 3 -c tessedit_char_whitelist={allowed_characters}'
    val = pytesseract.image_to_data(image, config=custom_config)
    texts = []
    area = 0
    for x in val.splitlines():
        texts.append(x.split('\t'))
     
    print(val)


    for i in range(1,len(texts)):
        curr = texts[i]
        if curr[-2] == '-1':
            continue
        if float(curr[-2]) <= 10:
            # print(curr)
            continue
        if len(curr[-1]) == 0 or curr[-1] == ' ':
            continue
            
        else:
            # textbox
            # width * height
            w = int(curr[-4]) + 5
            h = int(curr[-3]) + 5
            x = int(curr[-6])
            y = int(curr[-5])
            image = whiten_pixels(image, x ,y , w, h)
            # if debug == True:
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
    return image

def find_images(i, image, dilation_iterations, output_dir='extracted_images/'):
    image_path = os.path.join(INPUT_DIR_STEP_1, f'out{i}.png')
    original_image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Create rectangular structuring element and dilate

    # (5,5) matrix of 1s
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

    # dilate enhances the foreground (white) , suppresses background. Needs a binary image 
    dilate = cv2.dilate(thresh, kernel, iterations=dilation_iterations)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    output_image = image.copy()
    img_cnt = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (36,255,12), 2)

        # instead of bounding rectangle, save the image after reintroducing the pixels
        final_image = original_image[y:y+h, x:x+w]
        output_image_path = os.path.join(output_dir, f'output_{i}_{img_cnt}.png')
        img_cnt += 1
        cv2.imwrite(output_image_path, final_image)
    
    if not os.path.exists('image_boundaries'):
        os.makedirs('image_boundaries')
    output_image_path = os.path.join('image_boundaries', f'output{i}.png')
    if not os.path.exists(output_dir):
        os.makedirs('image_boundaries')
    cv2.imwrite(output_image_path, output_image)

def remove_small_noises(i, dilation_iterations):
    image_path = os.path.join(INPUT_DIR_STEP_2, f'output{i}.png')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Create rectangular structuring element and dilate

    # (5,5) matrix of 1s
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

    # dilate enhances the foreground (white) , suppresses background. Needs a binary image 
    dilate = cv2.dilate(thresh, kernel, iterations=dilation_iterations)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    output_image = image.copy()
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < 200 and h < 200: 
            # remove dirty pixels
            output_image = whiten_pixels(output_image, x, y, w, h)
    return output_image

def perform(idx, confidence_cutoff, debug,  output_dir = INPUT_DIR_STEP_2):
    image_path = os.path.join(INPUT_DIR_STEP_1, f'out{idx}.png')
    image = cv2.imread(image_path)
    val = pytesseract.image_to_data(image)
    texts = []
    area = 0
    for x in val.splitlines():
        texts.append(x.split('\t'))
    for i in range(1,len(texts)):
        curr = texts[i]
        if debug == True:
            print('text is ', curr[-1])
            print('confidence is ', curr[-2])
            print('\n')
        if curr[-2] == '-1':
            continue
        if float(curr[-2]) <= confidence_cutoff:
            # print(curr)
            continue
        if len(curr[-1]) == 0 or curr[-1] == ' ':
            continue
            
        else:
            # textbox
            # hotfix for underline. improve
            w = int(curr[-4]) + 35
            h = int(curr[-3]) + 35
            x = int(curr[-6]) - 10
            y = int(curr[-5]) - 10
            image = whiten_pixels(image, x ,y , w, h)
            if debug == True:
                cv2.imshow('image', image)
                cv2.waitKey(0)
    output_image_path = os.path.join(output_dir, f'output{idx}.png')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(output_image_path, image)

def whiten_pixels(image, x, y, w, h):
    whitened_image = image.copy()
    whitened_image[y:y+h, x:x+w] = (255, 255, 255)
    return whitened_image


    
if __name__ == "__main__":
    # entry point of the file
    if not os.path.exists(INPUT_DIR_STEP_1):
        print("Input folder temp/ doesn't exist. Check if you have converted pdf to png files.")
    


    parser = argparse.ArgumentParser(description="Extract images from pdf page.")
    parser.add_argument('--pages', type=int, default=1, help='Number of pages present in temp.')
    parser.add_argument('--single_page',  type=int, help='If present, we parse a single page (give page number)')
    parser.add_argument('--debug', action='store_true',  help='Enable debugging mode')
    parser.add_argument('--confidence_cutoff', default=15 , type=float, help='Confidence cutoff for text detection(from 0 to 100)')
    
    args = parser.parse_args()
    if args.debug:
        debug = True
    else:
        debug = False
    if args.single_page is not None:

        perform(args.single_page - 1,args.confidence_cutoff, debug)
        # find_images(i, 0.3, 32, 10)
        find_images(args.single_page - 1, remove_small_noises(args.single_page - 1, 10), 35)
        sys.exit(0)
    else:
        for i in range(args.pages):
            print(i)
            perform(i,args.confidence_cutoff, args.debug)
            find_images(i, remove_small_noises(i, 10), 35)
            # find_images(i, 0.3, 32, 10)

    # once we have the output file, find the images.
    


