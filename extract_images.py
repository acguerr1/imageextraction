import cv2
import numpy as np
import pytesseract
import os
import numpy as np
import argparse
import sys
from utils import PDF_PAGES, PAGES_WO_TEXT_DIR, IMAGE_BOUNDARIES, EXTRACTED_IMAGES_DIR, BINARY_PAGES, CROPPED_PAGES
from utils import whiten_pixels

def pre_process(i, dir, dilation_iterations, image):
    if image == None:
        image_path = os.path.join(dir, f'pg{i}.png')
        image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Create rectangular structuring element and dilate

    # matrix of 1s
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

    # dilate enhances the foreground (white) , suppresses background. Needs a binary image 
    dilate = cv2.dilate(thresh, kernel, iterations=dilation_iterations)

    # Find contours
    return cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



def find_images(i, image, dilation_iterations, debug_mode, output_dir=EXTRACTED_IMAGES_DIR):
    original_image = cv2.imread(os.path.join(PDF_PAGES, f'pg{i}.png'))
    cnts = pre_process(i, PAGES_WO_TEXT_DIR, dilation_iterations, None)
    
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    output_image = image.copy()
    img_cnt = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (36,255,12), 2)
        
        # sometimes because of the low iterations of the previous cause, pixels won't be caught. In that case, we need to check for w and h again.
        if w > 500 and h > 500:
            final_image = original_image[y:y+h, x:x+w]
            output_image_path = os.path.join(output_dir, f'page{i+1}_image{img_cnt+1}.png')
            img_cnt += 1
            cv2.imwrite(output_image_path, final_image)
    if debug_mode:
        if not os.path.exists(IMAGE_BOUNDARIES):
            os.makedirs(IMAGE_BOUNDARIES)
        output_image_path = os.path.join(IMAGE_BOUNDARIES, f'pg{i}.png')
        cv2.imwrite(output_image_path, output_image)

def remove_small_noises(i, dilation_iterations):
    cnts = pre_process(i, PAGES_WO_TEXT_DIR, dilation_iterations, None)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    output_image = cv2.imread(os.path.join(PAGES_WO_TEXT_DIR, f'pg{i}.png'))
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < 500 and h < 500: 
            # remove dirty pixels
            output_image = whiten_pixels(output_image, x, y, w, h)
    return output_image

def remove_text(idx, confidence_cutoff, debug, output_dir = PAGES_WO_TEXT_DIR):

    image_path = os.path.join(BINARY_PAGES, f'pg{idx}.png')
    # image_path = os.path.join(CROPPED_PAGES, f'pg{idx}.png')
    if not os.path.exists(image_path):
        return
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
            continue
        if len(curr[-1]) == 0 or curr[-1].isspace():
            continue
        else:
            w = int(curr[-4]) + 35
            h = int(curr[-3]) + 35
            x = int(curr[-6]) - 10
            y = int(curr[-5]) - 10
            image = whiten_pixels(image, x ,y , w, h)
            if debug == True:
                cv2.imshow('image', image)
                cv2.waitKey(0)
    output_image_path = os.path.join(output_dir, f'pg{idx}.png')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(output_image_path, image)
    # print(output_image_path)

if __name__ == "__main__":
    # entry point of the file
    if not os.path.exists(BINARY_PAGES):
        print(f'Input folder {BINARY_PAGES} doesn\'t exist. Check if you have converted pdf to png files.')
    
    # Parsing command line argument
    parser = argparse.ArgumentParser(description="Extract images from pdf page.")
    parser.add_argument('--pages', type=int, default=1, help='Number of pages present in pdf.')
    parser.add_argument('--single_page',  type=int, help='If present, we parse a single page (give page number)')
    parser.add_argument('--debug', action='store_true',  help='Enable debugging mode')
    parser.add_argument('--confidence_cutoff', default=15 , type=float, help='Confidence cutoff for text detection(from 0 to 100)')
    
    args = parser.parse_args()
    
    if args.debug:
        debug_mode = True
    else:
        debug_mode = False

    if args.single_page is not None:
        # Single page mode: User wants to extract images from a single page
        print(f'Extracting images from page {args.single_page}')
        remove_text(args.single_page - 1,args.confidence_cutoff, debug_mode)
        find_images(args.single_page - 1, remove_small_noises(args.single_page - 1, 10), 35, debug_mode)
    
    else:
        for i in range(args.pages):
            print(f'Extracting images from page {i+1}')
            remove_text(i,args.confidence_cutoff, debug_mode)
            find_images(i, remove_small_noises(i, 10), 35, debug_mode)

    


