import cv2
import numpy as np
import pytesseract
import os
import numpy as np
import argparse
import sys


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

def perform(idx, confidence_cutoff, debug,  output_dir = 'output'):
    image_path = os.path.join('temp', f'out{idx}.png')
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
            # width * height
            # hotfix for underline. improve
            w = int(curr[-4]) + 35
            h = int(curr[-3]) + 35
            x = int(curr[-6]) - 15
            y = int(curr[-5]) - 15
            if curr[-1] == "P=27T/R" :
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
                cv2.imshow('image', image)
                cv2.waitKey(0)
            image = whiten_pixels(image, x ,y , w, h)
            if debug == True:
                cv2.imshow('image', image)
                cv2.waitKey(0)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # image = remove_dash(image)
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
    if not os.path.exists('temp'):
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
        sys.exit(0)
    else:
        for i in range(args.pages):
            print(i)
            perform(i,args.confidence_cutoff, args.debug)


