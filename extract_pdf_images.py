import fitz #from PyMuPDF
from pathlib import Path
import re
import io
import os.path
import glob
from PIL import Image


#function goes through each page of pdf to check for images and generates png based on what it identifies

def extract_images(pdf):

    print(pdf)

    pdf_file = fitz.open(pdf)

    try:
        
        for page_index in range(len(pdf)): 
            page = pdf_file[page_index]
            image_list = page.get_images()
    
            if image_list:
                print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
            else:
                print("[!] No images found on page", page_index)
            for image_index, img in enumerate(page.get_images(), start=1):
                # get the XREF of the image
                xref = img[0]
          
                # extract the image bytes
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]
          
                # get the image extension
                image_ext = base_image["ext"]
        
                image = Image.open(io.BytesIO(image_bytes))
                image.save(open(f"{pdf}_page{page_index+1}_image{image_index}.{image_ext}", "wb"))
        
    except IndexError:
        pass


extract_images('sample_papers/Leifson.pdf')