# imageextraction

ALWAYS CHECK EXTRACTION OUTPUT BEFORE PERFORMING FURTHER ANALYSIS.

Versions used: 
Python 3.11.3
pip 23.2.1

Installation steps:
1. Ensure you have python3 and pip installed. 
2. Install the requirements using `pip install -r requirements.txt`
3. Once installation is successful, follow the next steps to run the program
4. If working on windows, please ensure you add the following line to extract_images.py `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'`. The path should be where you have installed tesseract-OCR on your windows machine. 

Steps to extract images from PDF:
1. Add pdf to sample_papers folder
2. run `python3 pdf_image_extractor.py --file filename`. Note filename should be with extension (eg. --file Leifson.pdf)
3. The images will be present in final_output folder


To perform each step individually (for debugging purposes):
3. On command line, run `python3 pdf_to_image_converter.py --file filename`
4. Once complete, run `python3 extract_images.py -h` to find which arguments can be provided. (eg `python3 extract_images.py --debug --single_page 2` ). Note that --debug doesn't need a value, just --debug needs to be added to enable debugging mode. 




PicAxe FAQ: 

1. What is PicAxe?
     PicAxe is a free and open source software that automatically extracts images (diagrams, graphs, photographs, and some tables) from PDFs that contain text and images. We (researchers at the Santa Fe Institute, the University of Chicago, and Arizona State University) are developing PicAxe to perform well on heterogenous corpora, meaning that PicAxe should perform well regardless of differences in PDF layout, content, and how PDFs were produced.
   
2. How does PicAxe work?
     PicAxe accepts an input of PDFs, converts individual PDF pages to binary PNGs, uses pytesseract's OCR capabilities to identify and eliminate text, applies OTSU thresholding and performs contour detection on the remaining marks, eliminates odd contours and uses dialtion to combine close contours, extracts content in the final bounding boxes from original PDF and returns that content as extracted PNGs. If PicAxe performed well, extracted PNGs should only contain images found on the pages of the original PDF.
   
3. What PDF features cause performance errors and are you doing anything to address them?
   During testing, we identified six major PDF features that will cause performance errors:
    a. If your PDFs were produced via scanning (typically PDFs of older documents), and the scan quality is poor (low resolution, low contrast, scanner abberations, tilted content), PicAxe may not perform well. While PicAxe does apply some pre-processing steps to improve image extraction, users should always check extraction results against original PDFs to ensure PicAxe did not miss images in the case of poor scan quality. We are testing different pre-processing steps to improve overall extraction.
    b. If your PDFs were produced via scanning (typically PDFs of older documents) and there are dark scan borders, PicAxe will currently identify borders as images and the entire page will be extracted as a result. We are working to improve automatic border removal as a feature of PicAxe, but as of August 2024, users should ideally remove these borders with cropping tools before applying PicAxe. 
    c. If your PDFs include long vertical or horizontal lines as page organizers (typically PDFs of newer, "born digital" documents), PicAxe will identify those lines as images and extract other images along with those lines. We are working to improve line detection as a feature of PicAxe, but as of August 2024, users should ideally remove these lines with cropping tools before applying PicAxe. PicAxe will also have difficulty exctracting some tables depending on how lines are used to format the tables. We do not recommend relying on PicAxe as a table extractor as of August 2024. We are working to improve table extraction as a seperate step from image extraction.
    d. If your PDF includes mathematical equations or uses text symbols (periods, dashes) for page organization, PicAxe will likely identify these symbols as images. Users should remove these false positive results before further analysis on extracted images.
    f. PicAxe will extract non-textual PDF features like library logos and barcodes. Users should remove these false positive results before further analysis on extracted images.
       e. PicAxe may extract sets of marks in ways that users do not want depending on how marks are combined on a page. If multiple figures on a single PDF page are spaced close together, PicAxe may extract those figures as a single image; depending on your research needs, you might need to manually seperate those figures that PicAxe extracted as a single image. If multiple marks in a single figure are spaced far apart, PicAxe may extract those marks seperately; depending on your research needs, you might need to go extract that single figure manually to ensure that all marks are together.
