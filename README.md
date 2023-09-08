# imageextraction

Versions used: 
Python 3.11.3
pip 23.2.1

Installation steps:
1. Ensure you have python3 and pip installed. 
2. Install the requirements using `pip install -r requirements.txt`
3. Once installation is successful, follow the next steps to run the program

Steps to extract images from PDF:
1. Add pdf to sample_papers folder
2. run `python3 pdf_image_extractor.py --file filename`. Note filename should be with extension (eg. --file Leifson.pdf)
3. The images will be present in final_output folder


To perform each step individually (for debugging purposes):
3. On command line, run `python3 pdf_to_image_converter.py --file filename`
4. Once complete, run `python3 extract_images.py -h` to find which arguments can be provided. (eg `python3 extract_images.py --debug --single_page 2` ). Note that --debug doesn't need a value, just --debug needs to be added to enable debugging mode. 