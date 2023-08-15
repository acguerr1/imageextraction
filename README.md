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
2. On command line, run `python3 pdf_to_image_converter.py --file Ketchem.pdf`
3. Once complete, run `python3 extract_images.py -h` to find which arguments can be provided. eg. `python3 extract_images.py --single_page 3 --dilation_iterations 10 --area_filter 0.3`
4. For new extraction method, run `python3 extract_images_2.py -h` to find which arguments can be provided.