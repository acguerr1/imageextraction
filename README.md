<img src="https://github.com/acguerr1/PicAxe/blob/goodies/picaxe_logo.png?raw=true" alt="PicAxe_logo" width="500"/>

# PicAxe

## 1. What is PicAxe?

**PicAxe** is open source research software (developed and tested with Python 3.10.0) that automatically extracts figures (diagrams, graphs, photographs, and tables) from PDFs that contain text and images. We (researchers at the Santa Fe Institute, the University of Chicago, and Arizona State University) are developing PicAxe to perform well on syntactically heterogenous corpora, meaning that PicAxe should perform accruate figure extraction regardless of differences in PDF layout, content, and how PDFs were produced (scanned physical material vs. "born-digital" PDFs). PicAxe is especially useful for researchers who want to extract images that are not embedded seperately from text in PDF syntax.

PicAxe identifies figures within text-image PDF files and returns the figures as PNG files. Users may choose between two PicAxe pipelines, **PicAxe-YOLO** and **PicAxe-OCR**. The process by which each pipeline identifies figures is different:

a. **PicAxe-YOLO** performs figure recognition and extraction with two pretrained YOLOv8 (Ultralytics, 2024)[^1] models. PicAxe-YOLO has three operational modes: figure-sensitive, table-sensitive, and combined. These modes were added because tables were a primary cause of extraction error. Users can select any of these modes to optimize extraction results, depending on whether their focus is on non-tabular figures, tables, or both.

b. **PicAxe-OCR** removes tables using Layout Parser (Shen et al., 2021)[^2], identifies and eliminates text using Paddle-OCR (PaddlePaddle Community, 2024)[^3], and finally extracts the remaining content. PicAxe-OCR was developed to investigate if accurate unsupervised figure extraction was possible without image training data beyond pretrained optical charcter recognition. PicAxe-OCR takes longer to run on the same sized corpus as PicAxe-YOLO. For corpora where PicAxe-YOLO produces inaccurate results, researchers might try PicAxe-OCR as it does not rely on pretrained image data.

c. The "old_version" branch contains a preliminary version of PicAxe-OCR that uses pytesseract for text removal and does not perform border removal from scanned PDFs. While it will run much faster than PicAxe-OCR (Paddle), the exrtaction results will be far less accruate.

   
## 2. What PDF features cause extraction performance errors and are you doing anything to address them?
   
During testing, we identified six major PDF features that will cause performance errors:

a. If your PDFs were produced via scanning (typically PDFs of older documents), and the scan quality is poor (low resolution, low contrast, scanner abberations, tilted content), PicAxe may not perform well. While PicAxe does apply some pre-processing steps to improve image extraction, users should always check extraction results against original PDFs to ensure PicAxe did not miss images in the case of poor scan quality. We are testing different pre-processing steps to improve overall extraction.

b. If your PDFs were produced via scanning (typically PDFs of older documents) and there are dark scan borders, PicAxe will currently identify borders as images and the entire page will be extracted as a result. Both PicAxe pipelines have an automatic border removal feature, but for optimal results users should remove borders with cropping tools before applying PicAxe. 

c. If your PDFs include long vertical or horizontal lines as page organizers (typically PDFs of newer, "born digital" documents), PicAxe will identify those lines as images and extract other images along with those lines. We are working to improve line detection as a feature of PicAxe, for optimal extraction, users should remove these lines before applying PicAxe. Both PicAxe pipelines have a table removal function, but it is a known issue that if two tables are adjacent to one other on a single page, PicAxe will likely miss one of the two tables.

d. If your PDF includes mathematical equations or uses text symbols (periods, dashes) for page organization, PicAxe will likely identify these symbols as images. Users should remove these false positive extraction results before further analysis on extracted images.

f. PicAxe will extract non-textual PDF features like library logos and barcodes. Users should remove these false positive extraction results before further analysis on extracted images.
       
e. PicAxe may extract sets of marks in ways that users do not want depending on how marks are combined on a page. If multiple figures on a single PDF page are spaced close together, PicAxe may extract those figures as a single image; depending on your research needs, you might need to manually seperate those figures that PicAxe extracted as a single image. If multiple marks in a single figure are spaced far apart, PicAxe may extract those marks separately; depending on your research needs, you might need to go extract that single figure manually to ensure that all marks are together.

## 3. How to cite PicAxe: 

If you use, analyze, or refer to PicAxe in your research, please cite us using the citation file (.cff), or with : Guerrero, A. C., Kamath, K., Zhou, Q., Felalaga, B., Damerow, J., and Dinner, A. R. (2025). PicAxe. DOI: 10.5281/zenodo.14873182. URL: https://github.com/acguerr1/PicAxe. 

## 4. How to report bugs/issues: 

Please report any code bugs/issues to Anna via email at acg(at)santafe(dot)edu with the subject line "PicAxe Issue" and a detailed description of the bug/issue.

## 5. How to contribute:

If you would like to contribute thoughts or code, please send an email to to Anna at acg(at)santafe(dot)edu with the subject line "PicAxe Contribute" and with a detailed description of who you are and how you would like to contribute.

## On Docker images:

To ease dependency issues, we have provided Docker files for each pipeline. If the corresponding Docker image is not pulled for some amount of time, it will be automatically deleted. Note that the Docker image might not exist at some point. 


[^1]: Ultralytics. YOLOv8. 2024. [https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md).
[^2]: Shen, Zejiang, Ruochen Zhang, Melissa Dell, Benjamin Charles Germain Lee, Jacob Carlson, and Weining Li. 2021. LayoutParser: A Unified Toolkit for Deep Learning Based Document Image Analysis. arXiv preprint. [https://doi.org/10.48550/arXiv.2103.15348](https://doi.org/10.48550/arXiv.2103.15348).
[^3]: PaddlePaddle Community. 2024. PaddleOCR. [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
