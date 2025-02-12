import os
import sys
from pdf2image import convert_from_path
from PIL import Image

def pdf_split_middle_and_save(input_dir, output_dir, shift_middle_percentage=0, aspect_ratio_threshold=1.2, dpi=150):
    """
    Splits pages in book-like PDFs found in the input directory, saves the split pages into a new PDF file
    in the output directory.
    
    :param input_dir: Directory containing input PDF files.
    :param output_dir: Directory where the split PDFs will be saved.
    :param shift_middle_percentage: Adjusts the middle split position.
    :param aspect_ratio_threshold: Aspect ratio above which the page is considered for splitting.
    :param dpi: The DPI to use for rendering the PDF pages.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for pdf_file in os.listdir(input_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, pdf_file)
            images = convert_from_path(pdf_path, dpi=dpi)

            split_images = []

            for page_nr, img in enumerate(images):
                width, height = img.size
                aspect_ratio = width / height

                if aspect_ratio > aspect_ratio_threshold:
                    if shift_middle_percentage > 49:
                        shift_middle_percentage = 49
                    elif shift_middle_percentage < -49:
                        shift_middle_percentage = -49

                    shift_middle = width / 100 * shift_middle_percentage

                    # Split the image into left and right parts
                    left_img = img.crop((0, 0, width // 2 + int(shift_middle), height))
                    right_img = img.crop((width // 2 + int(shift_middle), 0, width, height))

                    # Append the split images to the list
                    split_images.append(left_img)
                    split_images.append(right_img)
                else:
                    # If the aspect ratio does not exceed the threshold, keep the page as is
                    split_images.append(img)

            # Save the split images as a new PDF file
            output_pdf_path = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_split.pdf")
            split_images[0].save(output_pdf_path, save_all=True, append_images=split_images[1:], format="PDF")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python split_pdf.py <input_dir> <output_dir> [shift_middle_percentage] [aspect_ratio_threshold] [dpi]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    shift_middle_percentage = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    aspect_ratio_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 1.2
    dpi = int(sys.argv[5]) if len(sys.argv) > 5 else 150

    pdf_split_middle_and_save(input_dir, output_dir, shift_middle_percentage, aspect_ratio_threshold, dpi)
