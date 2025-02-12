# main.py
import os
import sys
import time
import shutil
import argparse
import tempfile
import subprocess

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

# Import the scripts_dir from the config file
from config import config
scripts_dir = config.scripts_dir
sample_papers_dir = config.sample_papers_dir
bulk_papers_dir = config.bulk_papers_dir
pdf_files = config.pdf_files
output_tables_dir = config.output_tables_dir
extracted_images = config.extracted_images

def copy_tables_to_output(output_tables_dir, extracted_images):
    # Ensure the target directory exists
    os.makedirs(extracted_images, exist_ok=True)

    # Copy files from output_tables_dir to extracted_images
    for filename in os.listdir(output_tables_dir):
        src_file = os.path.join(output_tables_dir, filename)
        dest_file = os.path.join(extracted_images, filename)
        
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dest_file)  # Copy the file to the target directory
    return None

# Function to run an individual script
def run_script(script_name):
    try:
        script_path = os.path.join(scripts_dir, script_name)
        
        # Ensure the script has the .py extension
        if not script_path.endswith('.py'):
            script_path += '.py'
        
        # Run the script
        subprocess.run([sys.executable, script_path], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")
        sys.exit(1)  # Exit if the script fails

def main(args):
    temp_dir = None
    try: 
        # Set the location of pdf_files
        if args.bulk:
            config.pdf_files = bulk_papers_dir
        elif args.sample:
            config.pdf_files = sample_papers_dir
        elif args.file_name:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Copy the specified file to the temporary directory
            file_path = os.path.join(sample_papers_dir, os.path.basename(args.file_name))
            if not os.path.exists(file_path):
                print(f"Error: The file {file_path} does not exist.")
                sys.exit(1)
            
            shutil.copy(file_path, temp_dir)
            config.pdf_files = temp_dir
        else:
            
            print("Error: Either --file, --bulk, or --sample must be provided.")
            sys.exit(1)
        
        # Add the pdf_files to the environment variables so that scripts can access it
        os.environ['PDF_FILES'] = config.pdf_files
       
        start = time.time()  # Record the start time

        # List of scripts to run sequentially
        scripts = ["convert_pdfs_to_images.py", "crop_borders.py", "remove_tables.py", \
                    "remove_text.py", "select_target_images.py", "extract_and_save_images.py"]
        
        # Run each script
        for script in scripts:
            run_script(script)
        
        # Add tables to extracted images
        copy_tables_to_output(output_tables_dir, extracted_images)

    except Exception as e:
        print(f"Error encounterd: {str(e)}")
        sys.exit(1)

    finally:
        # Clean up the temporary directory if it was created
        if temp_dir:
            shutil.rmtree(temp_dir)

    print("\nExtraction completed successfully.")
    end = time.time()  # Record the end time
    print(f"\nRuntime: {(end - start) / 60:.2f} minutes")  # Print the total runtime
    return None

# Main execution block
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Image Extraction of pdfs")
    parser.add_argument('--file', dest='file_name', required=False, help='The name of the PDF file to convert.')
    parser.add_argument('--bulk', action='store_true',  help='Enable bulk processing mode')
    parser.add_argument('--sample', action='store_true',  help='Enable sample papers only for processing')
    args = parser.parse_args()


    pdf_files = pdf_files # commented out else in main for debuggin
    main(args)
    


