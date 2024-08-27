# main.py
import os
import sys
import time
import subprocess

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

# Import the scripts_dir from the config file
from config import scripts_dir

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

# Main execution block
if __name__ == '__main__':
    start = time.time()  # Record the start time

    # List of scripts to run sequentially
    scripts = ["convert_pdfs_to_images.py", "crop_borders.py", "remove_tables.py", \
               "remove_text.py", "select_target_images.py", "extract_and_save_images.py"]

    # Run each script
    for script in scripts:
        run_script(script)

    print("\nExtraction completed successfully.")
    end = time.time()  # Record the end time
    print(f"\nRuntime: {(end - start) / 60:.2f} minutes")  # Print the total runtime
