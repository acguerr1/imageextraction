import sys
import os
import subprocess

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

from config import project_dir

# Part 1: Install packages
try:
    subprocess.run(['pip', 'install', '-r', f'{project_dir}/requirements.txt'], check=True)
except subprocess.CalledProcessError:
    print("Failed to install packages from requirements.txt")

# try:
#     subprocess.run(['brew', 'install', 'poppler'], check=True)
# except subprocess.CalledProcessError:
#     print("Failed to install Poppler (pdfinfo) using Homebrew")



# Part 2: Print list of packages installed and their versions

# Print header
print(f"\nInstallation Complete. \nList of Packages installed: \n")

# List of packages to check
my_pkgs = ['layoutparser', 'paddlepaddle', 'scipy', 'layoutparser', 'rembg', 'PyMuPDF', 'matplotlib', 
           'pdf2image', 'opencv-python', 'numpy', 'pillow', 'paddleocr', 'paddlepaddle', 'PyMuPDF']

# Check and print the version of each package
for pkg in my_pkgs:
    try:
        result = subprocess.run([
            'python', '-c',
            f'import importlib.metadata; '
            f'version = importlib.metadata.version("{pkg}"); '
            f'print("{pkg} version: " + version)'
        ], check=True, capture_output=True, text=True)

        print(result.stdout.strip())

    except subprocess.CalledProcessError:
        print(f"{pkg} version not found or package not installed.")

# Check and print Poppler version
try:
    subprocess.run(['pdfinfo', '-v'], check=True)
except subprocess.CalledProcessError:
    print("Poppler (pdfinfo) not found or not installed.")


