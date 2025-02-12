import sys
import os
import subprocess
import shutil

# Add the src directory to the Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

from config import config 
project_dir = config.project_dir 


def install_model():
    model_dir = os.path.expanduser('~/.cache/layoutparser/model_zoo/PubLayNet')
    os.makedirs(model_dir, exist_ok=True)

    required_files = [
        'inference.pdiparams',
        'inference.pdiparams.info', 
        'inference.pdmodel'
    ]

    model_url = 'https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar'
    model_path = os.path.join(model_dir, 'ppyolov2_r50vd_dcn_365e_publaynet.tar')

    files_exist = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)

    
    if not files_exist:
        print("\nDownloading LayoutParser model...")
        subprocess.run(['curl', '-L', model_url, '-o', model_path], check=True)
        subprocess.run(['tar', '-xf', model_path], cwd=model_dir, check=True)
        
        # Move files from subdirectory to model_dir
        extract_dir = os.path.join(model_dir, 'ppyolov2_r50vd_dcn_365e_publaynet')
        for file in os.listdir(extract_dir):
            shutil.move(os.path.join(extract_dir, file), model_dir)
        os.rmdir(extract_dir)
        
        if not all(os.path.exists(os.path.join(model_dir, f)) for f in required_files):
            print("Missing files:", [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))])
            raise RuntimeError("Model extraction failed - required files missing")
        


# Part 1: Install packages
try:
    subprocess.run(['pip', 'install', '-r', f'{project_dir}/requirements.txt'], check=True)
    install_model()  # Install models after packages are ready
except subprocess.CalledProcessError:
    print("Failed to install packages from requirements.txt")


# Part 2: Print list of packages installed and their versions
print(f"\nInstallation Complete. \nList of Packages installed: \n")

my_pkgs = ['layoutparser', 'paddlepaddle', 'scipy', 'layoutparser', 'rembg', 'PyMuPDF', 'matplotlib', 
           'pdf2image', 'opencv-python', 'numpy', 'pillow', 'paddleocr', 'paddlepaddle', 'PyMuPDF']

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

try:
    subprocess.run(['pdfinfo', '-v'], check=True)
except subprocess.CalledProcessError:
    print("Poppler (pdfinfo) not found or not installed.")
