
import os
import subprocess
import sys

# Assuming you have a virtual environment in the 'venv' directory
venv_dir = 'padenv'  # Replace with your virtual environment directory name

# Determine the path to the Python interpreter in the virtual environment
if os.name == 'nt':  # Windows
    python_interpreter = os.path.join(venv_dir, 'Scripts', 'python.exe')
else:  # Linux, macOS, etc.
    python_interpreter = os.path.join(venv_dir, 'bin', 'python')

# Check if the interpreter exists
if not os.path.exists(python_interpreter):
    print(f"Error: Python interpreter not found in virtual environment at {python_interpreter}")
    sys.exit(1)

# Compare the current interpreter with the virtual environment's interpreter
current_python = sys.executable

if current_python != python_interpreter:
    print(f"Current interpreter is {current_python}. Switching to {python_interpreter}...")

    # Re-run the script with the virtual environment's interpreter
    subprocess.run([python_interpreter] + sys.argv)

    # Exit the current interpreter
    sys.exit(0)

# Example: Run a Python command using the virtual environment's interpreter
try:
    subprocess.run([python_interpreter, '--version'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Failed to run Python command: {e}")
    sys.exit(1)

print(f"Using Python interpreter from: ./{python_interpreter}")
