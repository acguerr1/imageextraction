import sys

if sys.platform.startswith('darwin'):
    print("macOS")
elif sys.platform.startswith('win'):
    print("Windows")
elif sys.platform.startswith('linux'):
    print("Linux")
else:
    print("Your operating system could not be determined")
