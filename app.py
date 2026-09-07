import os
import subprocess
import sys

# Ensure some necessary system packages are present (optional, sometimes helps on spaces)
# os.system("apt-get update && apt-get install -y libgl1 libglib2.0-0")

from src.main import main

if __name__ == "__main__":
    main()
