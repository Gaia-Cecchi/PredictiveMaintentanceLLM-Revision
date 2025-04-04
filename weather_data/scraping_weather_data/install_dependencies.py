import subprocess
import sys
import os

def install_requirements():
    print("Installing required packages...")
    
    # Define the requirements with specific versions to ensure compatibility
    requirements = [
        "selenium==4.15.2",
        "webdriver-manager==4.0.1",
        "pandas"
    ]
    
    # Install each requirement individually
    for requirement in requirements:
        print(f"Installing {requirement}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            print(f"Successfully installed {requirement}")
        except Exception as e:
            print(f"Failed to install {requirement}: {e}")
            return False
    
    print("All dependencies installed successfully!")
    return True

if __name__ == "__main__":
    install_requirements()
