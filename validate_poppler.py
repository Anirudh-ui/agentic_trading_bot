import subprocess

def validate_poppler():
    try:
        result = subprocess.run(['pdftotext', '-v'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Poppler is installed and functional.")
        else:
            print("Poppler validation failed.")
    except FileNotFoundError:
        print("Poppler is not installed.")

if __name__ == "__main__":
    validate_poppler()