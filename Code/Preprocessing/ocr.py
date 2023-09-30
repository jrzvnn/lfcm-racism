import os
import pytesseract
from PIL import Image

# Input directory containing images
input_directory = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Raw_Data/images"

# Output directory to store OCR results
output_directory = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/ocr_output"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Function to perform OCR on an image and save the result
def perform_ocr(input_path, output_path):
    try:
        image = Image.open(input_path)
        text = pytesseract.image_to_string(image)
        
        # Create the output directory structure
        output_subdirectory = os.path.dirname(output_path)
        if not os.path.exists(output_subdirectory):
            os.makedirs(output_subdirectory)
        
        with open(output_path, "w") as f:
            f.write(text)
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

# Recursively process images in the input directory and its subdirectories
for root, _, files in os.walk(input_directory):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_directory)
            output_path = os.path.join(output_directory, relative_path + ".txt")
            perform_ocr(input_path, output_path)
            print(f"Processed: {input_path}")

print("OCR processing completed.")
