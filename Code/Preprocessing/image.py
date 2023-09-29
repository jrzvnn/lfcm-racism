from PIL import Image
import os

# Input and output folder paths
input_folder = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Raw_Data/images"
output_folder = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/images"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to resize an image and save it
def resize_image(input_path, output_path, size=(400, 400)):
    try:
        image = Image.open(input_path)
        image = image.resize(size, Image.LANCZOS)
        image.save(output_path, "JPEG")
        print(f"Resized {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# Recursively process files in the input folder
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(".jpg"):
            input_path = os.path.join(root, file)
            # Generate the corresponding output path
            relative_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            resize_image(input_path, output_path)
