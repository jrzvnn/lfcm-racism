import os
import csv

def preprocess_text(text):
    # Remove unnecessary spaces and symbols, and convert to lowercase
    text = text.strip()
    text = ' '.join(text.split())  # Remove extra spaces
    text = ''.join(char for char in text if char.isalnum() or char.isspace())  # Keep alphanumeric characters and spaces
    text = text.lower()  # Convert to lowercase
    return text

# Input and output directories
input_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/ocr_output"
output_file = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/data1.csv"  # Replace with your desired output file path

# Create a list to store extracted data
extracted_data = []

# Traverse input directory
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".txt"):
            input_path = os.path.join(root, filename)

            # Read the text file
            with open(input_path, 'r', encoding='utf-8') as txt_file:
                text_content = txt_file.read()

            # Preprocess the data
            image_name = os.path.splitext(filename)[0]  # Remove ".txt" extension
            processed_text = preprocess_text(text_content)

            # Check if both image_name and processed_text have values before appending
            if image_name and processed_text:
                extracted_data.append([image_name, processed_text])

# Write the extracted data to a CSV file
with open(output_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["image_name", "image_text"])  # Write header row
    csv_writer.writerows(extracted_data)

print(f"CSV file created at {output_file}")

