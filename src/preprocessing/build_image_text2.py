import os
import json
import csv

input_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/timeline_tweets"
csv_path = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/data1.csv"
output_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data"

# Load the image names from data1.csv
image_names = []
with open(csv_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        image_names.append(row['image_name'])

# Traverse input directory
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".json"):
            input_path = os.path.join(root, filename)
            output_path = os.path.join(output_dir, os.path.relpath(input_path, input_dir))

            # Read the JSON file
            with open(input_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

            # Get the image filenames from the JSON file
            json_image_names = [os.path.basename(image) for image in data.get('images', []) if image.endswith('.jpg')]

            # Check if there is a match in image names
            matched_json_filenames = []
            for image_name in image_names:
                for json_image_name in json_image_names:
                    if json_image_name == f"{image_name}.jpg":
                        matched_json_filenames.append(filename)
                        break

            # Update the CSV file with matched filenames
            with open(csv_path, 'r') as input_csv, open(os.path.join(output_dir, 'updated_data1.csv'), 'w', newline='') as output_csv:
                csv_reader = csv.reader(input_csv)
                csv_writer = csv.writer(output_csv)
                header = next(csv_reader)
                csv_writer.writerow(header + ['matched_filename'])
                for row in csv_reader:
                    if row[0] in matched_json_filenames:
                        csv_writer.writerow(row + [row[0]])
                    else:
                        csv_writer.writerow(row + [''])

            print(f"Processed: {input_path} -> {output_path}")
