import os
import json
import shutil

input_folder = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/timeline_tweets"
output_folder = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/output_folder"

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through subdirectories in the input folder
for subdir, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(subdir, file)

            # Read the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # Update the 'images' key in the JSON data
            if 'images' in data:
                data['images'] = [os.path.basename(image) for image in data['images']]

                # Write the updated data to a new file in the output directory
                output_file_path = os.path.join(output_folder, os.path.basename(subdir), file)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                with open(output_file_path, 'w') as output_file:
                    json.dump(data, output_file, indent=4)

# Copy the image files to the output directory
for subdir, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".jpg"):
            file_path = os.path.join(subdir, file)
            output_file_path = os.path.join(output_folder, os.path.basename(subdir), "images", file)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            shutil.copy(file_path, output_file_path)


