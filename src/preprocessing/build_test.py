import os
import json
import csv

def preprocess_tweet_text(tweet_text):
    # You can add your tweet text preprocessing logic here
    return tweet_text

def process_timeline_tweets(input_dir, output_dir, csv_path):
    # Create a dictionary to store mapping between entry_id and image filename
    entry_id_to_image = {}

    # Traverse input directory
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".json"):
                input_path = os.path.join(root, filename)

                # Read the JSON file
                with open(input_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)

                # Extract entry_id and image filename if "images" is not empty
                entry_id = data["id_str"]
                if data.get("images") and data["images"]:
                    image_filename = os.path.basename(data["images"][0])
                    entry_id_to_image[entry_id] = image_filename

    # Create a new CSV file for output
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["entry_id", "tweet_text", "image_filename"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Read the input CSV file
        with open("/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/test1.csv", 'r', encoding='utf-8') as input_csv:
            reader = csv.DictReader(input_csv)

            # Process each row
            for row in reader:
                entry_id = row["entry_id"]

                # Check if entry_id is in the dictionary
                if entry_id in entry_id_to_image:
                    image_filename = entry_id_to_image[entry_id]
                    row["image_filename"] = image_filename

                # Write the row to the new CSV file
                writer.writerow(row)

    print(f"Processing complete. Output CSV file: {csv_path}")

# Specify input and output directories, and the path of the input CSV
input_directory = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/timeline_tweets"
output_directory = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data"
csv_input_path = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/test1.csv"
csv_output_path = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/test3.csv"

# Process timeline tweets and update the CSV
process_timeline_tweets(input_directory, output_directory, csv_output_path)



