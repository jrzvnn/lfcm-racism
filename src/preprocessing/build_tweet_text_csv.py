import os
import json
import csv

# Input and output directories
input_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/timeline_tweets"
output_file = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/data.csv"  # Replace with your desired output file path

# Create a list to store extracted data
extracted_data = []

# Traverse input directory
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".json"):
            input_path = os.path.join(root, filename)

            # Read the JSON file
            with open(input_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

            # Extract "entry_id" and "tweet_text"
            entry_id = data.get("entry_id", "")
            tweet_text = data.get("tweet_text", "")

            # Clean "entry_id" by removing "tweet-" part
            if entry_id.startswith("tweet-"):
                entry_id = entry_id[len("tweet-"):]

            # Append the data to the list
            extracted_data.append([entry_id, tweet_text])

# Write the extracted data to a CSV file
with open(output_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["entry_id", "tweet_text"])  # Write header row
    csv_writer.writerows(extracted_data)

print(f"CSV file created at {output_file}")

