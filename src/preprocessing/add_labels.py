import os
import json
import csv

# Paths to the CSV file and labels directory
csv_file_path = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/data.csv"
labels_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/labels"

# Create a dictionary to store labels
labels_dict = {}

# Traverse labels directory to collect labels
for root, dirs, files in os.walk(labels_dir):
    for filename in files:
        if filename.endswith(".json"):
            label_file_path = os.path.join(root, filename)
            label_entry_id = os.path.splitext(filename)[0]  # Extract entry_id from filename
            with open(label_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                tweet_text = data.get("tweet_text", "")
                if tweet_text == "Racist in Text":
                    label = "1"
                elif tweet_text == "Not Racist in Text":
                    label = "0"
                else:
                    label = "0"  # Default label is "0" when no label is found
                labels_dict[label_entry_id] = {"label": label, "keyword": os.path.basename(root)}

# Open the CSV file and create a new CSV file with labels and keywords
output_csv_path = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/lfcm_timeline_dataset.csv"

with open(csv_file_path, 'r', newline='') as input_csv_file:
    csv_reader = csv.reader(input_csv_file)
    with open(output_csv_path, 'w', newline='') as output_csv_file:
        csv_writer = csv.writer(output_csv_file)
        header = next(csv_reader)  # Read the header
        header.extend(["label", "keyword"])  # Add new columns for labels and keywords
        csv_writer.writerow(header)  # Write the updated header

        for row in csv_reader:
            entry_id = row[0]  # Get the entry_id from the first column
            label_data = labels_dict.get(entry_id, {})
            label = label_data.get("label", "0")  # Default label is "0" if not found
            keyword = label_data.get("keyword", "")
            row.extend([label, keyword])  # Add the label and keyword to the row
            csv_writer.writerow(row)  # Write the updated row

print(f"CSV file with labels and keywords created at {output_csv_path}")

