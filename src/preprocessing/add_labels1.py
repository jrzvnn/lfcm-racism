import os
import csv
import json

# Paths
labels_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/labels"
csv_file = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/new_file1.csv"
output_csv_file = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/new.csv"

# Function to preprocess tweet text
def preprocess_tweet_text(tweet_text):
    # Implement your preprocessing logic here
    return tweet_text

# Read the existing CSV file
data = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Update the header with new column
data[0].append('is_racist')

# Traverse the labels directory and process each .json file
for root, dirs, files in os.walk(labels_dir):
    for filename in files:
        if filename.endswith(".json"):
            # Read the JSON file
            with open(os.path.join(root, filename), 'r', encoding='utf-8') as json_file:
                json_data = json.load(json_file)
                
                # Check if the 'json_file' column in CSV matches the .json file name
                for row in data[1:]:
                    if row[2] == filename:
                        # Check if 'tweet_image_text' key has value of 'Racist in Image Text'
                        if json_data.get('tweet_image_text') == 'Racist in Image Text':
                            row.append('1')
                        else:
                            row.append('0')
                        break

# Write the modified CSV data to the output file
with open(output_csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("Process completed successfully. Output file: ", output_csv_file)






