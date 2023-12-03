import os
import json
import csv

# Set the input and output directories
input_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/conversation_tweets"
output_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data"

# Output CSV file path
csv_file_path = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/comments.csv"

# Open the CSV file for writing
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    # Create a CSV writer
    csv_writer = csv.writer(csv_file)

    # Write the header row
    csv_writer.writerow(['parent_tweet_id', 'tweet_text'])

    # Traverse input directory
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".json"):
                input_path = os.path.join(root, filename)

                # Read the JSON file
                with open(input_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)

                # Get the values of 'parent_tweet_id' and 'tweet_text'
                parent_tweet_id = data.get('parent_tweet_id', '')
                tweet_text = data.get('tweet_text', '')

                # Write the values to the CSV file
                csv_writer.writerow([parent_tweet_id, tweet_text])

                print(
                    f"Processed: {input_path} -> parent_tweet_id: {parent_tweet_id}, tweet_text: {tweet_text}")

print(f"CSV file created at: {csv_file_path}")
