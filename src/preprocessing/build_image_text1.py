import os
import json
import pandas as pd

def preprocess_tweet_text(tweet_text):
    # Implement your tweet text preprocessing logic here
    # For now, let's just return the original text
    return tweet_text

def update_csv_with_id_str(csv_path, output_folder_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Traverse the output folder
    for root, dirs, files in os.walk(output_folder_path):
        for filename in files:
            if filename.endswith(".json"):
                json_path = os.path.join(root, filename)

                # Read the JSON file
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    json_data = json.load(json_file)

                # Check if the 'images' key exists in the JSON data
                if 'images' in json_data:
                    # Extract the 'id_str' from the JSON data
                    id_str = json_data.get('id_str', '')

                    # Iterate over the 'images' in the JSON data
                    for image in json_data['images']:
                        # Check if the image_name exists in the CSV DataFrame
                        mask = df['image_name'] == image

                        # If a match is found, update the 'id_str' column in the CSV DataFrame
                        if mask.any():
                            df.loc[mask, 'id_str'] = id_str
                            print(f"Updated CSV: {image} -> {id_str}")

    # Save the updated DataFrame to the output CSV
    output_csv_path = os.path.join(output_folder_path, 'output.csv')
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Saved updated CSV to: {output_csv_path}")

if __name__ == "__main__":
    # Paths
    input_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/output_folder"
    csv_path = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/data1.csv"
    output_folder_path = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data"

    # Update the CSV with 'id_str'
    update_csv_with_id_str(csv_path, input_dir)










