import os
import glob
import json

# Define the directory path
directory_path = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/conversation_tweets"

# Function to delete files that do not match the criteria
def delete_files_not_matching_criteria(directory_path):
    # Iterate through all files in the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".json"):
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    lang = data.get("lang", "")
                    in_reply_to_status_id = data.get("in_reply_to_status_id_str", "")
                    parent_tweet_id = data.get("parent_tweet_id", "")
                    tweet_text = data.get("tweet_text", "")
                    images = data.get("images", [])
                    
                    # Check the conditions for deletion
                    if lang != "en" or in_reply_to_status_id != parent_tweet_id or (not tweet_text.strip() and not images):
                        # Delete the file
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")

# Call the function to delete files
delete_files_not_matching_criteria(directory_path)
