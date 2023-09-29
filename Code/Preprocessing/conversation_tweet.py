import os
import json
import re

# Function to preprocess tweet text
def preprocess_tweet_text(tweet_text):
    # Convert tweet_text to a string
    tweet_text = str(tweet_text)
    
    # Remove links
    tweet_text = re.sub(r'http\S+', '', tweet_text)
    
    # Remove hashtags
    tweet_text = re.sub(r'#\w+', '', tweet_text)
    
    # Remove emojis (this is a basic example, you may want to refine it)
    tweet_text = tweet_text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove user mentions
    tweet_text = re.sub(r'@\w+', '', tweet_text)
    
    # Remove symbols
    tweet_text = re.sub(r'[^\w\s]', '', tweet_text)
    
    # Remove newlines
    tweet_text = tweet_text.replace('\n', ' ')
    
    return tweet_text

# Input and output directories
input_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Raw_Data/conversation_tweets"
output_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/conversation_tweets"

# Traverse input directory
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".json"):
            input_path = os.path.join(root, filename)
            output_path = os.path.join(output_dir, os.path.relpath(input_path, input_dir))

            # Read the JSON file
            with open(input_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

            # Preprocess the tweet_text
            data["tweet_text"] = preprocess_tweet_text(data["tweet_text"])

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Write the modified JSON data to the output directory
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)

            print(f"Processed: {input_path} -> {output_path}")