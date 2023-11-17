import os
import glob

# Define the directory path
directory_path = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Raw_Data/timeline_tweets"

# Define the file pattern to keep
file_pattern = "tweet-*.json"

# Function to delete files that do not match the pattern
def delete_files_not_matching_pattern(directory_path, file_pattern):
    # Use glob to find files matching the pattern
    matching_files = glob.glob(os.path.join(directory_path, file_pattern))
    
    # Iterate through all files in the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file does not match the pattern
            if not file.startswith("tweet-") or not file.endswith(".json"):
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# Call the function to delete files
delete_files_not_matching_pattern(directory_path, file_pattern)
