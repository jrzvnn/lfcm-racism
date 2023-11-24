import pandas as pd

# Read CSV file into a DataFrame
file_path = '/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/new_file.csv'
df = pd.read_csv(file_path)

# Clean 'json_file' column
df['json_file'] = df['json_file'].str.replace('tweet-', '')

# Save the modified DataFrame to a new CSV file
output_file_path = '/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/new_file1.csv'
df.to_csv(output_file_path, index=False)

print(f"'json_file' column cleaned. Result saved to {output_file_path}")
