import pandas as pd

# Read CSV file into a DataFrame
file_path = '/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/output_folder/output.csv'
df = pd.read_csv(file_path)

# Drop rows where 'id_str' is NaN
df = df.dropna(subset=['id_str'])

# Save the modified DataFrame to a new CSV file
output_file_path = '/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/new_file.csv'
df.to_csv(output_file_path, index=False)

print(f"Rows with NaN in 'id_str' column removed. Result saved to {output_file_path}")
