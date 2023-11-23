import pandas as pd

# Replace 'your_csv_file.csv' with the actual path to your CSV file
csv_file_path = '/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/new.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)

# Fill blank cells in the 'label' column with '0'
df['label'] = df['label'].fillna(0)

# Save the modified DataFrame back to the CSV file
df.to_csv(csv_file_path, index=False)
