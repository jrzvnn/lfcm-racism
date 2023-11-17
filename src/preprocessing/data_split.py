import pandas as pd
import numpy as np

# Specify the output directory
output_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/lfcm_timeline_dataset.csv')

# Shuffle the rows of the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Calculate the number of rows for each set
train_size = int(0.7 * len(df))
valid_size = int(0.15 * len(df))
test_size = len(df) - train_size - valid_size

# Split the DataFrame into train, validation, and test sets
train_data = df[:train_size]
valid_data = df[train_size:(train_size + valid_size)]
test_data = df[(train_size + valid_size):]

# Save the train, validation, and test sets to separate CSV files
train_data.to_csv(f"{output_dir}/train.csv", index=False)
valid_data.to_csv(f"{output_dir}/valid.csv", index=False)
test_data.to_csv(f"{output_dir}/test.csv", index=False)

