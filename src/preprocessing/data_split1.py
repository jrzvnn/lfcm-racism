import pandas as pd
import numpy as np

# Specify the output directory
output_dir = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/new1.csv')

# Shuffle the rows of the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Calculate the number of rows for each set
train_size = int(0.8 * len(df))
valid_size = len(df) - train_size

# Split the DataFrame into train and validation sets
train_data = df[:train_size]
valid_data = df[train_size:]

# Save the train and validation sets to separate CSV files
train_data.to_csv(f"{output_dir}/train1.csv", index=False)
valid_data.to_csv(f"{output_dir}/valid1.csv", index=False)
