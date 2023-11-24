import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/new.csv')

# Copy the value from 'json_file' column to 'id_str' column, minus the '.json'
df['id_str'] = df['json_file'].apply(lambda x: x.replace('.json', ''))

# Save the updated DataFrame back to the CSV file
df.to_csv('/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/new1.csv', index=False)
