import pandas as pd

# Read the CSV file with ids
ids_df = pd.read_csv('/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/dataset.csv', dtype={'id_str': object})

# Convert large integer values to strings
ids_df['entry_id'] = ids_df['entry_id'].astype(str)

# Read the tweet_txt.txt file
with open('/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Embeddings/fulldata.txt', 'r') as file:
    tweet_data = file.readlines()

# Replace the first column with ids from the CSV
for i in range(len(tweet_data)):
    tweet_data[i] = str(ids_df['entry_id'].iloc[i]) + tweet_data[i][tweet_data[i].index(','):]

# Write the updated data back to tweet_txt.txt
with open('/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Embeddings/fulldata.txt', 'w') as file:
    file.writelines(tweet_data)
