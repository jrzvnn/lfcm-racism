import pandas as pd

# Load the CSV files into pandas DataFrames
test_df = pd.read_csv('/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/test3.csv')
comments_df = pd.read_csv('/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/comments.csv')

# Merge the DataFrames based on the 'entry_id' column
merged_df = pd.merge(test_df, comments_df[['entry_id', 'tweet_text']], on='entry_id', how='left')

# Rename the 'tweet_text' column from comments_df to 'comment_text'
merged_df.rename(columns={'tweet_text_y': 'comment_text'}, inplace=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('output.csv', index=False)
