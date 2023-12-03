import pandas as pd

# Input CSV file path
input_csv_path = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/comments.csv"

# Output CSV file path
output_csv_path = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/comments1.csv"

# Read the base CSV file
df = pd.read_csv(input_csv_path)

# Group by 'parent_tweet_id' and aggregate 'tweet_text'
aggregated_df = df.groupby('parent_tweet_id')['tweet_text'].agg(' '.join).reset_index()

# Save the modified CSV file
aggregated_df.to_csv(output_csv_path, index=False)

print(f"Modified CSV file created at: {output_csv_path}")
