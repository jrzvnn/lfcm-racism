import pandas as pd

def process_csv(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file, dtype={'entry_id': str})

    # Identify and mark duplicated rows based on 'entry_id', 'tweet_text_x', 'image_filename'
    duplicated_rows = df.duplicated(subset=['entry_id', 'tweet_text_x', 'image_filename'], keep='first')

    # Make duplicated rows blank for the specified columns
    columns_to_blank = ['entry_id', 'tweet_text_x', 'image_filename']

    # Explicitly cast empty string to the appropriate dtype
    for col in columns_to_blank:
        if df[col].dtype == 'int64':
            df.loc[duplicated_rows, col] = pd.NA
        else:
            df.loc[duplicated_rows, col] = ''

    # Write the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_csv = "/home/jrzvnn/Documents/Projects/lfcm-racism/Code/Preprocessing/output.csv"
    output_csv = "your_output_file1.csv"

    process_csv(input_csv, output_csv)




