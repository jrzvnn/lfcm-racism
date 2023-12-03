import csv

import csv

def delete_columns(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            # Delete 3rd and 4th columns (0-based index)
            del row[2:4]
            writer.writerow(row)

if __name__ == "__main__":
    input_filename = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/test.csv"  # Replace with your input file name
    output_filename = "/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/test1.csv"      # Replace with your output file name

    delete_columns(input_filename, output_filename)
    print(f"Columns 3 and 4 deleted. Output saved to {output_filename}")

