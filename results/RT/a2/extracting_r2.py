#%%
import pandas as pd
import glob
import os

# Define the directory as the current directory
directory_path = '.'

# Use glob to get all CSV files in the current directory
all_csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

# Initialize an empty list to hold the results
results = []

# Loop through the list of CSV files
for file in all_csv_files:
    try:
        # Read each CSV file into a dataframe
        df = pd.read_csv(file)
        
        # Check if required columns are present
        if 'Metric' in df.columns and 'Value' in df.columns:
            # Extract the value of r2_validation if it exists
            r2_values = df.loc[df['Metric'] == 'r2_validation', 'Value']
            if not r2_values.empty:
                r2_validation = r2_values.values[0]
                # Extract the name of the file without the extension
                file_name = os.path.splitext(os.path.basename(file))[0]
                # Append the result to the list
                results.append({'Metric': file_name + '_r2_validation', 'Value': r2_validation})
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Convert the list of results into a dataframe
results_df = pd.DataFrame(results)

# Sort the dataframe by the 'Metric' column in alphabetical order
results_df = results_df.sort_values(by='Metric')

# Define the output directory and ensure it exists
output_directory = './summary'
os.makedirs(output_directory, exist_ok=True)

# Save the results to a new CSV file
output_path = os.path.join(output_directory, 'r2_values.csv')
results_df.to_csv(output_path, index=False)

# Print the results
print(results_df)

# %%
