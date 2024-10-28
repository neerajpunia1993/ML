import os

# Define the directory containing the files
directory = r"C:\Users\neeraj\OneDrive - Cytel\Documents\CIC\Population_Enrichment\change name"

# List of roll numbers
roll_numbers = ['d19021', 'd19022', 'd19023', 'd19024']

# Get the list of files in the directory
files = os.listdir(directory)

# Ensure we have the correct number of files
if len(files) != len(roll_numbers):
    print("The number of files does not match the number of roll numbers.")
else:
    for i, file in enumerate(files):
        # Extract the original file extension
        _, file_extension = os.path.splitext(file)

        # Construct the new file name using the roll number and keep the original extension
        new_file_name = f"{roll_numbers[i]}{file_extension}"

        # Full paths for old and new file names
        old_file_path = os.path.join(directory, file)
        new_file_path = os.path.join(directory, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: "{file}" to "{new_file_name}"')

print("File renaming completed.")
