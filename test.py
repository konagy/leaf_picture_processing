import os

# Define the folder path
folder_path = './pictures'
# List all files in the given folder
files = os.listdir(folder_path)
# Filter out directories, keeping only files
files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

# Print the file names
for file in files:
    print(file)
