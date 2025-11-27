import kagglehub
import shutil
import os

path = kagglehub.dataset_download("sanikamal/rock-paper-scissors-dataset")

print("Path to dataset files:", path)

# Define source and destination directories
source_dir = path  # The directory where the dataset was downloaded
dest_dir = os.path.join(os.getcwd(), 'datasets')  # Project's data directory

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)
# Move files from source to destination
for item in os.listdir(source_dir):
    s = os.path.join(source_dir, item)
    d = os.path.join(dest_dir, item)
    if os.path.isdir(s):
        shutil.move(s, d)
    else:
        shutil.move(s, d)
print(f"Datasets moved to {dest_dir}")