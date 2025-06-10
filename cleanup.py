import os
from PIL import Image

# Define the root directory of your dataset
dataset_path = 'data'

# List of valid image extensions
valid_extensions = ('.jpg', '.jpeg', '.png')

# Function to remove non-image files
def cleanup_corrupt_images(root_dir):
    corrupted_count = 0
    total_files = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            # Check if the file has a valid extension
            if file.lower().endswith(valid_extensions):
                try:
                    # Attempt to open the image to test for corruption
                    with Image.open(file_path) as img:
                        img.verify() # Verify that the file is not corrupted
                except Exception as e:
                    print(f"Removing corrupted file: {file_path} - Error: {e}")
                    os.remove(file_path)
                    corrupted_count += 1
    
    print(f"\nCleanup complete. Scanned {total_files} files.")
    if corrupted_count > 0:
        print(f"Removed {corrupted_count} corrupted files.")
    else:
        print("No corrupted files were found.")

# Run the cleanup on your dataset folder
print("Starting cleanup for corrupt image files...")
cleanup_corrupt_images(dataset_path)