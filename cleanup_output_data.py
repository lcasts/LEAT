import os

# List of folder names
folders = ["output_data/data_raw/excel/",
           "output_data/data_raw/txt/",
           "output_data/emissions/",
           "output_data/trajectory"]

def delete_files_in_folders(folder_list):
    for folder in folder_list:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file == ".gitkeep":  # Exclude ".gitkeep"
                        continue
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        pass

# Prompt user for confirmation
def confirm_deletion():
    response = input("Are you sure you want to delete all files in the specified folders? (yes/no): ").strip().lower()
    return response == 'yes'

# Main function
def main():
    if confirm_deletion():
        delete_files_in_folders(folders)
        print("Files deleted successfully.")
    else:
        print("Operation canceled. No files were deleted.")

if __name__ == "__main__":
    main()