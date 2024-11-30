import glob
import os

def get_files(directory, type):
    # Get all the files in the directory
    file_list = glob.glob(directory)

    # Write only the filenames into a text file
    with open(f'lists/lists_HSI/{type}.txt', 'w') as f:
        for file in file_list:
            filename = os.path.basename(file)  # Extract only the filename
            f.write(filename + '\n')

    print(f"Filenames for {type} have been saved.")


directory = r'<Path to training files>'
directory2 = r'<Path to validation files>'
directory3 = r'<Path to testing files>'

get_files(directory, 'train')
get_files(directory2, 'val')
get_files(directory3, 'test')

