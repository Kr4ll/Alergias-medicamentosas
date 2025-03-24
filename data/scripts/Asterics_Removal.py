import re
import os

def remove_words_between_asterisks(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # to find words enclosed in "**"
        modified_content = re.sub(r'\*\*.*?\*\*', '', content)

        # place the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(modified_content)

        print(f"Successfully removed words between '**' in the file: {file_path}")

    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")

    except Exception as e:
        print(f"An error occurred: {e}")

def process_directory(directory_path):
    counter=0
    counter2=0
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        counter2+=1
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            remove_words_between_asterisks(file_path)


process_directory('../resources/clean_dataset/ByNotes/10%NoAlergico')
process_directory('../resources/clean_dataset/ByNotes/10%Alergico')
