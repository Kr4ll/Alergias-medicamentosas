import re
import os
def extract_text_between(text, start_str):
    # Regular expression to find the text from start_str to end_str, including those markers
    pattern = re.escape(start_str) + '.*'
    return re.sub(pattern, '', text, flags=re.DOTALL)


# Function to read from a file, extract text, and overwrite the same file
def process_directory(directory_path, start_str):
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            extracted_text = extract_text_between(text, start_str)

            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(extracted_text)

start_str = "Juicio Clínico"


process_directory('../NoAlergico', start_str)
process_directory('../Alergico', start_str)
