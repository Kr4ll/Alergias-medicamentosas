
import os
def count_words_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            words = content.split()
            word_count = len(words)
            return word_count
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

def process_directory(directory_path):
    counter=0
    counter2=0
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        counter2+=1
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            if(count_words_in_file(file_path)>530):
                counter +=1
    return (str(counter) + '/' + str(counter2))


print(process_directory('../ByPatients/NoAlergico'))
print(process_directory('../ByPatients/Alergico'))