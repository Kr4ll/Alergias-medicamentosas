import os
import shutil


def append_and_delete(source_file, target_file):
    # Read content from the source file
    with open(source_file, 'r', encoding='utf-8') as src:
        content = src.read()

    # Append content to the target file
    with open(target_file, 'a', encoding='utf-8') as tgt:
        tgt.write(content)

    # Delete the source file
    os.remove(source_file)

def join_by_patient(sourcePath, targetPath):
    for filename in os.listdir(targetPath):
        id = filename.split('_')[0]
        for filename2 in os.listdir(sourcePath):
            id2 = filename2.split('_')[0]
            if id == id2 and filename2 != filename:
                append_and_delete(os.path.join(sourcePath, filename2), os.path.join(targetPath, filename))


alergic='../Alergico'
nonAlergic='../NoAlergico'
seguimientoAlergic='../SeguimientoAlergico'

seguimientoNoAlergic='../SeguimientoNoAlergico'

join_by_patient(alergic, alergic)
join_by_patient(nonAlergic, nonAlergic)

# Iterate over the files



