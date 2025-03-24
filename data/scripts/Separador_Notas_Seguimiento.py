import os
import shutil

def classificator (filesPath, pathDest):
    os.makedirs(pathDest, exist_ok=True)

    listFiles = os.listdir(filesPath)
    for filename in listFiles:
        file_path = os.path.join(filesPath, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        if "Informe de Seguimiento" in text:
            shutil.move(os.path.join(filesPath, filename), os.path.join(pathDest, filename))

alergic='../Alergico'
nonAlergic='../NoAlergico'
seguimientoAlergic='../SeguimientoAlergico'

seguimientoNoAlergic='../SeguimientoNoAlergico'

alergicFiles = os.listdir(alergic)
nonAlergicFiles = os.listdir(nonAlergic)

classificator (alergic, seguimientoAlergic)
classificator (nonAlergic, seguimientoNoAlergic)
