import os
import shutil
nonRegis='./NoRegistrados'
nonRegistered='./No_Registrados_Por_Pacientes'
os.makedirs(nonRegistered, exist_ok=True)

files = os.listdir(nonRegis)

# Iterate over the files
for filename in files:
    id = filename.split('_')[0]
    os.makedirs(os.path.join(nonRegistered,id), exist_ok=True)
    shutil.copy(os.path.join(nonRegis, filename), os.path.join(nonRegistered, id))

