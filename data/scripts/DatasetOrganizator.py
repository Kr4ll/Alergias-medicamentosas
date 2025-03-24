import xlrd
import os
import shutil

# Map to save patients from the excel
patients = {}

# Directories
patientsNotes='./Informes 2015-2023 WORD/anonymised'
alergics= './Alergico'
nonAlergicS='./NoAlergico'
nonRegistered='./NoRegistrados'

# Open the Excel file
workbook = xlrd.open_workbook('CONS ALE DGTCO PRUEBAS ALERGIA_ANOM.xls')

# Select the first sheet
worksheet = workbook.sheet_by_index(0)

# Iterate through all rows and columns
for row_index in range(1,worksheet.nrows):
    patients[worksheet.cell_value(row_index, 0)]=worksheet.cell_value(row_index, 10)

# Close the workbook when done (not necessary for xlrd)
files = os.listdir(patientsNotes)

# Iterate over the files
for filename in files:
    id = filename.split('_')[0]
    if id not in patients:
        shutil.copy(os.path.join(patientsNotes, filename), os.path.join(nonRegistered, filename))

    elif patients[id]=='NEGATIVO':
        shutil.copy(os.path.join(patientsNotes, filename), os.path.join(nonAlergicS, filename))
        print(f"Moved {filename} to {nonAlergicS}")
    else:
        shutil.copy(os.path.join(patientsNotes, filename), os.path.join(alergics, filename))
        print(patients[id])

