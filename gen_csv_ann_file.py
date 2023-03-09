import os
import pandas as pd

os.chdir(r'IRMAS_Training_Data')

folders = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']

files = []

for folder in folders:
    for file in os.listdir(folder):
        files.append([os.path.join('IRMAS_Training_Data', folder, file), folder, folders.index(folder)])

pd.DataFrame(files, columns=['path', 'class', 'classId']).to_csv('training_annotation_file.csv')