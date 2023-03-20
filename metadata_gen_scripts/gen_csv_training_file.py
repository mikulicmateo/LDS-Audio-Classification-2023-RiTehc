import os
import pandas as pd

current_dir = os.getcwd()
training_dataset_path = r'IRMAS_Training_Data'
os.chdir(training_dataset_path)

folders = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']
columns = ['path', 'tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']
type = {
    "tru": 0,
    "gac": 0,
    "sax": 0,
    "cel": 0,
    "flu": 0,
    "gel": 0,
    "vio": 0,
    "cla": 0,
    "pia": 0,
    "org": 0,
    "voi": 0
}


files = []

for folder in folders:
    type.update({str(folder): 1})

    for file in os.listdir(folder):
        files.append([os.path.join('IRMAS_Training_Data', folder, file), 
                      type.get("tru"), 
                      type.get("gac"), 
                      type.get("sax"), 
                      type.get("cel"), 
                      type.get("flu"), 
                      type.get("gel"), 
                      type.get("vio"),
                      type.get("cla"), 
                      type.get("pia"), 
                      type.get("org"), 
                      type.get("voi")])
        
    type.update({str(folder): 0})

os.chdir(os.path.join(current_dir, 'IRMAS_Training_Data'))
pd.DataFrame(files, columns=columns).to_csv('training_annotation_file.csv')