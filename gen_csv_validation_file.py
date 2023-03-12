import os
import pandas as pd

current_dir = os.getcwd()
validation_dataset_path = r'.../IRMAS_Validation_Data'
os.chdir(validation_dataset_path)

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

files = os.listdir(os.getcwd())
files.sort()

data = []

for i in range(0, len(files), 2):
    txt_file = files[i]
    wav_file = files[i+1]

    if not(wav_file.__contains__(txt_file[: len(txt_file) - 4])):
        print("Error: Adjecent files not the same!")
        break

    with open(txt_file) as f:
        lines = f.readlines()

    for line in lines:
        type.update({line[0:3]: 1})

    data.append([os.path.join('IRMAS_Validation_Data', wav_file), 
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
    
    for line in lines:
        type.update({line[0:3]: 0})
    

os.chdir(os.path.join(current_dir, 'IRMAS_Validation_Data'))
pd.DataFrame(data, columns=columns).to_csv('validation_annotation_file.csv')