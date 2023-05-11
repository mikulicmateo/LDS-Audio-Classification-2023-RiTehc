import os
import requests
import json

api_url = "http://localhost:8000/final_prediction"

DATA_LOCATION = "/home/mateo/Lumen-data-science/TestData/drive-download-20230430T150303Z-001"
# PATH_FILE = "/home/mateo/Lumen-data-science/TestData/drive-download-20230430T150303Z-001/track_75.wav"
final_dict = {}
for file in os.listdir(DATA_LOCATION):
    f = open(os.path.join(DATA_LOCATION, file), 'rb')
    form_data = {'file': f}
    server = requests.post(api_url, files=form_data)
    output = server.json()
    final_dict[file] = output

with open("test.json", "w") as outfile:
    outfile.write(json.dumps(final_dict, indent=4))


