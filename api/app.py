from fastapi import FastAPI, File, UploadFile
import soundfile as sf
import torchaudio
import base64

import uvicorn

app = FastAPI(
    title="RiTehc model api - LDS 2023",
    description="This app servers as api to classification model of RiTehc team for Lumen Data Science 2023. competition",
    version="1.0.0"
)

@app.get("/upload")
def test_upload(file: UploadFile):
    if not file:
        return {"message": "ERROR! No upload file sent"}
    audio = (None, None)
    try:
        audio = torchaudio.load(file.file)
    except:
        return {"message": "ERROR! File could not be read."}
    #send to model
    return {"filename": file.filename, "len": file.content_type, "content": audio[1]}


if __name__ == "__main__":
    uvicorn.run("app:app")