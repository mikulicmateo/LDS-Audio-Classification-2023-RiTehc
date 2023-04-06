from fastapi import FastAPI, File, UploadFile
from AudioService import AudioService
from ModelService import ModelService
import librosa
import tempfile
import os

import uvicorn

app = FastAPI(
    title="RiTehc model api - LDS 2023",
    description="This app servers as api to classification model of RiTehc team for Lumen Data Science 2023. "
                "competition",
    version="1.0.0"
)

audio_service = AudioService()
model_service = ModelService()


@app.post("/upload")
def test_upload(file: UploadFile = File()):
    if not file:
        return {"message": "ERROR! No upload file sent"}
    try:
        tf = tempfile.NamedTemporaryFile(delete=False)
        tfName = tf.name
        tf.seek(0)
        tf.write(file.file.read())
        tf.flush()

        sample_rate = librosa.get_samplerate(tfName)

        stream = librosa.stream(tfName,
                                block_length=1,
                                frame_length=3 * int(sample_rate),
                                hop_length=int(sample_rate),
                                fill_value=0,
                                mono=True)

        spectrograms = audio_service.get_spectrograms_from_stream(stream, sample_rate)
        prediction_dict = model_service.classify_audio(spectrograms)

        tf.close()
        os.unlink(tf.name)
        return prediction_dict
    except Exception as e:
        return {"message": f"ERROR! File could not be read. \nException: {e}"}


if __name__ == "__main__":
    uvicorn.run("app:app")
