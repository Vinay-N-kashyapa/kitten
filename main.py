import os
import wave
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort

app = FastAPI()

# Enable CORS so your website can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model_nano_int8.onnx"
MODEL_URL = "https://huggingface.co/KittenML/KittenTTS/resolve/main/model_nano_int8.onnx"

# Download the model on Render's disk
if not os.path.exists(MODEL_PATH):
    print("Downloading KittenTTS model...")
    r = requests.get(MODEL_URL, allow_redirects=True)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)
    print("Model downloaded successfully.")

# Load ONNX session
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
print("ONNX model loaded successfully.")

VOICE_MAP = {
    "vikram": 0, "shalini": 1, "aditya": 2, "neha": 3,
    "rajesh": 4, "sneha": 5, "abhijit": 6, "priya": 1, "anish": 2
}

class TTSRequest(BaseModel):
    text: str
    voice: str

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    try:
        clean_text = req.text.strip()
        voice_id = VOICE_MAP.get(req.voice.lower(), 0)

        tokens = [ord(c) for c in clean_text if ord(c) < 256]
        if not tokens:
            raise HTTPException(status_code=400, detail="Text has no valid characters")

        input_ids = np.array([tokens], dtype=np.int64)
        voice_tensor = np.array([voice_id], dtype=np.int64)

        results = session.run(None, {
            "input_ids": input_ids,
            "voice": voice_tensor
        })

        audio_data = results[0]
        audio_array = np.array(audio_data, dtype=np.float32).flatten()

        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_array * 32767).astype(np.int16)

        output_file = "/tmp/output.wav" if os.path.exists("/tmp") else "output.wav"
        with wave.open(output_file, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_int16.tobytes())

        return FileResponse(output_file, media_type="audio/wav", filename="speech.wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
