import os
import io
import wave
import numpy as np
import urllib.request
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download the lightweight v0.7 Nano model if not present locally (only ~24MB)
MODEL_PATH = "model_nano_int8.onnx"
if not os.path.exists(MODEL_PATH):
    print("Downloading lightweight nano model...", flush=True)
    url = "https://huggingface.co/KittenML/KittenTTS/resolve/main/model_nano_int8.onnx"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download completed!", flush=True)

# Optimize memory limit for Render's 512MB RAM free plan
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(MODEL_PATH, session_options, providers=['CPUExecutionProvider'])
print("ONNX model loaded successfully!", flush=True)

# Voice mapping matching your frontend KITTEN_VOICE_MAP
# Voices: 0=Bella, 1=Jasper, 2=Luna, 3=Bruno, 4=Rosie, 5=Hugo, 6=Kiki, 7=Leo
VOICE_MAP = {
    # 7 Interviewers
    "vikram": 3, 
    "shalini": 2, 
    "aditya": 5, 
    "neha": 6, 
    "rajesh": 7, 
    "sneha": 0, 
    "abhijit": 1, 
    
    # 2 Mentors
    "priya": 0, 
    "anish": 5, 
    
    # 4 Teachers
    "kashyap": 1, 
    "karthic": 3, 
    "maya": 2, 
    "divya": 4
}

class TTSRequest(BaseModel):
    text: str
    voice: str

@app.post("/api/tts")
@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    try:
        clean_text = req.text.strip()
        voice_id = VOICE_MAP.get(req.voice.lower(), 0)

        # Simple char to token mapping (no phonemizer required for this model!)
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

        # Convert float audio to int16 WAV
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_int16 = (audio_array * 32767).astype(np.int16)

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_io.seek(0)
        return Response(content=wav_io.read(), media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health():
    return {"status": "healthy"}
