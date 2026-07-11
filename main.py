import os
import wave
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "kokoro-v1.0.onnx"
VOICES_PATH = "voices.bin"

# Download Kokoro ONNX model files if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading Kokoro-82M model...")
    r = requests.get("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

if not os.path.exists(VOICES_PATH):
    print("Downloading voices configuration...")
    r = requests.get("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices.bin")
    with open(VOICES_PATH, "wb") as f:
        f.write(r.content)

# Initialize Kokoro ONNX Engine
from kokoro_onnx import KokoroOnnx
kokoro = KokoroOnnx(MODEL_PATH, VOICES_PATH)
print("Kokoro-82M God Mode loaded successfully!")

# Map your interviewers/teachers to natural Studio Voices
VOICE_MAP = {
    # Female personas
    "shalini": "af_sarah",
    "neha": "af_bella",
    "sneha": "af_sky",
    "priya": "bf_emma",
    "maya": "af_nicole",
    "divya": "af_sarah",
    
    # Male personas
    "vikram": "am_adam",
    "aditya": "am_michael",
    "rajesh": "am_fenrir",
    "abhijit": "bm_george",
    "anish": "am_adam",
    "kashyap": "am_michael",
    "karthic": "bm_lewis"
}

class TTSRequest(BaseModel):
    text: str
    voice: str

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    try:
        voice_name = VOICE_MAP.get(req.voice.lower(), "am_adam")
        clean_text = req.text.strip()

        # Generate audio samples
        samples, sample_rate = kokoro.create(clean_text, voice=voice_name, speed=1.0, lang="en-us")

        # Convert to Int16 WAV PCM
        audio_array = np.clip(samples, -1.0, 1.0)
        audio_int16 = (audio_array * 32767).astype(np.int16)

        output_file = "/tmp/output.wav" if os.path.exists("/tmp") else "output.wav"
        with wave.open(output_file, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return FileResponse(output_file, media_type="audio/wav", filename="speech.wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
