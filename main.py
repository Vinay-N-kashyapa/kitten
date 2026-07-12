import os
import io
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize KittenTTS Model at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        from kittentts import KittenTTS
        print("Loading KittenTTS Model...")
        # This downloads a highly optimized ~40MB model + voice files 
        # that run easily inside Render's 512MB RAM limit.
        model = KittenTTS("KittenML/kitten-tts-mini-0.8")
        print("KittenTTS Model loaded successfully!")
    except Exception as e:
        print(f"Error loading KittenTTS: {e}")
        model = None

class TTSRequest(BaseModel):
    text: str
    voice: str

@app.post("/api/tts")
@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    global model
    if model is None:
        try:
            from kittentts import KittenTTS
            model = KittenTTS("KittenML/kitten-tts-mini-0.8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TTS Engine not initialized: {e}")

    try:
        # KittenTTS supports: 'Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo'
        # Map voice name to case-insensitive proper casing
        voice_map = {v.lower(): v for v in ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']}
        selected_voice = voice_map.get(req.voice.lower(), "Rosie")

        # Generate audio using the library (handles phonemes, styles, and speed automatically)
        audio = model.generate(req.text, voice=selected_voice, speed=1.0)
        
        # Write to WAV buffer
        wav_io = io.BytesIO()
        sf.write(wav_io, audio, 24000, format='WAV', subtype='PCM_16')
        wav_io.seek(0)
        
        return Response(content=wav_io.read(), media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health():
    return {"status": "healthy"}
