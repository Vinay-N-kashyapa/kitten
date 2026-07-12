import os
import io
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import soundfile as sf

# ── Monkey-Patching phonemizer for newer python/phonemizer compatibility ──
try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    if not hasattr(EspeakWrapper, 'set_data_path'):
        print("Applying compatibility patch to EspeakWrapper...")
        EspeakWrapper.set_data_path = lambda path: os.environ.update({"ESPEAK_DATA_PATH": path})
        print("Patch applied successfully!")
except Exception as e:
    print(f"Skipping EspeakWrapper compatibility patch: {e}")

# Initialize FastAPI App
app = FastAPI(title="KittenTTS Microservice")

# Enable CORS so the web app can communicate with the server
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
        # Use the official lightweight Mini model
        model = KittenTTS("KittenML/kitten-tts-mini-0.8")
        print("KittenTTS Model loaded successfully!")
    except Exception as e:
        print(f"Error loading KittenTTS: {e}")
        model = None

class TTSRequest(BaseModel):
    text: str
    voice: str = "Rosie"
    vibe: str = "neutral"
    speed: float = 1.0

@app.post("/api/tts")
@app.post("/tts")
async def generate_speech(req: TTSRequest):
    global model
    if model is None:
        try:
            from kittentts import KittenTTS
            model = KittenTTS("KittenML/kitten-tts-mini-0.8")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TTS Engine is not initialized: {e}")

    try:
        # KittenTTS supports: 'Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo'
        # Map voice name to proper casing just in case
        voice_map = {v.lower(): v for v in ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']}
        selected_voice = voice_map.get(req.voice.lower(), "Rosie")

        print(f"Generating speech for text: '{req.text[:30]}...' with voice: {selected_voice}")
        
        # Generate raw audio array using the kittentts package
        audio = model.generate(req.text, voice=selected_voice, speed=req.speed)
        
        # Write numpy float array to WAV byte buffer
        wav_io = io.BytesIO()
        sf.write(wav_io, audio, 24000, format='WAV', subtype='PCM_16')
        wav_io.seek(0)
        
        return Response(content=wav_io.read(), media_type="audio/wav")

    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "healthy", "engine": "KittenTTS", "model": "kitten-tts-mini-0.8"}

if __name__ == "__main__":
    import uvicorn
    # Render binds to PORT env variable automatically
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
