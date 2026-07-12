import os
import io
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import soundfile as sf
import onnxruntime as ort

# ── Monkey-Patching onnxruntime SessionOptions to prevent Out-Of-Memory (OOM) on Render's 512MB free plan ──
_original_InferenceSession = ort.InferenceSession

def custom_InferenceSession(model_path, *args, **kwargs):
    print(f"Intercepted ONNX session creation for {model_path}. Injecting memory-optimized options.", flush=True)
    # Configure low-memory session options (Forces 1 CPU thread & sequential execution to save RAM)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # Inject optimized options
    kwargs['sess_options'] = session_options
    return _original_InferenceSession(model_path, *args, **kwargs)

ort.InferenceSession = custom_InferenceSession

# ── Monkey-Patching phonemizer for newer python/phonemizer compatibility ──
try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    if not hasattr(EspeakWrapper, 'set_data_path'):
        print("Applying compatibility patch to EspeakWrapper...", flush=True)
        EspeakWrapper.set_data_path = lambda path: os.environ.update({"ESPEAK_DATA_PATH": path})
        print("Patch applied successfully!", flush=True)
except Exception as e:
    print(f"Skipping EspeakWrapper compatibility patch: {e}", flush=True)

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
        print("Loading KittenTTS Model...", flush=True)
        # Use mini model with optimized thread options to fit under 512MB
        model = KittenTTS("KittenML/kitten-tts-mini-0.8")
        print("KittenTTS Model loaded successfully!", flush=True)
    except Exception as e:
        print(f"Error loading KittenTTS: {e}", flush=True)
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

        print(f"Generating speech for text: '{req.text[:30]}...' with voice: {selected_voice}", flush=True)
        
        # Generate raw audio array using the kittentts package
        audio = model.generate(req.text, voice=selected_voice, speed=req.speed)
        
        # Write numpy float array to WAV byte buffer
        wav_io = io.BytesIO()
        sf.write(wav_io, audio, 24000, format='WAV', subtype='PCM_16')
        wav_io.seek(0)
        
        return Response(content=wav_io.read(), media_type="audio/wav")

    except Exception as e:
        print(f"Inference error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "healthy", "engine": "KittenTTS", "model": "kitten-tts-mini-0.8"}

if __name__ == "__main__":
    import uvicorn
    # Render binds to PORT env variable automatically
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
