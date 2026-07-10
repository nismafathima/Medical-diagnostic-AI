# utils/speech_handler.py
# Whisper-based speech to text for patient symptom voice input
# OPTIMIZATION: Uses tiny model (32M params) instead of base (74M) for 3x faster CPU inference

import whisper
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

_model = None

def get_model():
    """Load Whisper tiny model for fast CPU inference.
    - Tiny: 39M params, 140MB (3x faster than base)
    - Base: 74M params, 140MB
    - Trade-off: Slightly lower accuracy but much faster
    """
    global _model
    if _model is None:
        try:
            # Use tiny model for CPU optimization
            _model = whisper.load_model("tiny", device="cpu")
        except Exception as e:
            print(f"⚠️ Could not load whisper tiny model: {e}. Trying base...")
            try:
                _model = whisper.load_model("base", device="cpu")
            except Exception as e2:
                print(f"⚠️ Could not load whisper model: {e2}")
                return None
    return _model


def transcribe(audio_bytes: bytes) -> str:
    """Convert audio bytes to text using Whisper with CPU optimization.
    
    Args:
        audio_bytes: Audio data in bytes
        
    Returns:
        Transcribed text or error message
    """
    try:
        model = get_model()
        if model is None:
            return "[Error: Whisper model could not be loaded]"
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            # Use fp16=False for CPU (CPU doesn't support fp16 efficiently)
            result = model.transcribe(tmp_path, language="en", fp16=False, verbose=False)
            return result["text"].strip()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        print(f"Transcription error: {e}")
        return f"[Transcription failed: {str(e)}]"
