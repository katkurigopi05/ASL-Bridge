#!/usr/bin/env python3
"""
server/app.py — FastAPI Server for ASL Bridge

DESCRIPTION:
    Local REST API server exposing ASL translation endpoints.
    Bridges the Python ML pipeline with the frontend UI.

ENDPOINTS:
    GET  /                      — Health check + API info
    GET  /stream                — SSE stream of real-time predictions
    POST /translate             — Translate text to ASL fingerspelling data
    POST /speak                 — Text-to-speech: speak the given text
    GET  /vocab                 — Return loaded vocabulary
    GET  /status                — Pipeline status (model loaded, webcam active, etc.)
    POST /start-camera          — Start webcam inference
    POST /stop-camera           — Stop webcam inference
    POST /start-listen          — Start speech-to-text listening
    POST /stop-listen           — Stop speech-to-text listening

USAGE:
    python server/app.py
    uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload

INPUTS:
    --host          Server host (default: 127.0.0.1)
    --port          Server port (default: 8000)
    --checkpoint    Model checkpoint path
    --config        Path to config.yaml

OUTPUTS:
    REST API on http://127.0.0.1:8000
    API docs at http://127.0.0.1:8000/docs
"""

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("server")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ── Load config ──
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}

server_config = config.get("server", {})

# ── FastAPI App ──
app = FastAPI(
    title="ASL Bridge API",
    description="Bidirectional ASL ↔ Audio translation API. Fully local, no cloud.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=server_config.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──
inference_engine = None
tts_engine = None
stt_engine = None
camera_thread = None
camera_running = False
latest_prediction = {
    "label": None,
    "confidence": 0.0,
    "top3": [],
    "stabilized_label": None,
    "buffer_fill": 0.0,
    "latency_ms": 0.0,
    "timestamp": 0,
}
prediction_lock = threading.Lock()
confirmed_signs = []


# ── Pydantic Models ──
class TranslateRequest(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str


# ── Startup ──
@app.on_event("startup")
async def startup():
    global tts_engine, stt_engine, inference_engine
    logger.info("🚀 ASL Bridge server starting...")

    # Initialize TTS
    try:
        from pipeline.tts import TTSEngine
        tts_engine = TTSEngine(config)
        logger.info("✅ TTS engine loaded")
    except Exception as e:
        logger.warning(f"⚠️ TTS unavailable: {e}")

    # Initialize STT
    try:
        from pipeline.stt import STTEngine
        stt_engine = STTEngine(config)
        logger.info("✅ STT engine loaded")
    except Exception as e:
        logger.warning(f"⚠️ STT unavailable: {e}")

    # Initialize inference engine (without checkpoint by default)
    try:
        from pipeline.inference import InferenceEngine

        # Look for best checkpoint
        ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth"
        if ckpt_path.exists():
            inference_engine = InferenceEngine(config, checkpoint_path=str(ckpt_path))
            logger.info(f"✅ Inference engine loaded with model: {ckpt_path}")
        else:
            inference_engine = InferenceEngine(config)
            logger.info("✅ Inference engine loaded (no model checkpoint)")
    except Exception as e:
        logger.warning(f"⚠️ Inference engine unavailable: {e}")


# ── Mount frontend ──
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ── Routes ──
@app.get("/", response_class=JSONResponse)
async def root():
    """Health check and API info."""
    return {
        "name": "ASL Bridge API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "GET /": "This info page",
            "GET /stream": "SSE stream of predictions",
            "POST /translate": "Text → ASL fingerspelling",
            "POST /speak": "Text → Speech",
            "GET /vocab": "Loaded vocabulary",
            "GET /status": "Pipeline status",
            "POST /start-camera": "Start webcam inference",
            "POST /stop-camera": "Stop webcam inference",
        },
    }


@app.get("/status")
async def get_status():
    """Return current pipeline status."""
    return {
        "camera_active": camera_running,
        "model_loaded": inference_engine is not None and inference_engine.model is not None,
        "tts_available": tts_engine is not None,
        "stt_available": stt_engine is not None and stt_engine._recognizer is not None,
        "confirmed_signs": confirmed_signs[-20:],  # last 20 confirmed signs
        "avg_latency_ms": inference_engine.avg_latency_ms if inference_engine else 0.0,
    }


@app.get("/stream")
async def stream_predictions():
    """
    Server-Sent Events endpoint for real-time predictions.
    Frontend polls this for live updates.
    """
    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        while True:
            with prediction_lock:
                data = json.dumps(latest_prediction)
            yield {"event": "prediction", "data": data}
            await asyncio.sleep(0.1)  # 10 updates per second

    return EventSourceResponse(event_generator())


@app.post("/translate")
async def translate_text(request: TranslateRequest):
    """
    Translate text to ASL fingerspelling data.
    Returns a sequence of letters/signs for the frontend to render.
    """
    text = request.text.strip().upper()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    # Build fingerspelling sequence
    signs = []
    for char in text:
        if char.isalpha():
            signs.append({
                "type": "letter",
                "value": char,
                "display_ms": 800,  # display duration per letter
            })
        elif char == " ":
            signs.append({
                "type": "space",
                "value": " ",
                "display_ms": 400,
            })
        else:
            signs.append({
                "type": "other",
                "value": char,
                "display_ms": 300,
            })

    return {
        "original_text": request.text,
        "signs": signs,
        "total_duration_ms": sum(s["display_ms"] for s in signs),
    }


@app.post("/speak")
async def speak_text(request: SpeakRequest):
    """Speak text using TTS engine."""
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not available")

    tts_engine.speak(request.text)
    return {"status": "queued", "text": request.text}


@app.get("/vocab")
async def get_vocab():
    """Return the loaded vocabulary."""
    vocab_path = PROJECT_ROOT / "data" / "processed" / "asl_alphabet" / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path, "r") as f:
            return json.load(f)
    return {"error": "No vocabulary loaded", "num_classes": 0}


@app.post("/start-camera")
async def start_camera():
    """Start webcam inference in background thread."""
    global camera_thread, camera_running

    if camera_running:
        return {"status": "already_running"}

    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not available")

    def camera_loop():
        global camera_running, latest_prediction, confirmed_signs
        camera_running = True

        try:
            for frame_idx, keypoints in inference_engine.extractor.extract_from_webcam(0, show_preview=False):
                if not camera_running:
                    break

                result = inference_engine.process_frame(keypoints)
                result["timestamp"] = time.time()

                with prediction_lock:
                    latest_prediction.update(result)

                if result.get("stabilized_label"):
                    confirmed_signs.append(result["stabilized_label"])

        except Exception as e:
            logger.error(f"Camera loop error: {e}")
        finally:
            camera_running = False

    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()

    return {"status": "started"}


@app.post("/stop-camera")
async def stop_camera():
    """Stop webcam inference."""
    global camera_running
    camera_running = False
    return {"status": "stopped"}


@app.post("/start-listen")
async def start_listen():
    """Start speech-to-text background listening."""
    if stt_engine is None:
        raise HTTPException(status_code=503, detail="STT engine not available")

    def on_text(text):
        logger.info(f"🎤 Heard: {text}")
        # Could trigger text → ASL translation here

    stt_engine.listen_continuous(on_text)
    return {"status": "listening"}


@app.post("/stop-listen")
async def stop_listen():
    """Stop speech-to-text listening."""
    if stt_engine:
        stt_engine.stop_continuous()
    return {"status": "stopped"}


@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    """Serve the frontend UI."""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


def main():
    parser = argparse.ArgumentParser(description="ASL Bridge — FastAPI Server")
    parser.add_argument("--host", type=str, default=server_config.get("host", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=server_config.get("port", 8000))
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))

    args = parser.parse_args()

    logger.info(f"🚀 Starting ASL Bridge server at http://{args.host}:{args.port}")
    logger.info(f"   API docs: http://{args.host}:{args.port}/docs")
    logger.info(f"   Frontend: http://{args.host}:{args.port}/ui")

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
