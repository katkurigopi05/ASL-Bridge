#!/usr/bin/env python3
"""
pipeline/tts.py — Text-to-Speech Wrapper (pyttsx3)

DESCRIPTION:
    Offline text-to-speech engine using pyttsx3. Provides a threaded
    queue-based interface so TTS calls don't block the main inference loop.

USAGE:
    python pipeline/tts.py --text "Hello, this is ASL Bridge"
    python pipeline/tts.py --interactive

INPUTS:
    --text          Text string to speak
    --interactive   Enter interactive mode (type to speak)
    --rate          Speech rate in words per minute (default: 150)
    --volume        Volume 0.0-1.0 (default: 0.9)

OUTPUTS:
    Audio output through system speakers
"""

import argparse
import logging
import queue
import sys
import threading
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("tts")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class TTSEngine:
    """
    Thread-safe text-to-speech engine using pyttsx3.

    Maintains a background thread that consumes text from a queue
    and speaks it sequentially. This prevents blocking the main
    inference pipeline.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Parsed config.yaml dictionary
        """
        self.config = config
        tts_config = config.get("tts", {})

        self._rate = tts_config.get("rate", 150)
        self._volume = tts_config.get("volume", 0.9)
        self._voice_index = tts_config.get("voice_index", 0)

        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._engine = None
        self._engine_lock = threading.Lock()

        # Initialize engine
        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize the pyttsx3 engine."""
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)

            # Set voice
            voices = self._engine.getProperty("voices")
            if voices and self._voice_index < len(voices):
                self._engine.setProperty("voice", voices[self._voice_index].id)
                logger.info(f"✅ TTS initialized: {voices[self._voice_index].name}")
            else:
                logger.info("✅ TTS initialized with default voice")

        except ImportError:
            logger.error("❌ pyttsx3 not installed. Run: pip install pyttsx3")
            self._engine = None
        except Exception as e:
            logger.error(f"❌ TTS init failed: {e}")
            self._engine = None

    def speak(self, text: str) -> None:
        """
        Add text to the speech queue.
        Non-blocking — returns immediately.

        Args:
            text: Text string to speak
        """
        if not text or not text.strip():
            return

        self._queue.put(text.strip())
        logger.debug(f"Queued for TTS: '{text.strip()}'")

        # Start background thread if not running
        if not self._running:
            self.start()

    def speak_sync(self, text: str) -> None:
        """
        Speak text synchronously (blocks until done).

        Args:
            text: Text string to speak
        """
        if self._engine is None:
            # Fallback to macOS native 'say' if pyttsx3 failed
            if sys.platform == "darwin":
                import subprocess
                try:
                    subprocess.run(["say", text])
                except Exception as e:
                    logger.error(f"Fallback macOS 'say' failed: {e}")
            else:
                logger.error("TTS engine not initialized")
            return

        with self._engine_lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS speak error: {e}")

    def start(self) -> None:
        """Start the background speech thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._speech_loop, daemon=True)
        self._thread.start()
        logger.info("🔊 TTS background thread started")

    def stop(self) -> None:
        """Stop the background speech thread."""
        self._running = False
        self._queue.put(None)  # Sentinel to unblock the queue
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("🔇 TTS background thread stopped")

    def _speech_loop(self) -> None:
        """Background thread that processes the speech queue."""
        while self._running:
            try:
                text = self._queue.get(timeout=1.0)
                if text is None:
                    break

                logger.info(f"🔊 Speaking: '{text}'")
                self.speak_sync(text)
                self._queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech loop error: {e}")

    @property
    def is_speaking(self) -> bool:
        """Check if TTS is currently speaking or has items queued."""
        return not self._queue.empty()

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ASL Bridge — Text-to-Speech Engine")
    parser.add_argument("--text", type=str, help="Text to speak")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--rate", type=int, default=150, help="Speech rate (WPM)")
    parser.add_argument("--volume", type=float, default=0.9, help="Volume (0.0-1.0)")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))

    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {"tts": {"rate": args.rate, "volume": args.volume}}

    # Override config with CLI args
    config.setdefault("tts", {})
    config["tts"]["rate"] = args.rate
    config["tts"]["volume"] = args.volume

    tts = TTSEngine(config)

    if args.text:
        tts.speak_sync(args.text)
    elif args.interactive:
        logger.info("🎤 Interactive TTS mode. Type text and press Enter. Ctrl+C to quit.")
        try:
            while True:
                text = input("\n> ").strip()
                if text:
                    tts.speak_sync(text)
        except (KeyboardInterrupt, EOFError):
            logger.info("\nExiting interactive mode")
    else:
        # Demo
        tts.speak_sync("Hello! This is ASL Bridge. Text to speech is working.")


if __name__ == "__main__":
    main()
