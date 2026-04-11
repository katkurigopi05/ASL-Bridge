#!/usr/bin/env python3
"""
pipeline/stt.py — Speech-to-Text Wrapper (SpeechRecognition)

DESCRIPTION:
    Offline speech-to-text engine using SpeechRecognition library.
    Listens to microphone input and converts speech to text for
    the Voice → Signs translation mode.

USAGE:
    python pipeline/stt.py                    # listen and transcribe
    python pipeline/stt.py --interactive      # continuous listening mode
    python pipeline/stt.py --audio file.wav   # transcribe audio file

INPUTS:
    --interactive   Continuous listening mode
    --audio         Path to audio file to transcribe
    --timeout       Listening timeout in seconds (default: 5)
    --config        Path to config.yaml

OUTPUTS:
    Transcribed text string from speech input
"""

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import yaml

logger = logging.getLogger("stt")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class STTEngine:
    """
    Speech-to-text engine using SpeechRecognition library.

    Supports:
    - Single-shot listening (listen once, return text)
    - Continuous background listening with callback
    - Audio file transcription
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Parsed config.yaml dictionary
        """
        self.config = config
        stt_config = config.get("stt", {})

        self._energy_threshold = stt_config.get("energy_threshold", 300)
        self._pause_threshold = stt_config.get("pause_threshold", 0.8)
        self._phrase_time_limit = stt_config.get("phrase_time_limit", 10)

        self._recognizer = None
        self._microphone = None
        self._stop_listening = None
        self._is_listening = False

        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize the SpeechRecognition engine and microphone."""
        try:
            import speech_recognition as sr

            self._sr = sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = self._energy_threshold
            self._recognizer.pause_threshold = self._pause_threshold

            logger.info("✅ SpeechRecognition engine initialized")

            # Test microphone availability
            try:
                self._microphone = sr.Microphone()
                logger.info("✅ Microphone available")
            except (OSError, AttributeError) as e:
                logger.warning(f"⚠️ Microphone not available: {e}")
                logger.info("   Install PyAudio: pip install pyaudio")
                self._microphone = None

        except Exception as e:
            logger.error(f"❌ SpeechRecognition init error: {e}")
            self._recognizer = None

    def listen_once(self, timeout: int = 5) -> Optional[str]:
        """
        Listen to microphone and return transcribed text (single shot).

        Args:
            timeout: Maximum seconds to wait for speech

        Returns:
            Transcribed text string, or None if failed
        """
        if self._recognizer is None or self._microphone is None:
            logger.error("STT engine or microphone not initialized")
            return None

        try:
            with self._microphone as source:
                logger.info("🎤 Adjusting for ambient noise...")
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("🎤 Listening... (speak now)")

                audio = self._recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=self._phrase_time_limit,
                )

            return self._transcribe(audio)

        except self._sr.WaitTimeoutError:
            logger.warning("⏰ Listening timed out — no speech detected")
            return None
        except Exception as e:
            logger.error(f"Listening error: {e}")
            return None

    def listen_continuous(self, callback: Callable[[str], None]) -> None:
        """
        Start continuous background listening.
        Calls callback(text) whenever speech is detected.

        Args:
            callback: Function that receives transcribed text strings
        """
        if self._recognizer is None or self._microphone is None:
            logger.error("STT engine or microphone not initialized")
            return

        def audio_callback(recognizer, audio):
            """Called by background listener for each phrase."""
            text = self._transcribe(audio)
            if text:
                callback(text)

        with self._microphone as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=1.0)

        self._stop_listening = self._recognizer.listen_in_background(
            self._microphone,
            audio_callback,
            phrase_time_limit=self._phrase_time_limit,
        )
        self._is_listening = True
        logger.info("🎤 Continuous listening started (background)")

    def stop_continuous(self) -> None:
        """Stop continuous background listening."""
        if self._stop_listening is not None:
            self._stop_listening(wait_for_stop=False)
            self._is_listening = False
            logger.info("🔇 Continuous listening stopped")

    def transcribe_file(self, audio_path: str) -> Optional[str]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to WAV, AIFF, or FLAC audio file

        Returns:
            Transcribed text or None
        """
        if self._recognizer is None:
            logger.error("STT engine not initialized")
            return None

        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None

        try:
            with self._sr.AudioFile(str(audio_path)) as source:
                audio = self._recognizer.record(source)
            return self._transcribe(audio)
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return None

    def _transcribe(self, audio) -> Optional[str]:
        """
        Transcribe audio data using available recognizer.
        Tries Google (needs internet), then PocketSphinx (offline).

        Args:
            audio: SpeechRecognition AudioData object

        Returns:
            Transcribed text string, or None
        """
        # Try Google first (more accurate but needs internet)
        try:
            text = self._recognizer.recognize_google(audio)
            logger.info(f"📝 Transcribed (Google): '{text}'")
            return text
        except self._sr.UnknownValueError:
            logger.debug("Google: could not understand audio")
        except self._sr.RequestError:
            logger.debug("Google: API unavailable, trying offline...")

        # Fallback to PocketSphinx (offline)
        try:
            text = self._recognizer.recognize_sphinx(audio)
            logger.info(f"📝 Transcribed (Sphinx): '{text}'")
            return text
        except self._sr.UnknownValueError:
            logger.warning("Could not understand audio (all engines)")
            return None
        except Exception as e:
            logger.debug(f"Sphinx error: {e}")
            return None

    @property
    def is_listening(self) -> bool:
        """Whether continuous listening is active."""
        return self._is_listening


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ASL Bridge — Speech-to-Text Engine")
    parser.add_argument("--interactive", action="store_true", help="Continuous listening mode")
    parser.add_argument("--audio", type=str, help="Audio file to transcribe")
    parser.add_argument("--timeout", type=int, default=5, help="Listening timeout (seconds)")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))

    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {"stt": {}}

    stt = STTEngine(config)

    if args.audio:
        text = stt.transcribe_file(args.audio)
        if text:
            print(f"\n📝 Transcription: {text}")
        else:
            print("\n❌ Transcription failed")

    elif args.interactive:
        logger.info("🎤 Continuous listening mode. Press Ctrl+C to stop.\n")

        def on_text(text):
            print(f"\n📝 You said: {text}\n")

        stt.listen_continuous(on_text)

        try:
            import time
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            stt.stop_continuous()
            logger.info("\nExiting")

    else:
        # Single-shot listen
        text = stt.listen_once(timeout=args.timeout)
        if text:
            print(f"\n📝 You said: {text}")
        else:
            print("\n❌ No speech detected")


if __name__ == "__main__":
    main()
