#!/usr/bin/env python3
"""
pipeline/extractor.py — MediaPipe Landmark Extraction Engine

DESCRIPTION:
    Wraps MediaPipe Holistic (and Hands) to extract hand, pose, and face
    landmarks from video frames or static images. Provides both streaming
    (webcam/video) and single-shot (image) extraction modes.

FEATURES:
    - Live webcam feed with real-time landmark overlay
    - Video file processing with progress tracking
    - Single image extraction for static ASL alphabet
    - Automatic fallback: Holistic → Hands-only if Holistic fails
    - All MediaPipe calls wrapped in try/except for graceful degradation
    - Returns normalized numpy arrays ready for model input

USAGE (standalone):
    python pipeline/extractor.py --webcam 0
    python pipeline/extractor.py --video path/to/video.mp4
    python pipeline/extractor.py --image path/to/image.jpg --output keypoints.npy

INPUTS:
    --webcam    Webcam index (default: 0)
    --video     Path to a video file
    --image     Path to an image file
    --output    Output .npy path (for single file mode)
    --config    Path to config.yaml

OUTPUTS:
    - Yields (frame_idx, landmarks_array) tuples for streaming modes
    - Returns numpy array of shape (num_features,) for single images
    - Saves .npy files to data/processed/{dataset}/{split}/{class}/{sample_id}.npy

LANDMARK FORMAT:
    Hand landmarks: 21 points × 3 coords (x, y, z) = 63 features
    Full holistic: hand(63) + pose(33×4) + face(468×3) — configurable
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger("extractor")

# ── Resolve project root ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class MediaPipeExtractor:
    """
    MediaPipe-based landmark extractor for ASL hand signs.

    Supports three modes:
    1. extract_from_webcam(cam_index) — live webcam stream
    2. extract_from_video(video_path) — video file processing
    3. extract_from_image(image_path) — single static image

    All methods return normalized numpy arrays of hand landmarks.
    """

    def __init__(self, config: dict):
        """
        Initialize the extractor with configuration.

        Args:
            config: Parsed config.yaml dictionary
        """
        self.config = config
        self.mp_config = config.get("mediapipe", {})

        # Lazy-import MediaPipe to fail gracefully
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.mp_holistic = mp.solutions.holistic
            self.mp_hands = mp.solutions.hands
            self._mediapipe_available = True
            logger.info("✅ MediaPipe loaded successfully")
        except ImportError as e:
            logger.error(f"❌ MediaPipe not available: {e}")
            logger.info("   Install with: pip install mediapipe==0.10.14")
            self._mediapipe_available = False

        # Landmark dimensions
        self.hand_landmarks_count = self.mp_config.get("landmarks", {}).get("hand", 21)
        self.hand_features = self.hand_landmarks_count * 3  # x, y, z per landmark

    def _create_holistic(self):
        """Create a MediaPipe Holistic instance with config parameters."""
        if not self._mediapipe_available:
            return None

        holistic_config = self.mp_config.get("holistic", {})
        try:
            holistic = self.mp_holistic.Holistic(
                static_image_mode=holistic_config.get("static_image_mode", False),
                model_complexity=holistic_config.get("model_complexity", 1),
                min_detection_confidence=holistic_config.get("min_detection_confidence", 0.5),
                min_tracking_confidence=holistic_config.get("min_tracking_confidence", 0.5),
            )
            return holistic
        except Exception as e:
            logger.error(f"Failed to create Holistic model: {e}")
            return None

    def _create_hands(self):
        """Create a MediaPipe Hands instance as fallback."""
        if not self._mediapipe_available:
            return None

        hands_config = self.mp_config.get("hands", {})
        try:
            hands = self.mp_hands.Hands(
                static_image_mode=hands_config.get("static_image_mode", False),
                max_num_hands=hands_config.get("max_num_hands", 2),
                model_complexity=hands_config.get("model_complexity", 1),
                min_detection_confidence=hands_config.get("min_detection_confidence", 0.7),
                min_tracking_confidence=hands_config.get("min_tracking_confidence", 0.5),
            )
            return hands
        except Exception as e:
            logger.error(f"Failed to create Hands model: {e}")
            return None

    def _extract_hand_landmarks(self, results, source: str = "holistic") -> Optional[np.ndarray]:
        """
        Extract hand landmark coordinates from MediaPipe results.

        Args:
            results: MediaPipe processing results
            source: "holistic" or "hands" — determines how to access landmarks

        Returns:
            numpy array of shape (63,) — 21 landmarks × 3 coords, or None if no hand detected
        """
        landmarks = None

        try:
            if source == "holistic":
                # Try right hand first, then left hand
                if results.right_hand_landmarks:
                    landmarks = results.right_hand_landmarks
                elif results.left_hand_landmarks:
                    landmarks = results.left_hand_landmarks
            elif source == "hands":
                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                    landmarks = results.multi_hand_landmarks[0]

            if landmarks is None:
                return None

            # Convert to numpy array: [x0, y0, z0, x1, y1, z1, ...]
            coords = []
            for lm in landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            keypoints = np.array(coords, dtype=np.float32)

            # Validate shape
            if keypoints.shape[0] != self.hand_features:
                logger.warning(
                    f"Unexpected landmark count: {keypoints.shape[0]} "
                    f"(expected {self.hand_features})"
                )
                return None

            return keypoints

        except Exception as e:
            logger.debug(f"Landmark extraction error: {e}")
            return None

    def _get_static_hands(self):
        """Get or create a persistent static-mode Hands instance for batch processing."""
        if not hasattr(self, '_static_hands') or self._static_hands is None:
            if not self._mediapipe_available:
                return None
            try:
                self._static_hands = self.mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    model_complexity=self.mp_config.get("hands", {}).get("model_complexity", 1),
                    min_detection_confidence=self.mp_config.get("hands", {}).get(
                        "min_detection_confidence", 0.7
                    ),
                )
            except Exception as e:
                logger.error(f"Failed to create static Hands: {e}")
                self._static_hands = None
        return self._static_hands

    def extract_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a single image.
        Uses a persistent static-mode Hands instance for efficiency in batch processing.

        Args:
            image_path: Path to the image file

        Returns:
            numpy array of shape (63,) or None if no hand detected
        """
        if not self._mediapipe_available:
            logger.error("MediaPipe not available")
            return None

        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        try:
            # Read and convert image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Use persistent static Hands instance (much faster for batch)
            hands = self._get_static_hands()
            if hands is not None:
                try:
                    results = hands.process(image_rgb)
                    keypoints = self._extract_hand_landmarks(results, source="hands")
                    if keypoints is not None:
                        return keypoints
                except Exception as e:
                    logger.debug(f"Hands failed on {image_path.name}: {e}")

            logger.debug(f"No hand detected in: {image_path.name}")
            return None

        except Exception as e:
            logger.error(f"Image processing error for {image_path}: {e}")
            return None

    def close_static(self):
        """Close persistent static-mode instances (call after batch processing)."""
        if hasattr(self, '_static_hands') and self._static_hands is not None:
            self._static_hands.close()
            self._static_hands = None

    def extract_from_video(
        self, video_path: str
    ) -> Generator[tuple[int, Optional[np.ndarray]], None, None]:
        """
        Extract hand landmarks from each frame of a video file.

        Args:
            video_path: Path to the video file

        Yields:
            (frame_index, landmarks_array) tuples.
            landmarks_array is shape (63,) or None if no hand detected in that frame.
        """
        if not self._mediapipe_available:
            logger.error("MediaPipe not available")
            return

        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Video: {video_path.name} | {total_frames} frames | {fps:.1f} FPS")

        holistic = self._create_holistic()
        hands_fallback = self._create_hands()

        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Try Holistic
                    keypoints = None
                    if holistic is not None:
                        try:
                            results = holistic.process(image_rgb)
                            keypoints = self._extract_hand_landmarks(results, source="holistic")
                        except Exception as e:
                            logger.debug(f"Holistic failed on frame {frame_idx}: {e}")

                    # Fallback to Hands
                    if keypoints is None and hands_fallback is not None:
                        try:
                            results = hands_fallback.process(image_rgb)
                            keypoints = self._extract_hand_landmarks(results, source="hands")
                        except Exception as e:
                            logger.debug(f"Hands fallback failed on frame {frame_idx}: {e}")

                    yield (frame_idx, keypoints)

                except Exception as e:
                    logger.warning(f"Frame {frame_idx} processing error: {e}")
                    yield (frame_idx, None)

                frame_idx += 1

        finally:
            cap.release()
            if holistic is not None:
                holistic.close()
            if hands_fallback is not None:
                hands_fallback.close()

        logger.info(f"Processed {frame_idx} frames from {video_path.name}")

    def extract_from_webcam(
        self, cam_index: int = 0, show_preview: bool = True
    ) -> Generator[tuple[int, Optional[np.ndarray]], None, None]:
        """
        Extract hand landmarks from live webcam feed.

        Args:
            cam_index: Webcam device index (default: 0)
            show_preview: Whether to display the video feed with landmark overlay

        Yields:
            (frame_index, landmarks_array) tuples.
            Press 'q' to stop the webcam feed.
        """
        if not self._mediapipe_available:
            logger.error("MediaPipe not available")
            return

        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            logger.error(f"Failed to open webcam at index {cam_index}")
            return

        logger.info(f"🎥 Webcam opened (index {cam_index}). Press 'q' to quit.")

        holistic = self._create_holistic()
        hands_fallback = self._create_hands()

        frame_idx = 0
        fps_start_time = time.time()
        fps_frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Webcam frame read failed")
                    break

                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process with Holistic
                    keypoints = None
                    results = None
                    if holistic is not None:
                        try:
                            results = holistic.process(image_rgb)
                            keypoints = self._extract_hand_landmarks(results, source="holistic")
                        except Exception as e:
                            logger.debug(f"Holistic failed: {e}")

                    # Fallback to Hands
                    if keypoints is None and hands_fallback is not None:
                        try:
                            results_hands = hands_fallback.process(image_rgb)
                            keypoints = self._extract_hand_landmarks(results_hands, source="hands")
                        except Exception as e:
                            logger.debug(f"Hands fallback failed: {e}")

                    # Draw landmarks on preview
                    if show_preview and results is not None:
                        try:
                            if hasattr(results, "right_hand_landmarks") and results.right_hand_landmarks:
                                self.mp_drawing.draw_landmarks(
                                    frame,
                                    results.right_hand_landmarks,
                                    self.mp_holistic.HAND_CONNECTIONS,
                                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                    self.mp_drawing_styles.get_default_hand_connections_style(),
                                )
                            if hasattr(results, "left_hand_landmarks") and results.left_hand_landmarks:
                                self.mp_drawing.draw_landmarks(
                                    frame,
                                    results.left_hand_landmarks,
                                    self.mp_holistic.HAND_CONNECTIONS,
                                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                    self.mp_drawing_styles.get_default_hand_connections_style(),
                                )
                        except Exception:
                            pass  # Drawing is non-critical

                    # Calculate FPS
                    fps_frame_count += 1
                    elapsed = time.time() - fps_start_time
                    if elapsed > 1.0:
                        fps = fps_frame_count / elapsed
                        fps_start_time = time.time()
                        fps_frame_count = 0

                        if show_preview:
                            cv2.putText(
                                frame,
                                f"FPS: {fps:.1f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 255, 0),
                                2,
                            )

                    # Display status
                    if show_preview:
                        status = "HAND DETECTED" if keypoints is not None else "NO HAND"
                        color = (0, 255, 0) if keypoints is not None else (0, 0, 255)
                        cv2.putText(
                            frame,
                            status,
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            color,
                            2,
                        )
                        cv2.imshow("ASL Bridge — Extractor", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            logger.info("Webcam stopped by user (q pressed)")
                            break

                    yield (frame_idx, keypoints)

                except Exception as e:
                    logger.warning(f"Frame {frame_idx} error: {e}")
                    yield (frame_idx, None)

                frame_idx += 1

        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
            if holistic is not None:
                holistic.close()
            if hands_fallback is not None:
                hands_fallback.close()

        logger.info(f"Webcam session ended. Processed {frame_idx} frames.")

    def save_keypoints(
        self,
        keypoints: np.ndarray,
        dataset: str,
        split: str,
        gloss: str,
        sample_id: str,
    ) -> Path:
        """
        Save extracted keypoints to the standard project path.

        Args:
            keypoints: numpy array of landmarks
            dataset: Dataset name (e.g., "asl_alphabet")
            split: Data split ("train", "val", "test")
            gloss: Sign label/gloss name
            sample_id: Unique sample identifier

        Returns:
            Path to the saved .npy file
        """
        processed_dir = PROJECT_ROOT / self.config["paths"]["data"]["processed"]
        output_path = processed_dir / dataset / split / gloss / f"{sample_id}.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(str(output_path), keypoints)
        logger.debug(f"Saved: {output_path}")
        return output_path


def main():
    """Standalone CLI for the extractor."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ASL Bridge — MediaPipe Landmark Extractor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--webcam", type=int, nargs="?", const=0, help="Webcam index (default: 0)")
    group.add_argument("--video", type=str, help="Video file path")
    group.add_argument("--image", type=str, help="Image file path")
    parser.add_argument("--output", type=str, help="Output .npy path")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to config.yaml",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {"mediapipe": {}}

    extractor = MediaPipeExtractor(config)

    if args.webcam is not None:
        # Live webcam mode
        logger.info(f"Starting webcam extraction (index: {args.webcam})")
        frame_count = 0
        detected_count = 0
        for frame_idx, keypoints in extractor.extract_from_webcam(args.webcam):
            frame_count += 1
            if keypoints is not None:
                detected_count += 1
        logger.info(f"Session: {detected_count}/{frame_count} frames had hand detections")

    elif args.video:
        # Video file mode
        video_path = Path(args.video)
        frames = []
        for frame_idx, keypoints in extractor.extract_from_video(str(video_path)):
            if keypoints is not None:
                frames.append(keypoints)

        if frames:
            all_keypoints = np.stack(frames)
            output_path = args.output or str(video_path.with_suffix(".npy"))
            np.save(output_path, all_keypoints)
            logger.info(f"✅ Saved {all_keypoints.shape} to {output_path}")
        else:
            logger.error("❌ No landmarks detected in video")

    elif args.image:
        # Single image mode
        keypoints = extractor.extract_from_image(args.image)
        if keypoints is not None:
            output_path = args.output or str(Path(args.image).with_suffix(".npy"))
            np.save(output_path, keypoints)
            logger.info(f"✅ Saved {keypoints.shape} to {output_path}")
        else:
            logger.error("❌ No hand detected in image")


if __name__ == "__main__":
    main()
