"""
Speaker enrollment script.

This script records a voice sample and stores it as an authorized speaker.
It is intended to be used before running the main voice commander node.
"""

from __future__ import annotations

import argparse
import os
import yaml

from robot_voice_commander.modules.audio.capture import AudioCapture
from robot_voice_commander.modules.speaker.verifier import SpeakerVerifier


def load_config() -> dict:
    """
    Loads the project configuration file.
    """
    cfg_path = os.environ.get("RVC_CONFIG", "config/settings.yaml")

    with open(cfg_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register an authorized speaker voice sample."
    )

    parser.add_argument(
        "name",
        help="Name of the speaker to register. Example: arizbeth",
    )

    args = parser.parse_args()

    cfg = load_config()

    audio_cfg = cfg["audio"]
    speaker_cfg = cfg["speaker_verification"]

    audio_capture = AudioCapture(audio_cfg)
    speaker_verifier = SpeakerVerifier(speaker_cfg)

    current_speakers = speaker_verifier.list_speakers()

    print("\nAuthorized speakers currently registered:")
    if current_speakers:
        for speaker in current_speakers:
            print(f" - {speaker}")
    else:
        print(" - None")

    print("\nSpeaker enrollment")
    print("------------------")
    print(f"Speaker name: {args.name}")
    print("Please say a short phrase when the system starts listening.")
    print("Recommended phrase:")
    print('"Robot, this is my authorized voice command."')
    print("\nListening...\n")

    audio = audio_capture.record_until_silence()

    if audio is None:
        print("No voice was detected. Speaker was not enrolled.")
        audio_capture.close()
        return

    success = speaker_verifier.enroll_speaker(args.name, audio)

    audio_capture.close()

    if success:
        print(f"\nSpeaker '{args.name}' was enrolled successfully.")
    else:
        print(f"\nSpeaker '{args.name}' could not be enrolled.")


if __name__ == "__main__":
    main()