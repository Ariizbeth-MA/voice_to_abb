"""
Password verification test script.

Records a voice sample, transcribes it with Whisper,
and checks if it contains the correct password.
"""

from __future__ import annotations

import os
import yaml

from robot_voice_commander.modules.audio.capture import AudioCapture
from robot_voice_commander.modules.audio.transcriber import Transcriber
from robot_voice_commander.modules.speaker.password_verifier import PasswordVerifier


def load_config() -> dict:
    cfg_path = os.environ.get("RVC_CONFIG", "config/settings.yaml")
    with open(cfg_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    cfg = load_config()

    audio_capture = AudioCapture(cfg["audio"])
    transcriber = Transcriber(cfg["whisper"])
    verifier = PasswordVerifier(cfg["speaker_verification"])

    print("\nVerificación de contraseña")
    print("--------------------------")
    print(f"Di la contraseña cuando el sistema esté escuchando.")
    print("\nEscuchando...\n")

    audio = audio_capture.record_until_silence()
    audio_capture.close()

    if audio is None:
        print("No se detectó voz.")
        return

    transcript = transcriber.transcribe(audio)

    if transcript is None or not transcript.text:
        print("No se pudo transcribir el audio.")
        return

    print(f"Transcripción: '{transcript.text}'")

    authorized, message = verifier.verify(transcript.text)

    print(f"\nResultado: {message}")

    if authorized:
        print("✓ Puedes continuar con el comando de voz.")
    else:
        print("✗ Acceso denegado.")


if __name__ == "__main__":
    main()