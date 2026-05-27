import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

from neural_doa_train import ANGLE_STEP, DoANet, extract_features


DEFAULT_MODEL = Path("sessions/combined_neural_doa/lightweight_gcc_mlp.pt")
SAMPLE_RATE = 16000
RESPEAKER_HINTS = ("respeaker", "seeed", "usb")


def load_model(model_path, input_dim=306):
    model = DoANet(input_dim)
    state = torch.load(model_path, map_location="cpu")

    # Older saved models used nn.Sequential directly: "0.weight", "3.bias", ...
    # The simplified DoANet stores the same layers under "net.0.weight", ...
    if state and not next(iter(state)).startswith("net."):
        state = {f"net.{key}": value for key, value in state.items()}

    model.load_state_dict(state)
    model.eval()
    return model


def predict_angle(model, audio):
    feature = extract_features(audio)
    x = torch.from_numpy(feature).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        label = int(logits.argmax(dim=1).item())
    return label * ANGLE_STEP


def list_input_devices():
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"{idx:2d}: {dev['name']} ({dev['max_input_channels']} input channels)")


def find_respeaker_device():
    candidates = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] >= 4:
            candidates.append((idx, dev))

    for idx, dev in candidates:
        name = dev["name"].lower()
        if any(hint in name for hint in RESPEAKER_HINTS):
            return idx

    return candidates[0][0] if candidates else None


def validate_device(device_id, channels):
    dev = sd.query_devices(device_id, "input")
    if dev["max_input_channels"] < channels:
        raise RuntimeError(
            f"Selected input device has only {dev['max_input_channels']} channels, "
            f"but {channels} are required.\n"
            f"Device: {dev['name']}\n"
            "The ReSpeaker is probably not selected or not visible to sounddevice."
        )
    return dev


def main():
    parser = argparse.ArgumentParser(description="Realtime neural DoA estimate from ReSpeaker audio.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    device_id = args.device if args.device is not None else find_respeaker_device()
    if device_id is None:
        list_input_devices()
        raise RuntimeError(
            "No input device with at least 4 channels was found. "
            "Plug in the ReSpeaker and rerun this script."
        )

    dev = validate_device(device_id, args.channels)
    model = load_model(args.model)
    samples = int(SAMPLE_RATE * args.seconds)

    print(f"Loaded model: {args.model}")
    print(f"Using input device {device_id}: {dev['name']} ({dev['max_input_channels']} channels)")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            try:
                validate_device(device_id, args.channels)
            except Exception as exc:
                raise RuntimeError(
                    "Input device disappeared or is no longer valid. "
                    "The ReSpeaker may have been unplugged."
                ) from exc

            audio = sd.rec(
                samples,
                samplerate=SAMPLE_RATE,
                channels=args.channels,
                dtype="float32",
                device=device_id,
            )
            sd.wait()

            if audio.shape[1] < 4:
                raise RuntimeError("Need at least 4 audio channels from the ReSpeaker.")

            doa = predict_angle(model, audio[:, :4])
            print(f"\rNeural DoA: {doa:03d}°", end="", flush=True)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
