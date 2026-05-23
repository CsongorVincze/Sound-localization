import pyaudio
import wave
import sys

# Official ReSpeaker 4 Mic Array Parameters
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6 
RESPEAKER_WIDTH = 2
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "test_respeaker_raw.wav"

p = pyaudio.PyAudio()

def list_devices():
    print("\n--- Available Audio Devices ---")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        print(f"Index {i}: {dev['name']} (Inputs: {dev['maxInputChannels']}, Outputs: {dev['maxOutputChannels']})")
    print("-------------------------------\n")

def record(device_index):
    print(f"[*] Opening ReSpeaker at index {device_index}...")
    try:
        stream = p.open(
            rate=RESPEAKER_RATE,
            format=p.get_format_from_width(RESPEAKER_WIDTH),
            channels=RESPEAKER_CHANNELS,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
    except Exception as e:
        print(f"[!] Error opening stream: {e}")
        return

    print(f"[*] Recording {RECORD_SECONDS} seconds...")
    frames = []

    for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("[*] Done recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save as 6-channel WAV
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(RESPEAKER_CHANNELS)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"[*] Saved to {WAVE_OUTPUT_FILENAME}")
    print("[*] Please play this file in VLC or Audacity. Channels 1-4 should contain the mics.")

if __name__ == "__main__":
    list_devices()
    try:
        idx = input("Enter the Index of your ReSpeaker: ")
        if idx == "":
            print("No index provided. Exiting.")
            sys.exit(0)
        record(int(idx))
    except KeyboardInterrupt:
        print("\nAborted.")
