import sounddevice as sd

print("Available audio devices:")
print("=" * 60)
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"  [{i}] {dev['name']}")
        print(f"       Input channels: {dev['max_input_channels']}")
        print()
