#!/bin/bash
# Download and extract Google Speech Commands v0.02
# Run this on the remote server:  bash download_dataset.sh [target_dir]

TARGET=${1:-./data}
mkdir -p "$TARGET"

URL="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ARCHIVE="$TARGET/speech_commands_v0.02.tar.gz"

echo "Downloading Speech Commands v0.02 (~2.3 GB) to $TARGET ..."
wget -c -O "$ARCHIVE" "$URL"

echo "Extracting..."
tar -xzf "$ARCHIVE" -C "$TARGET"

echo "Verifying structure..."
EXPECTED_DIRS="yes no go stop left right up down on off backward forward follow learn"
MISSING=0
for d in $EXPECTED_DIRS; do
    if [ ! -d "$TARGET/$d" ]; then
        echo "  MISSING: $TARGET/$d"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -eq 0 ]; then
    echo "OK — dataset looks complete."
    echo "Run training with:"
    echo "  python train.py --data_dir $TARGET --out_dir . --num_workers 8"
else
    echo "$MISSING directories missing — extraction may have failed."
fi
