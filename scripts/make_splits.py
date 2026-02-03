#We are dividing your audio dataset into train / dev / test lists that pyannote will later use.

import os
import random

# Path to audio folder
audio_dir = "dataset/audio"

# Collect all wav files
uris = [
    f.replace(".wav", "")
    for f in os.listdir(audio_dir)
    if f.endswith(".wav")
]

# Safety check
if len(uris) != 89:
    print(f"Warning: expected 89 files, found {len(uris)}")

# Shuffle for randomness
random.seed(42)
random.shuffle(uris)

# Split sizes for 89 files
train = uris[:71]
dev = uris[71:80]
test = uris[80:89]

# Create splits folder if not exists
os.makedirs("dataset/splits", exist_ok=True)

def write_split(name, data):
    with open(f"dataset/splits/{name}.txt", "w", encoding="utf-8") as f:
        for uri in data:
            f.write(uri + "\n")

write_split("train", train)
write_split("dev", dev)
write_split("test", test)

# Print summary
print("Dataset split completed:")
print(f"  Train: {len(train)} files")
print(f"  Dev  : {len(dev)} files")
print(f"  Test : {len(test)} files")

# 71 for training
# 9 for validation (dev)
# 9 for testing