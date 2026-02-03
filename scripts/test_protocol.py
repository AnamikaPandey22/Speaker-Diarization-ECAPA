#It checks whether pyannote can correctly read your dataset (audio + RTTM) using database.yml.
#get_protocol → asks pyannote:“Give me the dataset described in database.yml” FileFinder → helps pyannote find audio files
import os
from pyannote.database import get_protocol, FileFinder

# 1. Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "database.yml")
os.environ["PYANNOTE_DATABASE_CONFIG"] = config_path

# 2. Initialize preprocessor
preprocessors = {'audio': FileFinder()}

# 3. Load the protocol
try:
    protocol = get_protocol(
        'HindiBhojpuri.SpeakerDiarization.Segmentation', 
        preprocessors=preprocessors
    )
    print("Protocol loaded successfully!")
except Exception as e:
    print(f"Failed to load protocol: {e}")
    exit()

# 4. Detailed Data Verification
# This replaces your previous testing loop
for file in protocol.test():
    print("\n" + "="*30)
    print(f"FILE URI:     {file['uri']}")
    print(f"AUDIO PATH:   {file['audio']}")
    
    # Load the annotation (the RTTM data)
    annotation = file['annotation']
    print(f"SEGMENTS FOUND: {len(annotation)}")
    
    print("-" * 30)
    print("START     | END       | SPEAKER")
    print("-" * 30)
    
    # Iterate through the first 5 segments to keep the output clean
    for i, (segment, track, label) in enumerate(annotation.itertracks(yield_label=True)):
        if i >= 5: 
            print("... (and more)")
            break
        print(f"{segment.start:9.2f}s | {segment.end:9.2f}s | {label}")
    
    print("="*30)
    
    # Only check the first file for now
    break


# database.yml is correct, audio paths are correct
# RTTM files load correctly
# speaker segments exist
# segmentation training CAN start