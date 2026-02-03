import torch
import torchaudio
from pyannote.audio import Model
from pyannote.core import Annotation, Segment

# 1. PATHS
CHECKPOINT_PATH = "training_results/lightning_logs/version_2/checkpoints/epoch=4-step=2960.ckpt"
TEST_AUDIO = "dataset/audio/clip_07.wav" 

def run_test():
    print(f"Loading model directly...")
    model = Model.from_pretrained(CHECKPOINT_PATH)
    model.eval() # Set to evaluation mode

    # 2. Load Audio Manually
    waveform, sample_rate = torchaudio.load(TEST_AUDIO)
    
    # Model expects [batch, channels, samples] - adding a batch dimension
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    print("Running raw inference...")
    with torch.no_grad():
        # Get raw scores [batch, frames, speakers]
        # This returns probabilities for each speaker class
        scores = model(waveform)

    # 3. Simple thresholding to find speakers
    # If score > 0.5, we consider that speaker "active"
    print("\n--- Raw Model Detections ---")
    
    # We'll use a very simple logic to show you what the model sees
    # The output usually has several speaker 'slots' (e.g., 7 slots)
    num_speakers = scores.shape[-1]
    
    # Moving average/thresholding logic
    # (Simplified for debugging)
    for s in range(num_speakers):
        active_frames = torch.where(scores[0, :, s] > 0.5)[0]
        if len(active_frames) > 0:
            # Just showing first and last detection for this slot to keep it clean
            start_time = active_frames[0] * 0.016 # Approximate frame shift
            end_time = active_frames[-1] * 0.016
            print(f"Speaker Slot {s}: Detected activity between {start_time:.2f}s and {end_time:.2f}s")

if __name__ == "__main__":
    run_test()