import os
import torch
import matplotlib.pyplot as plt
from pyannote.metrics.diarization import DiarizationErrorRate

# THE ULTIMATE BYPASS (Fixes PyTorch 2.6 security errors)
import torch.serialization
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# IMPORTS
from pyannote.core import notebook
from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm

AUDIO_PATH = r"dataset/audio/clip_03.wav"
RTTM_PATH = r"dataset/rttm/clip_03.rttm"

# INITIALIZE PIPELINE
print("Initializing AI Pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="hf_token_here"  # Replace with your Hugging Face token
)

# --- RUN DIARIZATION ---
print("AI is analyzing the audio...")
prediction = pipeline(AUDIO_PATH)

# --- LOAD GROUND TRUTH ---
gt_dict = load_rttm(RTTM_PATH)
uri = list(gt_dict.keys())[0]
ground_truth = gt_dict[uri]

# --- FIXED: CALCULATE DER USING REPORT ---
metric = DiarizationErrorRate()
# We process the specific file to get a clean report
metric(ground_truth, prediction, notebook=True) 
report = metric.report(display=True)

print("\n" + "="*50)
print("FINAL EVALUATION REPORT")
print(report)
print("="*50 + "\n")

## --- VISUALIZATION (UNCHANGED) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

plt.sca(ax1)
notebook.plot_annotation(ground_truth, ax=ax1) 
ax1.set_title("REFERENCE (Ground Truth)", fontsize=14, fontweight='bold')

plt.sca(ax2)
notebook.plot_annotation(prediction, ax=ax2)
ax2.set_title("HYPOTHESIS (Model Prediction)", fontsize=14, fontweight='bold')

plt.xlabel("Time (seconds)", fontsize=12)
plt.tight_layout()

print("Diarization complete! Displaying plot...")
plt.show()