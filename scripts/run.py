import os
# MUST BE AT THE VERY TOP
os.environ["SPEECHBRAIN_LOCAL_STRATEGY"] = "copy"

import torch
import torchaudio
import pandas as pd
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.database.util import load_rttm 
from pyannote.metrics.diarization import DiarizationErrorRate, DiarizationPurity, DiarizationCoverage

# --- THE DEFINITIVE FIX FOR PYTORCH 2.6+ SECURITY ERRORS ---
import torch.serialization
original_load = torch.load
def forced_load(f, map_location=None, pickle_module=None, **kwargs):
    kwargs['weights_only'] = False
    return original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
torch.load = forced_load
# -----------------------------------------------------------

# Configuration - Update these paths to match your project structure
CHECKPOINT_PATH = "training_results/lightning_logs/version_2/checkpoints/epoch=4-step=2960.ckpt"
TEST_LIST_PATH = "dataset/splits/test.txt" 
AUDIO_DIR = "dataset/audio"
RTTM_DIR = "dataset/rttm"
OUTPUT_CSV = "overall_model_performance.csv"

def run_global_evaluation():
    # 1. Load the fine-tuned model
    print(f"Loading fine-tuned model from: {CHECKPOINT_PATH}")
    seg_model = Model.from_pretrained(CHECKPOINT_PATH)
    
    # 2. Initialize the Diarization Pipeline
    print("Initializing Pipeline...")
    pipeline = SpeakerDiarization(
        segmentation=seg_model,
        embedding="speechbrain/spkrec-ecapa-voxceleb", 
        clustering="AgglomerativeClustering",
    )

    # Balanced parameters for diverse speaker counts
    params = {
    "segmentation": {
        "threshold": 0.58,       # High threshold to kill False Alarms
        "min_duration_off": 0.2, # Prevents fragmented "flickering" between speakers
    },
    "clustering": {
        "method": "centroid",    
        "threshold": 0.62,       # Lower threshold to encourage speaker separation
        "min_cluster_size": 1,
    },
}
    pipeline.instantiate(params)

    # 3. Initialize Metrics
    # Using 'total' metrics to accumulate across all files
    total_der_metric = DiarizationErrorRate()
    
    # 4. Load filenames from test.txt
    with open(TEST_LIST_PATH, 'r') as f:
        # Extract the URI (filename without extension) from each line
        # Adjust the split logic if your test.txt has a different format (e.g., space-separated)
        test_files = [line.strip().split()[0] for line in f if line.strip()]

    print(f"Found {len(test_files)} files in test set. Starting Batch Processing...")
    print("-" * 50)

    for uri in test_files:
        audio_path = os.path.join(AUDIO_DIR, f"{uri}.wav")
        rttm_path = os.path.join(RTTM_DIR, f"{uri}.rttm")

        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for {uri}. Skipping.")
            continue
        
        # Load Reference RTTM
        try:
            reference = load_rttm(rttm_path)[uri]
        except Exception as e:
            print(f"Warning: Could not load RTTM for {uri}. Error: {e}")
            continue

        # Run Diarization
        waveform, sample_rate = torchaudio.load(audio_path)
        test_file = {"waveform": waveform, "sample_rate": sample_rate, "uri": uri}
        
        # We allow the AI to determine speaker count dynamically (min 2, max 7)
        hypothesis = pipeline(test_file, min_speakers=2, max_speakers=7)

        # Accumulate the metric
        total_der_metric(reference, hypothesis, detailed=True)
        print(f"Done: {uri}")

    # 5. Final Calculations
    print("\n" + "="*50)
    print("             FINAL GLOBAL REPORT")
    print("="*50)

    # This creates a detailed table per file
    report_df = total_der_metric.report(display=True)
    
    # Global DER is the value of the metric after processing all files
    global_der = abs(total_der_metric)
    global_accuracy = max(0, (1 - global_der) * 100)

    print(f"\nOVERALL SYSTEM ACCURACY : {global_accuracy:.2f}%")
    print(f"GLOBAL DIARIZATION ERROR: {global_der * 100:.2f}%")
    print("="*50)

    # Save detailed report to CSV for your documentation
    report_df.to_csv(OUTPUT_CSV)
    print(f"Detailed file-by-file breakdown saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_global_evaluation()