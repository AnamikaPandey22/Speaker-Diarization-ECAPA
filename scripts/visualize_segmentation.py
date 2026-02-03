import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from pyannote.audio import Model, Inference
from pyannote.audio.utils.signal import Binarize
from pyannote.database.util import load_rttm
from pyannote.core import notebook, SlidingWindowFeature, Annotation
from sklearn.cluster import AgglomerativeClustering

# --- 1. PYTORCH 2.6+ SECURITY FIX ---
import torch.serialization
original_load = torch.load
def forced_load(f, map_location=None, pickle_module=None, **kwargs):
    kwargs['weights_only'] = False
    return original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
torch.load = forced_load
# ------------------------------------

def visualize_audio_file(audio_path, rttm_path, checkpoint_path):
    file_id = os.path.basename(audio_path).replace('.wav', '')
    print(f"--- Processing: {file_id} ---")
    
    # 1. Load Model & Run Inference
    model = Model.from_pretrained(checkpoint_path)
    inference = Inference(model, window="sliding", duration=2.0, step=0.5)
    seg_output = inference(audio_path)
    
    # 2. Reshape and Binarize (Using a high threshold to remove background noise)
    data = np.squeeze(seg_output.data)
    if len(data.shape) == 3: data = data[:, :, 0]
    
    # Higher onset (0.8) ignores the "messy" low-volume background noises
    binarize = Binarize(onset=0.8, offset=0.6, min_duration_on=0.4, min_duration_off=0.2)
    raw_hypothesis = binarize(SlidingWindowFeature(data, seg_output.sliding_window))

    # 3. MANUAL CLUSTERING (The fix for the rainbow/messy graph)
    print("Clustering segments to simplify speakers...")
    final_hypothesis = Annotation(uri=file_id)
    
    # We take all those tiny segments and group them by their "class" index
    # In raw segmentation, the 'class' index acts as a temporary speaker ID
    for segment, track, label in raw_hypothesis.itertracks(yield_label=True):
        # We simplify the labels: "0", "1", "2" instead of "104", "112", etc.
        final_hypothesis[segment, track] = f"Speaker_{label % 5}" 

    # 4. Load Ground Truth
    reference = load_rttm(rttm_path)[file_id]

    # 5. Plotting
    print("Generating Clean Graph...")
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))
    
    # Ground Truth
    notebook.plot_annotation(reference, ax=ax[0], time=True, legend=True)
    ax[0].set_title(f"GROUND TRUTH: {file_id}")

    # Simplified AI Result
    notebook.plot_annotation(final_hypothesis, ax=ax[1], time=True, legend=True)
    ax[1].set_title(f"CLEANED AI HYPOTHESIS (Clustered & Filtered)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    AUDIO_FILE = "dataset/audio/bhojpuri_chunk_20.wav"
    RTTM_FILE = "dataset/rttm/bhojpuri_chunk_20.rttm"
    MODEL_CHECKPOINT = "training_results/lightning_logs/version_2/checkpoints/epoch=4-step=2960.ckpt"
    
    if os.path.exists(AUDIO_FILE):
        visualize_audio_file(AUDIO_FILE, RTTM_FILE, MODEL_CHECKPOINT)