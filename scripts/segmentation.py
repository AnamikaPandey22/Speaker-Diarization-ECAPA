

import os
import torch
import torchaudio
import torch.serialization
from pyannote.core import Segment, Timeline

# --- 1. MONKEY PATCH (Fixes PyTorch 2.6 Security Error) ---
original_load = torch.serialization.load
def forced_load(f, map_location=None, pickle_module=None, **kwargs):
    kwargs['weights_only'] = False
    return original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
torch.load = forced_load
torch.serialization.load = forced_load
# ---------------------------------------------------------

#Model : neural network, Segmentation : training logic, get_protocol : dataset loader, pl.Trainer : training engine

from pyannote.audio import Model
from pyannote.audio.tasks import Segmentation
from pyannote.database import get_protocol, FileFinder
import pytorch_lightning as pl

os.environ["PYANNOTE_DATABASE_CONFIG"] = "database.yml"

def train_segmentation():
    # 2. PREPROCESSORS
    def get_annotated(file):
        info = torchaudio.info(file["audio"])
        # Calculate duration: total frames / sample rate
        duration = info.num_frames / info.sample_rate
        # Return the 'Timeline' object the library is looking for
        return Timeline([Segment(0, duration)])

    preprocessors = {
        "audio": FileFinder(),
        "annotated": get_annotated,
    }

    # 3. LOAD PROTOCOL
    print("Loading Hindi-Bhojpuri Protocol...")
    protocol = get_protocol(
        'HindiBhojpuri.SpeakerDiarization.Segmentation', 
        preprocessors=preprocessors
    )

    # 4. SETUP TASK
    seg_task = Segmentation(
        protocol, 
        duration=2.0, 
        batch_size=4, 
        num_workers=0 
    )

    # 5. LOAD MODEL - Start from an English-trained segmentation model, and adapt it to Hindi/Bhojpuri.” This is transfer learning, not training from scratch.
    print("Attempting to load model...")
    model = Model.from_pretrained("pyannote/segmentation-3.0")
    model.task = seg_task

    # 6. TRAINER
    trainer = pl.Trainer(
        accelerator="cpu", 
        max_epochs=5,
        default_root_dir="training_results"
    )

    # 7. START
    print("--- Starting Fine-tuning ---")
    trainer.fit(model)

if __name__ == "__main__":
    train_segmentation()