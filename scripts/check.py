#It is only checking whether your computer has the “Segmentation” thing installed or not.
#Do you have something called Segmentation inside pyannote.audio?
#Segmentation is just a class name (a tool), not a model, not training.

try:
    from pyannote.audio.tasks import Segmentation
    print("Success! Segmentation task imported.")
except ImportError as e:
    print(f"Still failing: {e}")