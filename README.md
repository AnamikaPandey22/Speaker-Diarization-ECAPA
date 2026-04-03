🎙️ Speaker Diarization using Fine-Tuned PyAnnote Segmentation

📌 Project Overview

This project implements speaker diarization — answering the question:

“Who spoke when?”

The system is built using PyAnnote Audio, where a speaker segmentation model is fine-tuned on custom audio data, followed by speaker embedding extraction and clustering to identify different speakers in an audio file.

This project is suitable for:

1. Call analysis
2. Meeting transcription
3. Surveillance / forensics
4. Conversational AI pipelines
5. Academic research & viva evaluation

🧠 What is Speaker Diarization?

Speaker diarization breaks an audio file into:

Speech vs non-speech

Speaker segments

Speaker labels (Speaker 1, Speaker 2, …)

Example output:

00:00–00:12  → Speaker 1

00:12–00:20  → Speaker 2

00:20–00:35  → Speaker 1

🏗️ Project Architecture

Audio Input
   ↓
Preprocessing (Mono + 16kHz)
   ↓
Fine-Tuned Segmentation Model (PyAnnote)
   ↓
Speech Activity Detection
   ↓
Speaker Embedding Extraction (Speech Brain)
   ↓
Agglomerative Clustering
   ↓
Final Speaker Diarization Output (RTTM / Timeline)

🔧 Technologies & Libraries Used

Python 3.9+

PyTorch

PyAnnote Audio

Torchaudio

Scikit-learn

NumPy

Matplotlib

RTTM format for evaluation

📁 Project Structure

├── dataset/

│   ├── audio/                 # Training & evaluation audio files

│   ├── rttm/                  # Ground truth RTTM files

│   └── database.yml           # PyAnnote database configuration

│

├── models/

│   └── segmentation/          # Fine-tuned segmentation checkpoints

│

├── scripts/

│   ├── train_segmentation.py  # Model fine-tuning script

│   ├── diarization.py         # Inference + clustering

│   └── visualize.py           # Diarization visualization

│

├── outputs/

│   ├── rttm/                  # Predicted RTTM files

│   └── plots/                 # Speaker timeline plots

│

├── requirements.txt

├── .gitignore

└── README.md

🎯 Key Features

✅ Fine-tuned speaker segmentation model

✅ Works on real-world conversational audio

✅ Supports variable number of speakers

✅ Uses Agglomerative Clustering

✅ Produces RTTM diarization output

✅ Visualization of speaker segments

🔊 Audio Requirements (Important)

All audio must be:

Mono (1 channel)

16 kHz sampling rate

WAV format

📌 Why?

PyAnnote models are trained on mono 16kHz

Stereo or different sampling rates reduce accuracy

🚀 How to Run the Project

1️⃣ Create Virtual Environment

python -m venv venv

venv\Scripts\activate   # Windows

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Prepare Dataset

Place audio files in dataset/audio/

Place RTTM files in dataset/rttm/

Configure paths in dataset/database.yml

4️⃣ Fine-Tune Segmentation Model

python scripts/train_segmentation.py

5️⃣ Run Speaker Diarization

python scripts/diarization.py --audio sample.wav

6️⃣ Visualize Output

python scripts/visualize.py

📊 Output Formats

RTTM file for evaluation

Speaker timeline plots

Segment-level speaker annotations

📈 Evaluation Metrics

Diarization Error Rate (DER)

False Alarm

Missed Speech

Speaker Confusion

🧪 Model Details

Base Model: PyAnnote Speaker Segmentation

Training: Supervised fine-tuning

Loss Function: Binary cross-entropy (segmentation)

Clustering: Agglomerative Hierarchical Clustering

Embeddings: Speaker embeddings from PyAnnote pipeline

⚠️ Limitations

Overlapping speech is challenging

Performance depends on audio quality

Needs sufficient labeled RTTM data


🌱 Future Improvements

🔹 Overlap-aware diarization

🔹 Domain-specific embedding fine-tuning

🔹 Real-time diarization

🔹 Integration with ASR (speech-to-text)

👩‍💻 Author

Anamika Pandey

BCA | AI/ML | Speaker Diarization | PyAnnote

📜 License

This project is for academic and learning purposes.

Pre-trained models belong to their respective authors.

🙌 Acknowledgements

PyAnnote Audio Team

HuggingFace

PyTorch Community
