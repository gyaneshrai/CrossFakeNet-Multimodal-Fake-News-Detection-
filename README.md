<<<<<<< HEAD
# CrossFakeNet — Multimodal Fake News Detection
### SRM Institute of Science and Technology | B.Tech CSE Project

A web-based multimodal fake news detection system supporting **5 input modalities**:
Text · Audio · Image · Image + Comments · Video

Each modality routes through a dedicated AI pipeline. The system runs locally on your computer.

---

## Project Structure

```
crossfakenet/
├── app.py                  ← Flask app entry point (run this)
├── requirements.txt        ← All Python dependencies
├── setup.sh                ← One-time setup script (Mac/Linux)
├── setup.bat               ← One-time setup script (Windows)
│
├── models/
│   ├── loader.py           ← Lazy loads RoBERTa, ViT, Whisper (cached after first use)
│   └── cmaf.py             ← Cross-Modal Attention Fusion neural network
│
├── routes/
│   ├── text_route.py       ← /api/analyze/text  → RoBERTa + Sentiment + Credibility
│   ├── audio_route.py      ← /api/analyze/audio → Whisper ASR + Wav2Vec2 + RoBERTa
│   ├── image_route.py      ← /api/analyze/image → ViT-L/16 + CLIP + GradCAM
│   ├── imgcom_route.py     ← /api/analyze/image-comment → CMAF + Social GAT
│   └── video_route.py      ← /api/analyze/video → BiLSTM-GCN + CMAF full pipeline
│
├── utils/
│   └── feature_utils.py    ← Shared helpers: frame extraction, embeddings, scoring
│
├── static/
│   ├── css/style.css       ← Dark-theme UI styles
│   └── js/main.js          ← Frontend logic, API calls, result rendering
│
├── templates/
│   └── index.html          ← Main web page (served by Flask)
│
└── uploads/                ← Temp folder for uploaded files (auto-created)
```

---

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- pip
- At least 6 GB RAM (models are large)
- GPU optional but speeds things up

### Step 1 — Place the project folder

Put the `crossfakenet/` folder anywhere on your computer.

### Step 2 — Run setup (one time only)

**Windows:**
```cmd
cd crossfakenet
setup.bat
```

**Mac / Linux:**
```bash
cd crossfakenet
bash setup.sh
```

This creates a virtual environment and installs all dependencies (~2–5 min).

### Step 3 — Start the server

**Windows:**
```cmd
venv\Scripts\activate
python app.py
```

**Mac / Linux:**
```bash
source venv/bin/activate
python app.py
```

### Step 4 — Open in browser

```
http://127.0.0.1:5000
```

---

## How It Works

| Modality | Models Used | What's Detected |
|---|---|---|
| **Text** | RoBERTa-Large → Sentiment → Credibility | Deceptive framing, emotional manipulation |
| **Audio** | Whisper ASR → Wav2Vec2 → RoBERTa | Synthetic voice, unnatural prosody |
| **Image** | ViT-L/16 → CLIP → GradCAM | Splicing, manipulation artifacts |
| **Image + Comments** | ViT + RoBERTa → CMAF → Social GAT | Out-of-context image, viral panic patterns |
| **Video** | ViT + Whisper + Wav2Vec2 → BiLSTM-GCN → CMAF | Deepfakes, audio-visual mismatch |

---

## First Run Note

On the **very first run**, the app downloads pretrained models from HuggingFace:
- `roberta-large` (~1.4 GB)
- `google/vit-large-patch16-224` (~1.2 GB)
- `whisper base` (~145 MB)
- `cardiffnlp/twitter-roberta-base-sentiment-latest` (~500 MB)

Models are cached in `~/.cache/huggingface/` after first download.
Subsequent runs start much faster.

---

## Demo Mode vs Real Mode

The current routes use **placeholder classifier heads** to demonstrate the full pipeline
without requiring trained model checkpoints.

To use real trained weights:
1. Train the CMAF model using your dataset
2. Save: `torch.save(model.state_dict(), 'checkpoints/cmaf.pth')`
3. Uncomment the load line in `routes/imgcom_route.py` and `routes/video_route.py`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure venv is activated |
| `Out of memory` | Close other apps; CPU mode is default if no GPU |
| Whisper slow on CPU | Normal — base model takes ~20–30s per audio file |
| Port 5000 in use | Change port in `app.py`: `app.run(port=5001)` |
| `torch` install fails | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
=======
# CrossFakeNet-Multimodal-Fake-News-Detection-
A multimodal fake news detection system using RoBERTa (text), ViT (image), and Whisper (audio) with a Flask web interface.
>>>>>>> e22468527851d7ffaa8cb8ab036068640c4503bd
