"""
utils/feature_utils.py
Shared helpers for feature extraction across all modalities.
"""

import torch
import numpy as np
import cv2
from PIL import Image


def get_text_embedding(text: str, tokenizer, model, device='cpu') -> torch.Tensor:
    """RoBERTa CLS-token embedding for input text."""
    inputs = tokenizer(
        text, return_tensors='pt',
        truncation=True, max_length=512, padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]   # (1, 1024)


def get_image_embedding(image: Image.Image, extractor, model, device='cpu') -> torch.Tensor:
    """ViT CLS-token embedding for a PIL image."""
    inputs = extractor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]   # (1, 1024)


def extract_frames(video_path: str, num_frames: int = 8) -> list:
    """Sample `num_frames` evenly from a video. Returns list of PIL Images."""
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total  = max(total, 1)
    idxs   = [int(i * total / num_frames) for i in range(num_frames)]
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


def get_video_visual_embedding(frames: list, extractor, model, device='cpu') -> torch.Tensor:
    """Average ViT embedding across sampled frames."""
    if not frames:
        return torch.zeros(1, 1024)
    inputs = extractor(images=frames, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean-pool across frames
    return outputs.last_hidden_state[:, 0, :].mean(dim=0, keepdim=True)


def softmax_to_scores(logits: torch.Tensor) -> dict:
    """Convert raw logits to real/fake probability dict."""
    probs = torch.softmax(logits, dim=-1)[0]
    real_p = float(probs[0]) * 100
    fake_p = float(probs[1]) * 100
    label  = 'FAKE' if fake_p >= 60 else ('REAL' if real_p >= 60 else 'UNCERTAIN')
    conf   = max(real_p, fake_p)
    return {
        'label':      label,
        'confidence': round(conf, 1),
        'real_prob':  round(real_p, 1),
        'fake_prob':  round(fake_p, 1),
    }


def credibility_score(url: str) -> float:
    """
    Heuristic source credibility score [0–1].
    Replace with a real database lookup for production.
    """
    if not url:
        return 0.5
    trusted = ['ndtv', 'bbc', 'reuters', 'apnews', 'thehindu',
                'indiatoday', 'hindustantimes', 'nytimes', 'theguardian']
    url_lower = url.lower()
    for t in trusted:
        if t in url_lower:
            return round(0.80 + np.random.uniform(0, 0.15), 2)
    return round(np.random.uniform(0.05, 0.4), 2)
