"""
models/loader.py
Centralized lazy model loader.
Models are loaded once on first use and cached in memory.
"""

import torch
import whisper
from transformers import (
    RobertaTokenizer, RobertaModel,
    ViTImageProcessor, ViTModel,
    pipeline as hf_pipeline
)

_cache = {}

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# ── RoBERTa ──────────────────────────────────────────────
def get_roberta():
    if 'roberta' not in _cache:
        print("[loader] Loading RoBERTa-large...")
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model     = RobertaModel.from_pretrained('roberta-large').to(get_device())
        model.eval()
        _cache['roberta'] = (tokenizer, model)
    return _cache['roberta']

# ── ViT ──────────────────────────────────────────────────
def get_vit():
    if 'vit' not in _cache:
        print("[loader] Loading ViT-L/16...")
        extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
        model     = ViTModel.from_pretrained('google/vit-large-patch16-224').to(get_device())
        model.eval()
        _cache['vit'] = (extractor, model)
    return _cache['vit']

# ── Whisper ───────────────────────────────────────────────
def get_whisper():
    if 'whisper' not in _cache:
        print("[loader] Loading Whisper base...")
        _cache['whisper'] = whisper.load_model('base')
    return _cache['whisper']

# ── Sentiment pipeline ────────────────────────────────────
def get_sentiment():
    if 'sentiment' not in _cache:
        print("[loader] Loading sentiment pipeline...")
        _cache['sentiment'] = hf_pipeline(
            'sentiment-analysis',
            model='cardiffnlp/twitter-roberta-base-sentiment-latest',
            device=0 if get_device() == 'cuda' else -1
        )
    return _cache['sentiment']
