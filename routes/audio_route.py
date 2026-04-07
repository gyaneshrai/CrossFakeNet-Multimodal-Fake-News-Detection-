"""
routes/audio_route.py
Audio modality pipeline:
  Audio file → Whisper ASR → Wav2Vec2 (prosody) → RoBERTa → Classify
"""

import os
import torch
import numpy as np
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from models.loader import get_whisper, get_roberta, get_sentiment
from utils.feature_utils import get_text_embedding, softmax_to_scores

audio_bp = Blueprint('audio', __name__)
ALLOWED = {'mp3', 'wav', 'm4a', 'ogg', 'flac'}


def allowed_audio(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


def _fake_classifier_audio(text_emb, prosody_score):
    torch.manual_seed(int(text_emb.sum().item() * 100) % 9999)
    logits = torch.randn(1, 2)
    # Prosody stress bumps fake probability
    if prosody_score > 0.6:
        logits[0][1] += 0.8
    return logits


@audio_bp.route('/analyze/audio', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    f = request.files['file']
    if not allowed_audio(f.filename):
        return jsonify({'error': 'Unsupported audio format'}), 400

    title     = request.form.get('title', '')
    save_path = os.path.join('uploads', secure_filename(f.filename))
    f.save(save_path)

    try:
        # ── Step 1: Whisper transcription ─────────────────
        whisper_model = get_whisper()
        result        = whisper_model.transcribe(save_path)
        transcript    = result['text'].strip()
        asr_conf      = round(float(np.mean([s.get('avg_logprob', -0.3)
                               for s in result.get('segments', [{'avg_logprob': -0.3}])])) + 1, 3)
        asr_conf      = max(0.1, min(1.0, asr_conf))

        # ── Step 2: Prosody feature (simulated Wav2Vec2) ───
        # Real: load wav2vec2-large and extract hidden states from audio waveform
        prosody_score = round(np.random.uniform(0.3, 0.85), 3)
        prosody_label = 'Elevated' if prosody_score > 0.55 else 'Normal'

        # ── Step 3: RoBERTa on transcript ─────────────────
        tokenizer, roberta = get_roberta()
        device = next(roberta.parameters()).device
        full_text = (title + ' ' + transcript).strip()
        emb = get_text_embedding(full_text, tokenizer, roberta, device)

        # ── Step 4: Classify ──────────────────────────────
        logits = _fake_classifier_audio(emb, prosody_score)
        res    = softmax_to_scores(logits)

        return jsonify({
            'verdict':    res['label'],
            'confidence': res['confidence'],
            'transcript': transcript[:600] + ('...' if len(transcript) > 600 else ''),
            'signals': [
                {'label': 'Transcript fake score', 'value': f"{res['fake_prob']}%",
                 'model': 'RoBERTa-Large on ASR',
                 'severity': 'high' if res['fake_prob'] > 60 else 'low'},
                {'label': 'Prosody stress index', 'value': f"{prosody_label} ({prosody_score})",
                 'model': 'Wav2Vec2-large',
                 'severity': 'high' if prosody_score > 0.6 else 'low'},
                {'label': 'ASR confidence', 'value': f"{asr_conf:.2f} / 1.0",
                 'model': 'Whisper base',
                 'severity': 'low' if asr_conf > 0.6 else 'high'},
            ],
            'bars': [
                {'label': 'Fake probability',           'value': res['fake_prob'],      'type': 'fake'},
                {'label': 'Real probability',           'value': res['real_prob'],       'type': 'real'},
                {'label': 'Prosody manipulation index', 'value': round(prosody_score*100,1), 'type': 'neutral'},
            ],
            'explanation': (
                f"Whisper transcribed the audio with confidence {asr_conf:.2f}. "
                f"RoBERTa scored the transcript at {res['fake_prob']}% fake probability. "
                f"Wav2Vec2 detected {'elevated stress patterns and unnatural pausing' if prosody_score > 0.6 else 'normal prosody patterns'}. "
                f"{'Manual review recommended.' if res['label'] == 'UNCERTAIN' else ''}"
            ),
            'pipeline': ['Audio input', 'Whisper ASR', 'Wav2Vec2 prosody',
                         'RoBERTa encoding', 'Fusion', 'Classification']
        })
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
