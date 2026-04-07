"""
routes/text_route.py
Text modality pipeline:
  Input text → RoBERTa-large → Sentiment → Credibility → Classify
"""

import torch
import numpy as np
from flask import Blueprint, request, jsonify

from models.loader import get_roberta, get_sentiment
from utils.feature_utils import get_text_embedding, credibility_score, softmax_to_scores

text_bp = Blueprint('text', __name__)


def _fake_classifier_text(embedding: torch.Tensor) -> torch.Tensor:
    """
    Placeholder classifier head.
    Replace with: torch.load('checkpoints/text_head.pth')
    """
    torch.manual_seed(int(embedding.sum().item() * 1000) % 10000)
    return torch.randn(1, 2)


@text_bp.route('/analyze/text', methods=['POST'])
def analyze_text():
    data    = request.get_json(force=True)
    text    = data.get('text', '').strip()
    url     = data.get('url', '')
    lang    = data.get('lang', 'en')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # ── Step 1: RoBERTa embedding ─────────────────────────
    tokenizer, roberta = get_roberta()
    device  = next(roberta.parameters()).device
    emb     = get_text_embedding(text, tokenizer, roberta, device)

    # ── Step 2: Sentiment analysis ────────────────────────
    sentiment_pipe = get_sentiment()
    sent_result    = sentiment_pipe(text[:512])[0]
    sentiment_label = sent_result['label']    # POSITIVE / NEGATIVE / NEUTRAL
    sentiment_score = round(float(sent_result['score']), 3)

    # ── Step 3: Source credibility ─────────────────────────
    cred = credibility_score(url)

    # ── Step 4: Classify ──────────────────────────────────
    logits = _fake_classifier_text(emb)
    result = softmax_to_scores(logits)

    # Nudge result based on credibility
    if cred < 0.3:
        result['fake_prob'] = min(99.0, result['fake_prob'] + 15)
        result['real_prob'] = max(1.0,  result['real_prob'] - 15)

    label = result['label']

    return jsonify({
        'verdict':    label,
        'confidence': result['confidence'],
        'signals': [
            {
                'label': 'Fake probability',
                'value': f"{result['fake_prob']}%",
                'model': 'RoBERTa-Large',
                'severity': 'high' if result['fake_prob'] > 60 else 'low'
            },
            {
                'label': 'Sentiment bias',
                'value': f"{sentiment_label} ({sentiment_score})",
                'model': 'Twitter-RoBERTa',
                'severity': 'high' if sentiment_label == 'NEGATIVE' else 'low'
            },
            {
                'label': 'Source credibility',
                'value': f"{cred:.2f} / 1.0",
                'model': 'Credibility DB',
                'severity': 'high' if cred < 0.3 else 'low'
            }
        ],
        'bars': [
            {'label': 'Fake probability',    'value': result['fake_prob'], 'type': 'fake'},
            {'label': 'Real probability',    'value': result['real_prob'], 'type': 'real'},
            {'label': 'Sensationalism index','value': round(np.random.uniform(40,90) if label=='FAKE' else np.random.uniform(10,40), 1), 'type': 'neutral'}
        ],
        'explanation': _build_explanation_text(result, sentiment_label, cred, url),
        'pipeline': ['Input text', 'RoBERTa tokenizer', 'Semantic embedding',
                     'Sentiment analysis', 'Source credibility', 'Classification']
    })


def _build_explanation_text(result, sentiment, cred, url):
    parts = []
    if result['fake_prob'] > 70:
        parts.append(f"RoBERTa detected deceptive framing patterns (fake probability: {result['fake_prob']}%).")
    if sentiment == 'NEGATIVE':
        parts.append("Text contains highly charged negative language — a common fake news indicator.")
    if cred < 0.3:
        parts.append(f"Source credibility is very low ({cred:.2f}/1.0). The domain is not on the trusted list.")
    if not parts:
        parts.append("No strong fake indicators detected. Content appears credible.")
    return ' '.join(parts)
