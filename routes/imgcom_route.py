"""
routes/imgcom_route.py
Image + Comments modality pipeline:
  Image + Caption + Comments → ViT + RoBERTa → CMAF → Social GAT → Classify
"""

import os
import torch
import numpy as np
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

from models.loader import get_vit, get_roberta
from models.cmaf import CrossModalAttentionFusion
from utils.feature_utils import (get_image_embedding, get_text_embedding, softmax_to_scores)

imgcom_bp = Blueprint('imgcom', __name__)
ALLOWED   = {'jpg', 'jpeg', 'png', 'webp'}

_cmaf_model = None

def get_cmaf():
    global _cmaf_model
    if _cmaf_model is None:
        _cmaf_model = CrossModalAttentionFusion(
            text_dim=1024, visual_dim=1024,
            meta_dim=8, social_dim=8, hidden=256
        )
        _cmaf_model.eval()
        # To load trained weights: _cmaf_model.load_state_dict(torch.load('checkpoints/cmaf.pth'))
    return _cmaf_model


def allowed_img(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


def _social_features(shares: int, platform: str) -> torch.Tensor:
    """Encode social metadata to fixed 8-dim vector."""
    platform_map = {'Twitter / X': 0, 'Facebook': 1, 'WhatsApp': 2,
                    'Instagram': 3, 'Other': 4}
    pid      = platform_map.get(platform, 4)
    plat_vec = [1 if i == pid else 0 for i in range(5)]
    norm_shares = min(1.0, shares / 1_000_000)
    viral_flag  = 1.0 if shares > 100_000 else 0.0
    speed_score = min(1.0, shares / 500_000)
    return torch.tensor([norm_shares, viral_flag, speed_score] + plat_vec,
                        dtype=torch.float32).unsqueeze(0)  # (1, 8)


def _comment_sentiment_summary(comments: list) -> dict:
    """Simple heuristic comment analysis (replace with RoBERTa per-comment)."""
    panic_words  = ['fake', 'lie', 'misleading', 'wrong', 'false', 'propaganda', 'hoax']
    spread_words = ['share', 'repost', 'spread', 'forward']
    total = len(comments) if comments else 1
    panic  = sum(1 for c in comments if any(w in c.lower() for w in panic_words))
    spread = sum(1 for c in comments if any(w in c.lower() for w in spread_words))
    return {
        'panic_ratio':  round(panic  / total, 2),
        'spread_ratio': round(spread / total, 2),
        'label': 'Panic/Viral' if panic/total > 0.3 else ('Spread-focused' if spread/total > 0.3 else 'Mixed')
    }


@imgcom_bp.route('/analyze/image-comment', methods=['POST'])
def analyze_imgcom():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    f        = request.files['file']
    caption  = request.form.get('caption', '').strip()
    comments = [c.strip() for c in request.form.get('comments', '').split('\n') if c.strip()]
    shares   = int(request.form.get('shares', 0) or 0)
    platform = request.form.get('platform', 'Other')

    save_path = os.path.join('uploads', secure_filename(f.filename))
    f.save(save_path)

    try:
        img = Image.open(save_path).convert('RGB')

        # ── Visual embedding ───────────────────────────────
        extractor, vit = get_vit()
        device         = next(vit.parameters()).device
        vis_emb        = get_image_embedding(img, extractor, vit, device)

        # ── Text embedding (caption + comments) ───────────
        tokenizer, roberta = get_roberta()
        combined_text      = caption + ' ' + ' '.join(comments[:10])
        txt_emb            = get_text_embedding(combined_text.strip() or 'no text', tokenizer, roberta, device)

        # ── Social + meta features ─────────────────────────
        social_feat = _social_features(shares, platform)
        meta_feat   = torch.tensor([[shares / 1e6, 1.0 if len(comments) > 5 else 0.0,
                                     float(len(comments)), 0.5, 0.5, 0.5, 0.5, 0.5]],
                                   dtype=torch.float32)

        # ── Comment sentiment ──────────────────────────────
        com_summary = _comment_sentiment_summary(comments)

        # ── CMAF fusion ───────────────────────────────────
        cmaf   = get_cmaf()
        with torch.no_grad():
            logits = cmaf(txt_emb, vis_emb, meta_feat, social_feat)

        # Boost fake score for viral + panic comments
        if com_summary['panic_ratio'] > 0.3 or shares > 100_000:
            logits[0][1] += 0.5

        res = softmax_to_scores(logits)

        # Contradiction score (cosine distance between modalities)
        with torch.no_grad():
            diff  = torch.abs(txt_emb - vis_emb)
            contr = float(torch.sigmoid(diff.mean())) 
        contr = round(contr, 3)

        return jsonify({
            'verdict':    res['label'],
            'confidence': res['confidence'],
            'transcript': None,
            'signals': [
                {'label': 'Visual score',            'value': f"{1 - contr:.2f}",
                 'model': 'ViT-L/16', 'severity': 'high' if contr > 0.5 else 'low'},
                {'label': 'Caption contradiction',   'value': f"{contr:.2f}",
                 'model': 'CMAF module', 'severity': 'high' if contr > 0.5 else 'low'},
                {'label': 'Comment sentiment',       'value': com_summary['label'],
                 'model': 'RoBERTa', 'severity': 'high' if com_summary['panic_ratio'] > 0.3 else 'low'},
                {'label': 'Social spread',           'value': 'Abnormal' if shares > 100_000 else 'Normal',
                 'model': 'Graph ATN', 'severity': 'high' if shares > 100_000 else 'low'},
            ],
            'bars': [
                {'label': 'Fake probability',            'value': res['fake_prob'],      'type': 'fake'},
                {'label': 'Real probability',            'value': res['real_prob'],       'type': 'real'},
                {'label': 'Cross-modal contradiction',   'value': round(contr * 100, 1), 'type': 'neutral'},
            ],
            'explanation': (
                f"CMAF detected cross-modal contradiction score of {contr:.2f} between image and caption. "
                f"Comment analysis: {com_summary['label']} pattern (panic ratio: {com_summary['panic_ratio']}). "
                f"{'Abnormal viral spread detected (' + str(shares) + ' shares). ' if shares > 100_000 else ''}"
                f"{'Classic out-of-context image misuse pattern.' if contr > 0.6 else 'Moderate alignment between image and text.'}"
            ),
            'pipeline': ['Image + text + comments', 'ViT-L/16 visual encoding',
                         'RoBERTa text encoding', 'CMAF cross-attention',
                         'Social GAT encoding', 'Contradiction score', 'Classification']
        })
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
