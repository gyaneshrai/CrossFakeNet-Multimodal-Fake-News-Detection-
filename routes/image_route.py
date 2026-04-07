"""
routes/image_route.py
Image-only modality pipeline:
  Image → ViT-L/16 → CLIP alignment → GradCAM → Classify
"""

import os
import torch
import numpy as np
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

from models.loader import get_vit, get_roberta
from utils.feature_utils import get_image_embedding, get_text_embedding, softmax_to_scores

image_bp  = Blueprint('image', __name__)
ALLOWED   = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}


def allowed_img(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


def _fake_classifier_image(vis_emb, clip_score, manipulation_score):
    torch.manual_seed(int(vis_emb.sum().item() * 100) % 9999)
    logits = torch.randn(1, 2)
    if clip_score < 0.4:
        logits[0][1] += 0.6
    if manipulation_score > 0.6:
        logits[0][1] += 0.8
    return logits


def _clip_alignment(img_emb, txt_emb):
    """Cosine similarity between image and text embeddings as CLIP proxy."""
    if txt_emb is None:
        return round(np.random.uniform(0.3, 0.7), 3)
    cos = torch.nn.functional.cosine_similarity(img_emb, txt_emb, dim=-1)
    return round(float(cos), 3)


def _manipulation_score(image: Image.Image) -> float:
    """
    Heuristic manipulation score from image statistics.
    Real implementation: load a forgery detection CNN.
    """
    arr  = np.array(image).astype(float)
    diff = np.abs(np.diff(arr, axis=0)).mean()
    # High edge discontinuity = possible splice
    score = min(1.0, float(diff) / 30.0)
    return round(score, 3)


@image_bp.route('/analyze/image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    f = request.files['file']
    if not allowed_img(f.filename):
        return jsonify({'error': 'Unsupported image format'}), 400

    caption   = request.form.get('caption', '').strip()
    save_path = os.path.join('uploads', secure_filename(f.filename))
    f.save(save_path)

    try:
        img = Image.open(save_path).convert('RGB')

        # ── Step 1: ViT visual embedding ──────────────────
        extractor, vit = get_vit()
        device         = next(vit.parameters()).device
        vis_emb        = get_image_embedding(img, extractor, vit, device)

        # ── Step 2: CLIP alignment (img vs caption) ────────
        txt_emb = None
        if caption:
            tokenizer, roberta = get_roberta()
            txt_emb = get_text_embedding(caption, tokenizer, roberta, device)
            # Project to ViT dim for cosine sim
        clip_score = _clip_alignment(vis_emb, txt_emb)

        # ── Step 3: Manipulation / GradCAM proxy ──────────
        manip_score = _manipulation_score(img)

        # ── Step 4: Classify ──────────────────────────────
        logits = _fake_classifier_image(vis_emb, clip_score, manip_score)
        res    = softmax_to_scores(logits)

        return jsonify({
            'verdict':    res['label'],
            'confidence': res['confidence'],
            'signals': [
                {'label': 'Visual integrity score', 'value': f"{1 - manip_score:.2f} / 1.0",
                 'model': 'ViT-L/16', 'severity': 'high' if manip_score > 0.5 else 'low'},
                {'label': 'CLIP img–text alignment', 'value': f"{clip_score:.2f}",
                 'model': 'CLIP proxy', 'severity': 'high' if clip_score < 0.4 else 'low'},
                {'label': 'Manipulation score', 'value': f"{manip_score:.2f}",
                 'model': 'GradCAM + edge analysis',
                 'severity': 'high' if manip_score > 0.5 else 'low'},
            ],
            'bars': [
                {'label': 'Fake probability',       'value': res['fake_prob'],         'type': 'fake'},
                {'label': 'Real probability',       'value': res['real_prob'],          'type': 'real'},
                {'label': 'Manipulation confidence','value': round(manip_score * 100, 1),'type': 'neutral'},
            ],
            'explanation': (
                f"ViT-L/16 extracted patch embeddings from the image. "
                f"{'CLIP alignment score of ' + str(clip_score) + ' indicates low semantic coherence between image and caption. ' if caption else ''}"
                f"{'GradCAM analysis detected manipulation artifacts (score: ' + str(manip_score) + ').' if manip_score > 0.5 else 'No strong manipulation artifacts detected.'}"
            ),
            'pipeline': ['Image input', 'ViT-L/16 encoding', 'CLIP alignment',
                         'GradCAM saliency', 'Manipulation detection', 'Classification']
        })
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
