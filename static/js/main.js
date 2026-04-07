/* CrossFakeNet — main.js
   Handles modality switching, file uploads, API calls, and result rendering.
*/

// ── Pipeline step definitions ──────────────────────────────────────
const PIPELINES = {
  text: [
    { icon:'📝', label:'Input text' },
    { icon:'🔤', label:'RoBERTa' },
    { icon:'🎭', label:'Sentiment' },
    { icon:'🔍', label:'Credibility' },
    { icon:'⚖️', label:'Classify' }
  ],
  audio: [
    { icon:'🎙️', label:'Audio input' },
    { icon:'🗣️', label:'Whisper ASR' },
    { icon:'🔊', label:'Wav2Vec2' },
    { icon:'🔤', label:'RoBERTa' },
    { icon:'⚖️', label:'Classify' }
  ],
  image: [
    { icon:'🖼️', label:'Image input' },
    { icon:'👁️', label:'ViT-L/16' },
    { icon:'🔗', label:'CLIP align' },
    { icon:'🗺️', label:'GradCAM' },
    { icon:'⚖️', label:'Classify' }
  ],
  'image-comment': [
    { icon:'🖼️', label:'Image+text' },
    { icon:'👁️', label:'ViT-L/16' },
    { icon:'🔤', label:'RoBERTa' },
    { icon:'🔀', label:'CMAF' },
    { icon:'🕸️', label:'Social GAT' },
    { icon:'⚖️', label:'Classify' }
  ],
  video: [
    { icon:'🎬', label:'Video input' },
    { icon:'👁️', label:'ViT frames' },
    { icon:'🗣️', label:'Whisper' },
    { icon:'🔊', label:'Wav2Vec2' },
    { icon:'🔀', label:'BiLSTM-GCN' },
    { icon:'⚖️', label:'Classify' }
  ]
};

// ── Progress messages per modality ─────────────────────────────────
const PROGRESS = {
  text: [
    'Tokenizing input with RoBERTa tokenizer...',
    'Extracting semantic embeddings...',
    'Running sentiment & emotion analysis...',
    'Scoring source credibility signals...',
    'Computing final classification...'
  ],
  audio: [
    'Loading audio stream...',
    'Transcribing speech with Whisper ASR...',
    'Extracting prosody features via Wav2Vec2...',
    'Encoding transcript with RoBERTa-large...',
    'Fusing acoustic + textual signals...',
    'Computing final classification...'
  ],
  image: [
    'Loading and preprocessing image...',
    'Extracting patch embeddings via ViT-L/16...',
    'Running CLIP image-text alignment check...',
    'Generating GradCAM saliency map...',
    'Detecting manipulation artifacts...',
    'Computing final classification...'
  ],
  'image-comment': [
    'Preprocessing image and caption...',
    'Encoding visual features via ViT-L/16...',
    'Encoding text & comments via RoBERTa...',
    'Running Cross-Modal Attention Fusion (CMAF)...',
    'Encoding social signals via Graph ATN...',
    'Computing contradiction score...',
    'Computing final classification...'
  ],
  video: [
    'Extracting keyframes from video...',
    'Encoding frames with ViT-L/16...',
    'Transcribing audio via Whisper ASR...',
    'Extracting prosody via Wav2Vec2...',
    'Running BiLSTM temporal encoder...',
    'Running GCN spatial encoder...',
    'Running CMAF cross-modal fusion...',
    'Computing final classification...'
  ]
};

// ── State ─────────────────────────────────────────────────────────
let currentModal = 'text';

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  Object.keys(PIPELINES).forEach(renderPipeline);
});

// ── Switch modality ───────────────────────────────────────────────
function switchModal(mod) {
  document.querySelectorAll('.mod-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + mod).classList.add('active');
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('visible'));
  document.getElementById('panel-' + mod).classList.add('visible');
  currentModal = mod;

  // Reset UI
  document.getElementById('result-card').style.display  = 'none';
  document.getElementById('progress-wrap').classList.remove('show');
  document.getElementById('error-box').style.display    = 'none';
}

// ── Render pipeline strip ─────────────────────────────────────────
function renderPipeline(mod) {
  const steps = PIPELINES[mod];
  const el    = document.getElementById('pipe-' + mod);
  if (!el) return;
  el.innerHTML = steps.map((s, i) => `
    <div class="pipe-step active">
      <div class="pipe-dot">${s.icon}</div>
      <div class="pipe-label">${s.label}</div>
    </div>
    ${i < steps.length - 1 ? '<div class="pipe-arrow"></div>' : ''}
  `).join('');
}

// ── File upload helpers ───────────────────────────────────────────
function triggerUpload(id) {
  document.getElementById(id).click();
}

function showFile(type) {
  const fileMap = { audio:'audio-file', image:'image-file', imgcom:'imgcom-file', video:'video-file' };
  const zoneMap = { audio:'zone-audio',  image:'zone-image',  imgcom:'zone-imgcom',  video:'zone-video' };
  const icons   = { audio:'🎙️', image:'🖼️', imgcom:'💬', video:'🎬' };

  const file = document.getElementById(fileMap[type])?.files[0];
  if (!file) return;
  const zone = document.getElementById(zoneMap[type]);
  zone.innerHTML = `
    <div class="big">${icons[type]}</div>
    <div class="filename">${file.name}</div>
    <p>${(file.size / (1024 * 1024)).toFixed(2)} MB · ready to analyze</p>
  `;
}

// ── Run analysis (real API call) ──────────────────────────────────
async function runAnalysis(mod) {
  const btn = document.querySelector(`#panel-${mod} .analyze-btn`);
  btn.disabled    = true;
  btn.textContent = 'Analyzing…';

  document.getElementById('result-card').style.display = 'none';
  document.getElementById('error-box').style.display   = 'none';

  // Show progress
  const wrap    = document.getElementById('progress-wrap');
  const stepsEl = document.getElementById('progress-steps');
  wrap.classList.add('show');

  const steps = PROGRESS[mod];
  stepsEl.innerHTML = steps.map((s, i) =>
    `<div class="prog-step" id="pstep-${i}"><div class="prog-dot"></div><span>${s}</span></div>`
  ).join('');

  // Animate steps while request runs
  let stepIdx = 0;
  const stepInterval = setInterval(() => {
    if (stepIdx > 0) {
      const prev = document.getElementById(`pstep-${stepIdx - 1}`);
      if (prev) { prev.classList.remove('running'); prev.classList.add('done'); }
    }
    const cur = document.getElementById(`pstep-${stepIdx}`);
    if (cur) cur.classList.add('running');
    stepIdx++;
    if (stepIdx >= steps.length) clearInterval(stepInterval);
  }, 600);

  try {
    // Build request payload
    let response;

    if (mod === 'text') {
      response = await fetch('/api/analyze/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: document.getElementById('text-input').value,
          url:  document.getElementById('text-url').value,
          lang: document.getElementById('text-lang').value,
        })
      });

    } else if (mod === 'audio') {
      const fd = new FormData();
      const f  = document.getElementById('audio-file').files[0];
      if (!f) throw new Error('Please upload an audio file first.');
      fd.append('file',  f);
      fd.append('title', document.getElementById('audio-title').value);
      response = await fetch('/api/analyze/audio', { method:'POST', body:fd });

    } else if (mod === 'image') {
      const fd = new FormData();
      const f  = document.getElementById('image-file').files[0];
      if (!f) throw new Error('Please upload an image first.');
      fd.append('file',    f);
      fd.append('caption', document.getElementById('image-caption').value);
      response = await fetch('/api/analyze/image', { method:'POST', body:fd });

    } else if (mod === 'image-comment') {
      const fd = new FormData();
      const f  = document.getElementById('imgcom-file').files[0];
      if (!f) throw new Error('Please upload an image first.');
      fd.append('file',     f);
      fd.append('caption',  document.getElementById('imgcom-caption').value);
      fd.append('comments', document.getElementById('imgcom-comments').value);
      fd.append('shares',   document.getElementById('imgcom-shares').value || '0');
      fd.append('platform', document.getElementById('imgcom-platform').value);
      response = await fetch('/api/analyze/image-comment', { method:'POST', body:fd });

    } else if (mod === 'video') {
      const fd = new FormData();
      const f  = document.getElementById('video-file').files[0];
      if (!f) throw new Error('Please upload a video file first.');
      fd.append('file',   f);
      fd.append('title',  document.getElementById('video-title').value);
      fd.append('source', document.getElementById('video-source').value);
      fd.append('views',  document.getElementById('video-views').value  || '0');
      fd.append('shares', document.getElementById('video-shares').value || '0');
      response = await fetch('/api/analyze/video', { method:'POST', body:fd });
    }

    clearInterval(stepInterval);

    // Mark remaining steps as done
    for (let i = 0; i < steps.length; i++) {
      const el = document.getElementById(`pstep-${i}`);
      if (el) { el.classList.remove('running'); el.classList.add('done'); }
    }

    await sleep(400);
    wrap.classList.remove('show');

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || 'Server error');
    }

    const data = await response.json();
    renderResult(data, mod);

  } catch (err) {
    clearInterval(stepInterval);
    wrap.classList.remove('show');
    const eb = document.getElementById('error-box');
    eb.style.display  = 'block';
    eb.textContent    = '⚠ ' + err.message;
  } finally {
    btn.disabled    = false;
    const labels    = { text:'Text', audio:'Audio', image:'Image', 'image-comment':'Image + Comments', video:'Video' };
    btn.textContent = `Analyze ${labels[mod]} →`;
  }
}

// ── Render result card ────────────────────────────────────────────
function renderResult(data, mod) {
  const label   = data.verdict;   // 'FAKE' | 'REAL' | 'UNCERTAIN'
  const conf    = data.confidence;

  // Verdict
  const vEl = document.getElementById('verdict-text');
  vEl.textContent = label;
  vEl.className   = `verdict ${label}`;

  // Confidence pill
  const pill = document.getElementById('conf-pill');
  pill.textContent = `Confidence: ${conf}%`;
  pill.className   = `confidence-pill pill-${label}`;

  // Signal cards
  const sigGrid = document.getElementById('signal-grid');
  sigGrid.innerHTML = (data.signals || []).map(s => `
    <div class="signal-card">
      <div class="sig-label">${s.label}</div>
      <div class="sig-val ${s.severity === 'high' ? 'sig-high' : 'sig-low'}">${s.value}</div>
      <div class="sig-model">${s.model}</div>
    </div>
  `).join('');

  // Bars
  const barsEl = document.getElementById('bars-section');
  barsEl.innerHTML = (data.bars || []).map(b => `
    <div class="bar-wrap">
      <div class="bar-label"><span>${b.label}</span><span>${b.value}%</span></div>
      <div class="bar-track">
        <div class="bar-fill bar-${b.type}" style="width:0" data-target="${b.value}%"></div>
      </div>
    </div>
  `).join('');

  // Transcript (if present)
  const transEl = document.getElementById('transcript-section');
  if (data.transcript) {
    transEl.innerHTML = `
      <div class="transcript-box" style="margin-bottom:16px">
        <strong>Transcript:</strong> ${data.transcript}
      </div>`;
  } else {
    transEl.innerHTML = '';
  }

  // Explanation
  document.getElementById('explanation-box').innerHTML = data.explanation || '';

  // Footer
  const mNames = { text:'Text', audio:'Audio', image:'Image', 'image-comment':'Image+Comments', video:'Video' };
  document.getElementById('footer-text').textContent =
    `CrossFakeNet · ${mNames[mod]} analysis · ${new Date().toLocaleTimeString()}`;

  // Show card
  const card = document.getElementById('result-card');
  card.style.display = 'block';

  // Animate bars
  setTimeout(() => {
    card.querySelectorAll('.bar-fill').forEach(b => { b.style.width = b.dataset.target; });
  }, 100);

  // Scroll to result
  card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Utility ───────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
