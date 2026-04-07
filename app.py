"""
CrossFakeNet — Multimodal Fake News Detection
Main Flask application entry point.
Run: python app.py
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os

from routes.text_route     import text_bp
from routes.audio_route    import audio_bp
from routes.image_route    import image_bp
from routes.imgcom_route   import imgcom_bp
from routes.video_route    import video_bp

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150 MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Register modality blueprints
app.register_blueprint(text_bp,   url_prefix='/api')
app.register_blueprint(audio_bp,  url_prefix='/api')
app.register_blueprint(image_bp,  url_prefix='/api')
app.register_blueprint(imgcom_bp, url_prefix='/api')
app.register_blueprint(video_bp,  url_prefix='/api')

@app.route('/')
def index():
    return render_template('index.html')

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Max size is 150MB.'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error. Check console for details.'}), 500

if __name__ == '__main__':
    print("\n╔══════════════════════════════════════╗")
    print("║   CrossFakeNet — Starting server...  ║")
    print("║   Open: http://127.0.0.1:5000        ║")
    print("╚══════════════════════════════════════╝\n")
   app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))