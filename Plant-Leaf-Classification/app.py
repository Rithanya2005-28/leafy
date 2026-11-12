import base64
import hashlib
import json
import os

import cv2
import numpy as np
from flask import (Flask, jsonify, render_template, request, send_from_directory)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from compiler.command_parser import parse_command
from image_processing.preprocessing import preprocess_image, processed_preview
from model_loader import load_model
from security.image_security import verify_image_integrity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'leaf_classifier.keras')
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.json')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
LOW_CONFIDENCE = 0.50

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model / labels loaded once at startup
# ---------------------------------------------------------------------------
MODEL_ERROR = None
model = None
CLASS_NAMES = []

try:
    model = load_model(MODEL_PATH)
except Exception as exc:  # missing model, missing TensorFlow, corrupt file...
    MODEL_ERROR = str(exc)

if os.path.isfile(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, encoding='utf-8') as handle:
        CLASS_NAMES = json.load(handle)
elif os.path.isdir(DATASET_DIR):
    CLASS_NAMES = sorted(
        entry for entry in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, entry))
    )


def display_name(label):
    return label.replace('_', ' ')


def allowed_file(filename):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def read_metrics():
    if not os.path.isfile(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def decode_upload(file_storage):
    """Save the upload and make sure it is a real, readable image."""
    filename = secure_filename(file_storage.filename)
    if not filename or not allowed_file(filename):
        raise ValueError('Invalid file type. Please upload a JPG, JPEG or PNG image.')

    mimetype = (file_storage.mimetype or '').lower()
    if mimetype and not mimetype.startswith('image/'):
        raise ValueError('The uploaded file is not an image.')

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_storage.save(filepath)

    if cv2.imread(filepath) is None:
        os.remove(filepath)
        raise ValueError('The file is not a readable image or is corrupted.')

    return filename, filepath


def predict_file(filepath):
    if model is None:
        raise RuntimeError(MODEL_ERROR or 'Model not loaded. Run train_model.py first.')

    batch = preprocess_image(filepath)
    probabilities = np.asarray(model.predict(batch, verbose=0))[0]
    order = np.argsort(probabilities)[::-1]

    top_indices = order[:3]
    top_predictions = [
        {
            'label': CLASS_NAMES[index] if index < len(CLASS_NAMES) else 'Unknown',
            'display_name': display_name(CLASS_NAMES[index]) if index < len(CLASS_NAMES) else 'Unknown',
            'confidence': float(probabilities[index]),
        }
        for index in top_indices
    ]

    best = top_predictions[0]
    return {
        'prediction': best['label'],
        'display_name': best['display_name'],
        'confidence': best['confidence'],
        'low_confidence': best['confidence'] < LOW_CONFIDENCE,
        'top_predictions': top_predictions,
    }


def processed_data_uri(filepath):
    preview = processed_preview(filepath)
    ok, buffer = cv2.imencode('.png', cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    encoded = base64.b64encode(buffer.tobytes()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def build_response(filename, filepath):
    prediction = predict_file(filepath)
    return {
        'filename': filename,
        'image_url': f'/uploads/{filename}',
        'hash': verify_image_integrity(filepath),
        'processed_image': processed_data_uri(filepath),
        'message': 'Leaf classified successfully.',
        **prediction,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': len(CLASS_NAMES),
    })


@app.route('/api/info')
def api_info():
    metrics = read_metrics()
    return jsonify({
        'model_loaded': model is not None,
        'model_error': MODEL_ERROR if model is None else None,
        'num_classes': len(CLASS_NAMES),
        'max_file_mb': MAX_FILE_SIZE // (1024 * 1024),
        'low_confidence_threshold': LOW_CONFIDENCE,
        'test_accuracy': (metrics or {}).get('test_accuracy'),
        'test_top3_accuracy': (metrics or {}).get('test_top3_accuracy'),
        'test_samples': (metrics or {}).get('test_samples'),
    })


@app.route('/api/classes')
def api_classes():
    return jsonify({
        'classes': [
            {'label': label, 'display_name': display_name(label)}
            for label in CLASS_NAMES
        ]
    })


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/classify', methods=['POST'])
def classify():
    if model is None:
        return jsonify({
            'error': 'The model is not loaded. Run "python train_model.py" first.',
            'details': MODEL_ERROR,
        }), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
        filename, filepath = decode_upload(file)
        return jsonify(build_response(filename, filepath))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'Classification failed: {exc}'}), 500


@app.route('/command', methods=['POST'])
def command_classify():
    """Run a command such as ``CLASSIFY 123.jpg`` against an uploaded file."""
    if model is None:
        return jsonify({
            'error': 'The model is not loaded. Run "python train_model.py" first.',
            'details': MODEL_ERROR,
        }), 503

    data = request.get_json(silent=True) or {}
    command_text = str(data.get('command', '')).strip()
    if not command_text:
        return jsonify({'error': 'No command provided.'}), 400

    valid, result = parse_command(command_text, upload_dir=app.config['UPLOAD_FOLDER'])
    if not valid:
        return jsonify({'error': result}), 400

    try:
        return jsonify(build_response(result['filename'], result['filepath']))
    except Exception as exc:
        return jsonify({'error': f'Classification failed: {exc}'}), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_error):
    return jsonify({'error': 'File is too large. Maximum size is 5 MB.'}), 413


@app.errorhandler(404)
def handle_not_found(_error):
    if request.path.startswith('/api') or request.path in ('/classify', '/command'):
        return jsonify({'error': 'Endpoint not found.'}), 404
    return render_template('index.html'), 404


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
