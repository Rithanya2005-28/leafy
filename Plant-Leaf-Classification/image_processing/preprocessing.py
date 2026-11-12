import cv2
import numpy as np

IMG_SIZE = (224, 224)


def _load_rgb(image_path):
    """Read an image from disk and convert OpenCV's BGR order to RGB."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Could not read image: {image_path}')
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image(image_path):
    """Resize, convert to RGB, denoise with Gaussian blur, and scale to [-1, 1].

    The scaling matches MobileNetV2 preprocessing used during training.
    Returns a NumPy array with a batch dimension (1, 224, 224, 3).
    """
    img = _load_rgb(image_path)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img.astype('float32') / 127.5 - 1.0
    return np.expand_dims(img, axis=0)


def processed_preview(image_path):
    """Return the processed image as RGB uint8 for display in the UI."""
    img = _load_rgb(image_path)
    return cv2.GaussianBlur(img, (5, 5), 0)
