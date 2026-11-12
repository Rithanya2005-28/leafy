import hashlib
import os

def verify_image_integrity(file_path, expected_hash=None):
    """Compute SHA-256 hash of the image file.
    If ``expected_hash`` is provided, returns a tuple ``(is_valid, actual_hash)`` where ``is_valid`` is True when the hashes match.
    If ``expected_hash`` is None, simply returns the computed hash.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
        actual_hash = hashlib.sha256(file_bytes).hexdigest()
    if expected_hash is None:
        return actual_hash
    return (actual_hash == expected_hash, actual_hash)

