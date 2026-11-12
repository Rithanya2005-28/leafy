"""Unit tests for the non-model modules.

Run with:
    C:\\Users\\<you>\\leaf-venv\\Scripts\\python.exe -m unittest -v
"""

import os
import tempfile
import unittest

import numpy as np

from compiler.command_parser import parse_command, tokenize
from image_processing.preprocessing import preprocess_image, processed_preview
from security.image_security import verify_image_integrity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CommandParserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.upload_dir = self.tmp.name
        self.valid_file = os.path.join(self.upload_dir, 'leaf.jpg')
        with open(self.valid_file, 'wb') as handle:
            handle.write(b'not-a-real-image-but-the-file-exists')

    def tearDown(self):
        self.tmp.cleanup()

    def test_tokenize(self):
        tokens = tokenize('CLASSIFY leaf.jpg')
        self.assertEqual(tokens[0], ('COMMAND', 'CLASSIFY'))
        self.assertEqual(tokens[1], ('FILENAME', 'leaf.jpg'))
        self.assertEqual(tokens[2][0], 'EOF')

    def test_valid_command(self):
        ok, result = parse_command('CLASSIFY leaf.jpg', self.upload_dir)
        self.assertTrue(ok, result)
        self.assertEqual(result['filename'], 'leaf.jpg')

    def test_quoted_filename(self):
        spaced = os.path.join(self.upload_dir, 'my leaf.png')
        with open(spaced, 'wb') as handle:
            handle.write(b'x')
        ok, result = parse_command('CLASSIFY "my leaf.png"', self.upload_dir)
        self.assertTrue(ok, result)

    def test_empty_command(self):
        ok, message = parse_command('', self.upload_dir)
        self.assertFalse(ok)
        self.assertIn('empty command', message)

    def test_wrong_keyword(self):
        ok, message = parse_command('SCAN leaf.jpg', self.upload_dir)
        self.assertFalse(ok)
        self.assertIn('expected', message.lower())

    def test_missing_filename(self):
        ok, message = parse_command('CLASSIFY', self.upload_dir)
        self.assertFalse(ok)
        self.assertIn('missing filename', message.lower())

    def test_extra_token(self):
        ok, message = parse_command('CLASSIFY leaf.jpg now', self.upload_dir)
        self.assertFalse(ok)
        self.assertIn('unexpected token', message.lower())

    def test_path_traversal_rejected(self):
        ok, message = parse_command('CLASSIFY ../secret.jpg', self.upload_dir)
        self.assertFalse(ok)
        self.assertIn('path', message.lower())

    def test_bad_extension(self):
        ok, message = parse_command('CLASSIFY notes.txt', self.upload_dir)
        self.assertFalse(ok)
        self.assertIn('unsupported file type', message.lower())

    def test_missing_file(self):
        ok, message = parse_command('CLASSIFY absent.jpg', self.upload_dir)
        self.assertFalse(ok)
        self.assertIn('not found', message.lower())

    def test_unterminated_quote(self):
        ok, message = parse_command('CLASSIFY "leaf.jpg', self.upload_dir)
        self.assertFalse(ok)
        self.assertIn('unterminated', message.lower())


class SecurityTests(unittest.TestCase):
    def test_hash_roundtrip(self):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b'leaf-bytes')
            path = handle.name
        try:
            digest = verify_image_integrity(path)
            self.assertEqual(len(digest), 64)
            valid, actual = verify_image_integrity(path, digest)
            self.assertTrue(valid)
            self.assertEqual(actual, digest)
            valid, _ = verify_image_integrity(path, '0' * 64)
            self.assertFalse(valid)
        finally:
            os.remove(path)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            verify_image_integrity('definitely-missing.jpg')


class PreprocessingTests(unittest.TestCase):
    def test_grayscale_image_is_converted_to_rgb_batch(self):
        import cv2

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as handle:
            gray = np.full((120, 80), 128, dtype=np.uint8)
            cv2.imwrite(handle.name, gray)
            path = handle.name

        try:
            batch = preprocess_image(path)
            self.assertEqual(batch.shape, (1, 224, 224, 3))
            self.assertEqual(batch.dtype, np.float32)
            self.assertGreaterEqual(float(batch.min()), -1.0)
            self.assertLessEqual(float(batch.max()), 1.0)

            preview = processed_preview(path)
            self.assertEqual(preview.shape, (224, 224, 3))
            self.assertEqual(preview.dtype, np.uint8)
        finally:
            os.remove(path)

    def test_unreadable_image(self):
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as handle:
            handle.write(b'not an image')
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                preprocess_image(path)
        finally:
            os.remove(path)


class ProjectStructureTests(unittest.TestCase):
    def test_required_paths_exist(self):
        required = [
            'app.py',
            'train_model.py',
            'prepare_dataset.py',
            'requirements.txt',
            'compiler/command_parser.py',
            'security/image_security.py',
            'image_processing/preprocessing.py',
            'templates/index.html',
            'static/css/style.css',
            'static/js/script.js',
        ]
        for relative in required:
            with self.subTest(path=relative):
                self.assertTrue(
                    os.path.isfile(os.path.join(BASE_DIR, relative)),
                    f'Missing {relative}',
                )

    def test_class_names_match_split_folders(self):
        import json

        class_names_path = os.path.join(BASE_DIR, 'model', 'class_names.json')
        train_dir = os.path.join(BASE_DIR, 'dataset_split', 'train')
        if not os.path.isfile(class_names_path) or not os.path.isdir(train_dir):
            self.skipTest('Run prepare_dataset.py first')
        with open(class_names_path, encoding='utf-8') as handle:
            names = json.load(handle)
        folders = sorted(
            entry for entry in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, entry))
        )
        self.assertEqual(names, folders)


if __name__ == '__main__':
    unittest.main(verbosity=2)
