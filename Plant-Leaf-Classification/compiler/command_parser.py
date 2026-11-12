"""Small rule-based command parser for leaf classification.

Grammar:
    command  -> CLASSIFY filename
    filename -> one token ending in .jpg, .jpeg or .png

Pipeline: lexical analysis -> syntax analysis -> semantic analysis.
"""

import os

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


class CommandError(Exception):
    """Raised for lexical errors."""


def tokenize(command_text):
    """Lexical analysis: split the command into tokens."""
    tokens = []
    i = 0
    n = len(command_text)

    while i < n:
        if command_text[i].isspace():
            i += 1
            continue

        # Quoted filename, e.g. CLASSIFY "mango leaf.jpg"
        if command_text[i] == '"':
            end = command_text.find('"', i + 1)
            if end == -1:
                raise CommandError('Lexical error: unterminated quoted string')
            tokens.append(('FILENAME', command_text[i + 1:end]))
            i = end + 1
            continue

        start = i
        while i < n and not command_text[i].isspace():
            i += 1
        word = command_text[start:i]

        if not tokens:
            if word.upper() == 'CLASSIFY':
                tokens.append(('COMMAND', 'CLASSIFY'))
            else:
                tokens.append(('IDENTIFIER', word))
        else:
            tokens.append(('FILENAME', word))

    tokens.append(('EOF', ''))
    return tokens


def parse_command(command_text, upload_dir='uploads'):
    """Parse a CLASSIFY command.

    Returns ``(True, {'filename': ..., 'filepath': ...})`` when valid, otherwise
    ``(False, error_message)``.
    """
    # --- Syntax analysis (empty command) ---
    if not command_text or not command_text.strip():
        return False, 'Syntax error: empty command. Usage: CLASSIFY <filename>'

    # --- Lexical analysis ---
    try:
        tokens = tokenize(command_text)
    except CommandError as exc:
        return False, str(exc)

    # --- Syntax analysis ---
    if tokens[0][0] != 'COMMAND':
        return False, (
            f"Syntax error: expected 'CLASSIFY' but found '{tokens[0][1]}'. "
            'Usage: CLASSIFY <filename>'
        )

    if tokens[1][0] != 'FILENAME':
        return False, 'Syntax error: missing filename. Usage: CLASSIFY <filename>'

    if tokens[2][0] != 'EOF':
        return False, f"Syntax error: unexpected token '{tokens[2][1]}' after filename"

    filename = tokens[1][1]

    # --- Semantic analysis ---
    if not filename:
        return False, 'Semantic error: filename cannot be empty'

    if filename != os.path.basename(filename) or '/' in filename or '\\' in filename:
        return False, 'Semantic error: filename must not contain path components'

    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False, (
            f"Semantic error: unsupported file type '{extension}'. "
            'Use .jpg, .jpeg or .png'
        )

    filepath = os.path.join(upload_dir, filename)
    if not os.path.isfile(filepath):
        return False, (
            f"Semantic error: file '{filename}' was not found in uploads. "
            'Upload it first using the web UI or /classify endpoint.'
        )

    return True, {'filename': filename, 'filepath': filepath}
