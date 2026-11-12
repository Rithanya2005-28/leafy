(() => {
    'use strict';

    const MAX_BYTES = 5 * 1024 * 1024;
    const ALLOWED_TYPES = ['image/jpeg', 'image/png'];
    const ALLOWED_EXT = /\.(jpe?g|png)$/i;

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const previewWrap = document.getElementById('preview-wrap');
    const preview = document.getElementById('preview');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const uploadError = document.getElementById('upload-error');
    const classifyBtn = document.getElementById('classify-btn');
    const resetBtn = document.getElementById('reset-btn');
    const resultEmpty = document.getElementById('result-empty');
    const result = document.getElementById('result');
    const modelAlert = document.getElementById('model-alert');
    const commandForm = document.getElementById('command-form');
    const commandInput = document.getElementById('command-input');
    const commandOutput = document.getElementById('command-output');

    let selectedFile = null;
    let previewUrl = null;

    const escapeHtml = (value) => String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');

    const formatBytes = (bytes) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    };

    const percent = (value) => `${(Number(value) * 100).toFixed(1)}%`;

    function showError(message) {
        uploadError.textContent = message;
        uploadError.hidden = false;
    }

    function clearError() {
        uploadError.textContent = '';
        uploadError.hidden = true;
    }

    function setBusy(busy) {
        classifyBtn.classList.toggle('loading', busy);
        classifyBtn.disabled = busy || !selectedFile;
        resetBtn.disabled = busy || !selectedFile;
        classifyBtn.setAttribute('aria-busy', String(busy));
        classifyBtn.querySelector('.btn-label').textContent = busy
            ? 'Classifying'
            : 'Classify leaf';
    }

    function clearResult() {
        result.hidden = true;
        resultEmpty.hidden = false;
        document.getElementById('prediction-name').textContent = '—';
        document.getElementById('confidence-value').textContent = '0%';
        document.getElementById('confidence-fill').style.width = '0%';
        document.getElementById('confidence-ring').style.setProperty('--pct', '0');
        document.getElementById('top-predictions').innerHTML = '';
        document.getElementById('low-confidence').hidden = true;
        document.getElementById('processed-image').hidden = true;
        document.getElementById('original-image').hidden = true;
    }

    function validateFile(file) {
        if (!file) return 'Please choose an image file.';
        const typeOk = ALLOWED_TYPES.includes(file.type);
        const nameOk = ALLOWED_EXT.test(file.name);
        if (!typeOk && !nameOk) {
            return 'Invalid file type. Please upload a JPG, JPEG or PNG image.';
        }
        if (file.size > MAX_BYTES) {
            return `File is too large (${formatBytes(file.size)}). Maximum size is 5 MB.`;
        }
        if (file.size === 0) {
            return 'The selected file is empty.';
        }
        return null;
    }

    function selectFile(file) {
        const error = validateFile(file);
        if (error) {
            showError(error);
            return;
        }
        clearError();
        selectedFile = file;

        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = URL.createObjectURL(file);
        preview.src = previewUrl;
        previewWrap.hidden = false;
        fileName.textContent = file.name;
        fileSize.textContent = formatBytes(file.size);

        classifyBtn.disabled = false;
        resetBtn.disabled = false;
        commandOutput.hidden = true;
        clearResult();
    }

    function resetAll() {
        selectedFile = null;
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }
        fileInput.value = '';
        preview.removeAttribute('src');
        previewWrap.hidden = true;
        fileName.textContent = '';
        fileSize.textContent = '';
        classifyBtn.disabled = true;
        resetBtn.disabled = true;
        clearError();
        clearResult();
        setBusy(false);
    }

    /* ---------------- File selection ---------------- */
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) selectFile(fileInput.files[0]);
    });

    ['dragenter', 'dragover'].forEach((name) => {
        dropzone.addEventListener(name, (event) => {
            event.preventDefault();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach((name) => {
        dropzone.addEventListener(name, (event) => {
            event.preventDefault();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (event) => {
        const file = event.dataTransfer?.files?.[0];
        if (file) selectFile(file);
    });

    window.addEventListener('paste', (event) => {
        const item = Array.from(event.clipboardData?.items || [])
            .find((entry) => entry.type.startsWith('image/'));
        if (item) {
            event.preventDefault();
            selectFile(item.getAsFile());
        }
    });

    resetBtn.addEventListener('click', resetAll);

    /* ---------------- Classification ---------------- */
    function renderResult(data) {
        resultEmpty.hidden = true;
        result.hidden = false;

        document.getElementById('prediction-name').textContent = data.display_name || 'Unknown';
        document.getElementById('prediction-note').textContent =
            `Best match out of ${data.top_predictions?.length ? 'the top 3' : 'all'} predictions`;
        document.getElementById('status-pill').textContent = 'Processed · Integrity verified';

        const confidence = Number(data.confidence) || 0;
        document.getElementById('confidence-value').textContent = percent(confidence);
        document.getElementById('confidence-fill').style.width = percent(confidence);
        document.getElementById('confidence-ring').style.setProperty('--pct', (confidence * 100).toFixed(1));
        document.getElementById('low-confidence').hidden = !data.low_confidence;

        const list = document.getElementById('top-predictions');
        list.innerHTML = (data.top_predictions || []).map((item) => {
            const pct = percent(item.confidence);
            return `
                <li>
                    <span>${escapeHtml(item.display_name)}</span>
                    <span class="pct">${pct}</span>
                    <span class="bar"><span style="width: ${pct}"></span></span>
                </li>`;
        }).join('');

        const original = document.getElementById('original-image');
        original.src = `${data.image_url}?v=${Date.now()}`;
        original.hidden = false;

        const processed = document.getElementById('processed-image');
        if (data.processed_image) {
            processed.src = data.processed_image;
            processed.hidden = false;
        } else {
            processed.hidden = true;
        }

        document.getElementById('meta-file').textContent = data.filename || '—';
        document.getElementById('meta-hash').textContent = data.hash || '—';

        commandInput.placeholder = `CLASSIFY ${data.filename}`;
    }

    async function readJson(response) {
        try {
            return await response.json();
        } catch {
            return { error: `Server returned HTTP ${response.status}` };
        }
    }

    classifyBtn.addEventListener('click', async () => {
        if (!selectedFile) {
            showError('Please choose an image first.');
            return;
        }

        clearError();
        setBusy(true);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/classify', { method: 'POST', body: formData });
            const data = await readJson(response);
            if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
            renderResult(data);
        } catch (error) {
            showError(error.message || 'Classification failed. Please try again.');
            clearResult();
        } finally {
            setBusy(false);
        }
    });

    /* ---------------- Hash copy ---------------- */
    document.getElementById('copy-hash').addEventListener('click', async () => {
        const hash = document.getElementById('meta-hash').textContent;
        if (!hash || hash === '—') return;
        try {
            await navigator.clipboard.writeText(hash);
            const button = document.getElementById('copy-hash');
            button.textContent = 'Copied';
            setTimeout(() => { button.textContent = 'Copy'; }, 1500);
        } catch {
            showError('Could not copy the hash. Select it manually.');
        }
    });

    /* ---------------- Command mode ---------------- */
    commandForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const command = commandInput.value.trim();
        if (!command) {
            commandOutput.hidden = false;
            commandOutput.textContent = 'Error: enter a command, e.g. CLASSIFY 123.jpg';
            return;
        }

        commandOutput.hidden = false;
        commandOutput.textContent = 'Running…';

        try {
            const response = await fetch('/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command })
            });
            const data = await readJson(response);

            if (!response.ok) {
                commandOutput.textContent = `Error: ${data.error || `HTTP ${response.status}`}`;
                return;
            }

            const top = (data.top_predictions || [])
                .map((item) => `  ${item.display_name}: ${percent(item.confidence)}`)
                .join('\n');

            commandOutput.textContent = [
                'Status   : success',
                `File     : ${data.filename}`,
                `Species  : ${data.display_name}`,
                `Confidence: ${percent(data.confidence)}`,
                'Top 3    :',
                top,
                `SHA-256  : ${data.hash}`
            ].join('\n');

            renderResult(data);
        } catch (error) {
            commandOutput.textContent = `Error: ${error.message}`;
        }
    });

    /* ---------------- Model info ---------------- */
    async function loadInfo() {
        try {
            const response = await fetch('/api/info');
            const info = await readJson(response);

            document.getElementById('stat-classes').textContent = info.num_classes || '—';
            document.getElementById('stat-accuracy').textContent = info.test_accuracy != null
                ? percent(info.test_accuracy)
                : '—';
            document.getElementById('stat-model').textContent = info.model_loaded ? 'Online' : 'Offline';
            document.getElementById('about-classes').textContent = `${info.num_classes || 'the'} leaf`;

            if (!info.model_loaded) {
                modelAlert.hidden = false;
                modelAlert.textContent =
                    'The model is not loaded. Train it with: python train_model.py'
                    + (info.model_error ? ` (${info.model_error})` : '');
            }
        } catch {
            document.getElementById('stat-model').textContent = 'Offline';
        }
    }

    loadInfo();
    clearResult();
})();
