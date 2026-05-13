'use strict';

// ── State ────────────────────────────────────────────────────────────────────
let currentLang   = 'pashto';
let mediaRecorder = null;
let recordedChunks = [];
let timerInterval  = null;
let recordSeconds  = 0;
let audioBlob      = null;  // current audio for STT (from recorder or file upload)
let ttsObjectUrl   = null;  // current TTS audio blob URL

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const recordBtn     = $('record-btn');
const recordLabel   = $('record-label');
const recordTimer   = $('record-timer');
const audioUpload   = $('audio-upload');
const uploadLabel   = $('upload-label');
const uploadZone    = document.querySelector('.upload-zone');
const sttSubmit     = $('stt-submit');
const sttResultWrap = $('stt-result-wrap');
const sttResult     = $('stt-result');
const sttMeta       = $('stt-meta');
const sttCopy       = $('stt-copy');

const ttsText       = $('tts-text');
const charCount     = $('char-count');
const charCountWrap = document.querySelector('.char-count');
const lengthScale   = $('length-scale');
const lengthVal     = $('length-scale-val');
const ttsSubmit     = $('tts-submit');
const ttsAudioWrap  = $('tts-audio-wrap');
const ttsAudio      = $('tts-audio');
const ttsDownload   = $('tts-download');

const loading       = $('loading');
const loadingMsg    = $('loading-msg');
const toast         = $('toast');

// ── Language toggle ───────────────────────────────────────────────────────────
document.querySelectorAll('.lang-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentLang = btn.dataset.lang;
    const placeholder = currentLang === 'pashto'
      ? 'دلته د پښتو متن ولیکئ...'
      : 'متن دری را اینجا بنویسید...';
    ttsText.placeholder = placeholder;
  });
});

// ── Microphone recording ──────────────────────────────────────────────────────
recordBtn.addEventListener('click', async () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    stopRecording();
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast('Microphone API unavailable. Use Chrome/Edge on http://localhost:3000 (not a file://).', 'error');
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    startRecording(stream);
  } catch (err) {
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      showToast('Mic blocked — click the lock icon in the address bar and allow microphone, then refresh.', 'error');
    } else if (err.name === 'NotFoundError') {
      showToast('No microphone found. Plug one in and try again.', 'error');
    } else {
      showToast(`Mic error: ${err.name} — ${err.message}`, 'error');
    }
  }
});

function startRecording(stream) {
  recordedChunks = [];
  audioBlob = null;
  sttSubmit.disabled = true;

  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    audioBlob = new Blob(recordedChunks, { type: 'audio/webm' });
    sttSubmit.disabled = false;
    resetRecordUI();
    uploadLabel.textContent = 'Recording ready';
    uploadZone.classList.add('has-file');
  };

  mediaRecorder.start();
  recordBtn.classList.add('recording');
  recordLabel.textContent = 'Stop Recording';
  recordTimer.classList.remove('hidden');
  recordSeconds = 0;
  updateTimer();
  timerInterval = setInterval(updateTimer, 1000);
}

function stopRecording() {
  if (mediaRecorder) mediaRecorder.stop();
  clearInterval(timerInterval);
}

function resetRecordUI() {
  recordBtn.classList.remove('recording');
  recordLabel.textContent = 'Start Recording';
  recordTimer.classList.add('hidden');
}

function updateTimer() {
  recordSeconds++;
  const m = String(Math.floor(recordSeconds / 60)).padStart(1, '0');
  const s = String(recordSeconds % 60).padStart(2, '0');
  recordTimer.textContent = `${m}:${s}`;

  if (recordSeconds >= 30) stopRecording(); // max 30s per API limit
}

// ── File upload ───────────────────────────────────────────────────────────────
audioUpload.addEventListener('change', () => {
  const file = audioUpload.files[0];
  if (!file) return;

  if (file.size > 5 * 1024 * 1024) {
    showToast('File exceeds 5 MB limit.', 'error');
    audioUpload.value = '';
    return;
  }

  audioBlob = file;
  sttSubmit.disabled = false;
  uploadLabel.textContent = file.name;
  uploadZone.classList.add('has-file');

  // Cancel any active recording
  if (mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
});

// Drag & drop on upload zone
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('has-file'); });
uploadZone.addEventListener('dragleave', () => { if (!audioBlob) uploadZone.classList.remove('has-file'); });
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('audio/')) {
    audioUpload.files = e.dataTransfer.files;
    audioUpload.dispatchEvent(new Event('change'));
  }
});

// ── STT submit ────────────────────────────────────────────────────────────────
sttSubmit.addEventListener('click', async () => {
  if (!audioBlob) return;

  showLoading('Transcribing audio...');

  const form = new FormData();
  form.append('audio', audioBlob, 'audio.webm');
  form.append('language', currentLang);
  form.append('beam_size', '5');

  try {
    const res = await fetch('/api/stt', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      sttResult.textContent = data.error || `Error ${res.status}`;
      sttMeta.textContent = `Status: ${res.status}`;
      sttResultWrap.classList.remove('hidden');
      showToast(data.error || `Error ${res.status}`, 'error');
      return;
    }

    sttResult.textContent = data.text || '(no speech detected)';
    sttMeta.textContent =
      `Language: ${data.language} · Duration: ${data.duration_sec}s · Inference: ${data.inference_time_sec}s`;
    sttResultWrap.classList.remove('hidden');
    showToast('Transcription complete.', 'success');
  } catch (err) {
    sttResult.textContent = err.message;
    sttMeta.textContent = 'Request failed — check server logs';
    sttResultWrap.classList.remove('hidden');
    showToast(err.message, 'error');
  } finally {
    hideLoading();
  }
});

// Copy transcription
sttCopy.addEventListener('click', async () => {
  const text = sttResult.textContent;
  if (!text) return;
  await navigator.clipboard.writeText(text).catch(() => {});
  showToast('Copied!', 'success');
});

// ── TTS char counter ──────────────────────────────────────────────────────────
ttsText.addEventListener('input', () => {
  const len = ttsText.value.length;
  charCount.textContent = len;
  charCountWrap.classList.toggle('near-limit', len >= 400 && len < 500);
  charCountWrap.classList.toggle('at-limit', len >= 500);
});

// ── Speed slider ──────────────────────────────────────────────────────────────
lengthScale.addEventListener('input', () => {
  lengthVal.textContent = `${parseFloat(lengthScale.value).toFixed(1)}×`;
});

// ── TTS submit ────────────────────────────────────────────────────────────────
ttsSubmit.addEventListener('click', async () => {
  const text = ttsText.value.trim();
  if (!text) { showToast('Please enter some text.', 'error'); return; }
  if (text.length > 500) { showToast('Text must be ≤ 500 characters.', 'error'); return; }

  showLoading('Synthesizing speech...');
  ttsAudioWrap.classList.add('hidden');

  try {
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        language: currentLang,
        text,
        length_scale: parseFloat(lengthScale.value),
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `Error ${res.status}` }));
      throw new Error(err.error);
    }

    const blob = new Blob([await res.arrayBuffer()], { type: 'audio/wav' });
    if (ttsObjectUrl) URL.revokeObjectURL(ttsObjectUrl);
    ttsObjectUrl = URL.createObjectURL(blob);

    ttsAudio.src = ttsObjectUrl;
    ttsDownload.href = ttsObjectUrl;
    ttsDownload.download = `${currentLang}_speech.wav`;
    ttsAudioWrap.classList.remove('hidden');
    ttsAudio.play().catch(() => {});
    showToast('Speech generated.', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    hideLoading();
  }
});

// ── Loading helpers ───────────────────────────────────────────────────────────
function showLoading(msg = 'Processing...') {
  loadingMsg.textContent = msg;
  loading.classList.remove('hidden');
}

function hideLoading() {
  loading.classList.add('hidden');
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer = null;

function showToast(message, type = '') {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.className = `toast${type ? ' ' + type : ''}`;
  toast.classList.remove('hidden');
  toastTimer = setTimeout(() => toast.classList.add('hidden'), 4000);
}
