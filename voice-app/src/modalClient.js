'use strict';
const config = require('./config');
const logger = require('./logger');

const HEADERS = {
  'X-API-Key': config.apiKey,
  'Content-Type': 'application/json',
};

async function _post(url, body, timeoutMs) {
  return fetch(url, {
    method: 'POST',
    headers: HEADERS,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

async function _postWithRetry(url, body, timeoutMs) {
  let res = await _post(url, body, timeoutMs);
  if (res.status === 503) {
    logger.warn({ event: 'cold_start', url, action: 'waiting_15s_then_retry' });
    await new Promise(r => setTimeout(r, 15_000));
    res = await _post(url, body, timeoutMs);
  }
  return res;
}

async function transcribe(language, audioBuffer, beamSize = 5) {
  const url = config.endpoints[language].stt;
  const res = await _postWithRetry(
    url,
    { audio_base64: audioBuffer.toString('base64'), beam_size: beamSize },
    60_000,
  );
  if (!res.ok) {
    const text = await res.text();
    throw Object.assign(new Error(text), { status: res.status });
  }
  return res.json();
}

async function synthesize(language, text, overrides = {}) {
  const url = config.endpoints[language].tts;
  const body = { text, ...config.ttsDefaults[language], ...overrides };
  const res = await _postWithRetry(url, body, 120_000);
  if (!res.ok) {
    const text = await res.text();
    throw Object.assign(new Error(text), { status: res.status });
  }
  return Buffer.from(await res.arrayBuffer());
}

// Fire-and-forget warmup — hides the 41-sec cold start on the next TTS call.
// Called automatically after each STT request (same language).
function warmupTts(language) {
  const url = config.endpoints[language].tts;
  const body = { text: 'سلام', ...config.ttsDefaults[language] };
  fetch(url, {
    method: 'POST',
    headers: HEADERS,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(120_000),
  }).catch(() => {});
}

module.exports = { transcribe, synthesize, warmupTts };
