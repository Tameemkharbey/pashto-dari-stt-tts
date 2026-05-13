'use strict';
const express  = require('express');
const multer   = require('multer');
const { v4: uuidv4 } = require('uuid');
const rateLimiter = require('./rateLimiter');
const modal    = require('./modalClient');
const { convertToWav16k } = require('./audioUtils');
const logger   = require('./logger');

const router = express.Router();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter(_req, file, cb) {
    if (/audio\//i.test(file.mimetype)) return cb(null, true);
    cb(new Error('Only audio files are accepted.'));
  },
});

function userId(req) {
  if (!req.session.userId) req.session.userId = uuidv4();
  return req.session.userId;
}

function ip(req) {
  return req.ip || '0.0.0.0';
}

// ── Health ────────────────────────────────────────────────────────────────────
router.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'pashto-dari-voice' });
});

// ── Rate-limit stats (for debugging) ─────────────────────────────────────────
router.get('/limits', (req, res) => {
  res.json(rateLimiter.stats(userId(req), ip(req)));
});

// ── STT ───────────────────────────────────────────────────────────────────────
router.post('/stt', upload.single('audio'), async (req, res) => {
  const uid   = userId(req);
  const start = Date.now();
  const lang  = req.body.language;

  if (!['pashto', 'dari'].includes(lang))
    return res.status(422).json({ error: 'language must be "pashto" or "dari".' });
  if (!req.file)
    return res.status(422).json({ error: 'No audio file provided.' });

  const check = rateLimiter.check(uid, ip(req));
  if (!check.ok)
    return res.status(check.status).json({ error: check.message });

  rateLimiter.consume(uid, ip(req));
  modal.warmupTts(lang); // parallel warm-up hides next TTS cold start

  try {
    const wav    = await convertToWav16k(req.file.buffer);
    const result = await modal.transcribe(lang, wav, parseInt(req.body.beam_size) || 5);
    logger.info({ event: 'stt', userId: uid, language: lang, latency_ms: Date.now() - start, duration_sec: result.duration_sec });
    res.json(result);
  } catch (err) {
    logger.error({ event: 'stt_error', userId: uid, language: lang, status: err.status, msg: err.message });
    res.status(err.status || 500).json({ error: err.message || 'STT failed.' });
  } finally {
    rateLimiter.release(uid);
  }
});

// ── TTS ───────────────────────────────────────────────────────────────────────
router.post('/tts', express.json(), async (req, res) => {
  const uid   = userId(req);
  const start = Date.now();
  const { language, text, noise_scale, noise_scale_w, length_scale } = req.body;

  if (!['pashto', 'dari'].includes(language))
    return res.status(422).json({ error: 'language must be "pashto" or "dari".' });
  if (!text || !text.trim())
    return res.status(422).json({ error: 'text is required.' });
  if (text.length > 500)
    return res.status(422).json({ error: 'text must be ≤ 500 characters.' });

  const check = rateLimiter.check(uid, ip(req));
  if (!check.ok)
    return res.status(check.status).json({ error: check.message });

  rateLimiter.consume(uid, ip(req));

  try {
    const overrides = {};
    if (noise_scale   !== undefined) overrides.noise_scale   = noise_scale;
    if (noise_scale_w !== undefined) overrides.noise_scale_w = noise_scale_w;
    if (length_scale  !== undefined) overrides.length_scale  = length_scale;

    const wav = await modal.synthesize(language, text.trim(), overrides);
    logger.info({ event: 'tts', userId: uid, language, chars: text.length, latency_ms: Date.now() - start });

    res.set({
      'Content-Type': 'audio/wav',
      'Content-Length': wav.length,
      'Content-Disposition': `attachment; filename="${language}_tts.wav"`,
    });
    res.send(wav);
  } catch (err) {
    logger.error({ event: 'tts_error', userId: uid, language, status: err.status, msg: err.message });
    res.status(err.status || 500).json({ error: err.message || 'TTS failed.' });
  } finally {
    rateLimiter.release(uid);
  }
});

module.exports = router;
