'use strict';
const { spawn } = require('child_process');

/**
 * Convert any audio buffer to WAV 16 kHz mono using ffmpeg.
 * Required before sending browser/WhatsApp audio to the STT endpoint.
 */
function convertToWav16k(inputBuffer) {
  return new Promise((resolve, reject) => {
    const proc = spawn('ffmpeg', [
      '-i',  'pipe:0',
      '-ar', '16000',
      '-ac', '1',
      '-f',  'wav',
      'pipe:1',
    ], { stdio: ['pipe', 'pipe', 'pipe'] });

    const chunks = [];
    proc.stdout.on('data', chunk => chunks.push(chunk));
    proc.stderr.on('data', () => {});

    proc.on('error', err => {
      reject(new Error(
        `ffmpeg not found. Install it from https://ffmpeg.org and ensure it is in your PATH. (${err.message})`
      ));
    });

    proc.on('close', code => {
      if (code === 0) resolve(Buffer.concat(chunks));
      else reject(new Error(`Audio conversion failed (ffmpeg exited ${code})`));
    });

    proc.stdin.write(inputBuffer);
    proc.stdin.end();
  });
}

module.exports = { convertToWav16k };
