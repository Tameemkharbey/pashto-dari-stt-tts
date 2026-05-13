'use strict';
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const required = ['MODAL_API_KEY', 'PASHTO_STT_URL', 'PASHTO_TTS_URL', 'DARI_STT_URL', 'DARI_TTS_URL'];
for (const key of required) {
  if (!process.env[key]) throw new Error(`${key} is not set. Copy .env.example → .env and fill it in.`);
}

module.exports = {
  apiKey: process.env.MODAL_API_KEY,
  port: parseInt(process.env.PORT || '3000', 10),
  sessionSecret: process.env.SESSION_SECRET || 'dev-change-in-production',

  endpoints: {
    pashto: {
      stt: process.env.PASHTO_STT_URL,
      tts: process.env.PASHTO_TTS_URL,
    },
    dari: {
      stt: process.env.DARI_STT_URL,
      tts: process.env.DARI_TTS_URL,
    },
  },

  ttsDefaults: {
    pashto: { noise_scale: 0.4,   noise_scale_w: 0.8, length_scale: 1.0 },
    dari:   { noise_scale: 0.667, noise_scale_w: 0.8, length_scale: 1.0 },
  },
};
