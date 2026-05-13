'use strict';

function log(level, data) {
  // Never include API key in logs — filter it defensively
  const safe = { ...data };
  delete safe.apiKey;
  delete safe.api_key;
  delete safe['x-api-key'];
  console.log(JSON.stringify({ ts: new Date().toISOString(), level, ...safe }));
}

module.exports = {
  info:  (data) => log('INFO',  data),
  warn:  (data) => log('WARN',  data),
  error: (data) => log('ERROR', data),
};
