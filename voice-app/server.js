'use strict';
const express = require('express');
const session = require('express-session');
const path    = require('path');
const config  = require('./src/config');
const routes  = require('./src/routes');
const logger  = require('./src/logger');

const app = express();

app.set('trust proxy', 1);

app.use(session({
  secret: config.sessionSecret,
  resave: false,
  saveUninitialized: true,
  cookie: { secure: false, httpOnly: true, maxAge: 24 * 60 * 60 * 1000 },
}));

app.use(express.static(path.join(__dirname, 'public')));
app.use('/api', routes);

// 413 from multer
app.use((err, _req, res, _next) => {
  if (err.code === 'LIMIT_FILE_SIZE')
    return res.status(413).json({ error: 'Audio file exceeds 5 MB limit.' });
  res.status(500).json({ error: err.message });
});

app.listen(config.port, () => {
  logger.info({ msg: `Pashto & Dari Voice running at http://localhost:${config.port}` });
});
