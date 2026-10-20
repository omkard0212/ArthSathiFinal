/**
 * ArthSathi — Express Application
 */

require('dotenv').config();

const express   = require('express');
const cors      = require('cors');
const helmet    = require('helmet');
const morgan    = require('morgan');
const rateLimit = require('express-rate-limit');
const path      = require('path');

const healthRoutes    = require('./routes/health');
const usersRoutes     = require('./routes/users');
const { schemesRouter, insuranceRouter } = require('./routes/schemes');
const documentsRoutes = require('./routes/documents');
const voiceRoutes     = require('./routes/voice');
const i18nRoutes      = require('./routes/i18n');
const errorHandler    = require('./middleware/errorHandler');

const app = express();

// Trust proxy — required behind ngrok/reverse proxy
app.set('trust proxy', 1);

app.use(helmet());
app.use(cors({
  origin: [
    process.env.FRONTEND_URL || 'http://localhost:3000',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
  ],
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
}));

if (process.env.NODE_ENV !== 'test') app.use(morgan('dev'));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '900000', 10),
  max:      parseInt(process.env.RATE_LIMIT_MAX        || '200',    10),
  standardHeaders: true,
  legacyHeaders:   false,
});
app.use('/api/', limiter);

app.use('/uploads', express.static(path.join(__dirname, '..', 'uploads')));

// Routes
app.use('/api/health',    healthRoutes);
app.use('/api/users',     usersRoutes);
app.use('/api/schemes',   schemesRouter);
app.use('/api/insurance', insuranceRouter);
app.use('/api/documents', documentsRoutes);
app.use('/api/voice',     voiceRoutes);
app.use('/api/i18n',      i18nRoutes);

app.get('/', (req, res) => res.json({ name: 'ArthSathi API', version: '2.0.0', status: 'running' }));

app.use((req, res) => res.status(404).json({ success: false, error: { code: 'NOT_FOUND', message: `Route ${req.method} ${req.path} not found` } }));
app.use(errorHandler);

module.exports = app;
