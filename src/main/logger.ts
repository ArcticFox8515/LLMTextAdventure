import { createLogger, format, transports } from 'winston';
import fs from 'fs';
import path from 'path';

const logDir = 'logs';

// Ensure the logs directory exists
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir);
}

// Generate a unique log filename with the format server-YYYY-MM-DD-Number.log
const currentDate = new Date();
const dateString = currentDate.toISOString().split('T')[0]; // YYYY-MM-DD format

// Get existing log files for today and find the next number
const todaysLogs = fs.readdirSync(logDir)
  .filter(file => file.startsWith(`server-${dateString}`) && file.endsWith('.log'));
  
const nextNumber = todaysLogs.length > 0 
  ? Math.max(...todaysLogs.map(file => {
      const match = file.match(/server-\d{4}-\d{2}-\d{2}-(\d+)\.log/);
      return match ? parseInt(match[1]) : 0;
    })) + 1
  : 1;

const logFileName = `server-${dateString}-${nextNumber}.log`;
const logFilePath = path.join(logDir, logFileName);

// Clean up old log files, keeping only the latest 5
const allLogFiles = fs.readdirSync(logDir)
  .filter(file => file.match(/server-\d{4}-\d{2}-\d{2}-\d+\.log/))
  .map(file => ({
    name: file,
    path: path.join(logDir, file),
    time: fs.statSync(path.join(logDir, file)).mtime.getTime()
  }))
  .sort((a, b) => b.time - a.time); // Sort by time descending (newest first)

// Remove all but the 5 newest logs
if (allLogFiles.length > 5) {
  allLogFiles.slice(5).forEach(file => fs.unlinkSync(file.path));
}

// Create the Winston logger with file and console transport
const baseLogger = createLogger({
  level: 'info',
  format: format.combine(
    format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
    format.printf(({ timestamp, level, message, ...meta }) => {
      const splatSymbol = Symbol.for('splat');
      const metaString = splatSymbol in meta && Array.isArray(meta[splatSymbol])
        ? meta[splatSymbol].join(' ') 
        : '';
      return `${timestamp} [${level.toUpperCase()}]: ${message} ${metaString}`;
    })
  ),
  transports: [
    new transports.Console(),
    new transports.File({
      filename: logFilePath,
      maxsize: 20 * 1024 * 1024 // 20MB max size
    })
  ],
});

// Create a wrapper for the logger
export const logger = {
  info: baseLogger.info.bind(baseLogger),
  warn: baseLogger.warn.bind(baseLogger),
  error: baseLogger.error.bind(baseLogger),
  debug: baseLogger.debug.bind(baseLogger),

  // Custom method for incremental logging
  incremental: (text: string) => {
    process.stdout.write(text);
    fs.appendFileSync(logFilePath, text);
  },
};