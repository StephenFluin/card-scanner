import { Service, signal } from '@angular/core';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export type LogCategory =
  | 'app'
  | 'camera'
  | 'worker'
  | 'detect'
  | 'read'
  | 'name'
  | 'fusion'
  | 'capture'
  | 'catalog'
  | 'auth'
  | 'db';

export interface LogEntry {
  /** ms since page load */
  t: number;
  /** wall clock ISO time */
  time: string;
  level: LogLevel;
  category: LogCategory;
  message: string;
  data?: unknown;
}

const MAX_ENTRIES = 1500;
const LEVELS: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };

/**
 * In-memory ring buffer of structured log entries, mirrored to the console.
 * The debug panel renders it, and it is included in downloaded debug bundles.
 */
@Service()
export class Logger {
  private readonly buffer: LogEntry[] = [];
  private readonly _entries = signal<readonly LogEntry[]>([]);
  readonly entries = this._entries.asReadonly();
  /** Entries below this level are kept in the buffer but not echoed to the console. */
  readonly consoleLevel = signal<LogLevel>('info');
  private flushScheduled = false;

  log(level: LogLevel, category: LogCategory, message: string, data?: unknown): void {
    const entry: LogEntry = {
      t: Math.round(performance.now()),
      time: new Date().toISOString(),
      level,
      category,
      message,
      data,
    };
    this.buffer.push(entry);
    if (this.buffer.length > MAX_ENTRIES) this.buffer.splice(0, this.buffer.length - MAX_ENTRIES);
    if (LEVELS[level] >= LEVELS[this.consoleLevel()]) {
      const fn = level === 'error' ? console.error : level === 'warn' ? console.warn : console.log;
      if (data === undefined) fn(`[${category}] ${message}`);
      else fn(`[${category}] ${message}`, data);
    }
    // Batch signal updates: per-frame logging must not trigger a render per entry.
    if (!this.flushScheduled) {
      this.flushScheduled = true;
      queueMicrotask(() => {
        this.flushScheduled = false;
        this._entries.set([...this.buffer]);
      });
    }
  }

  debug(category: LogCategory, message: string, data?: unknown): void {
    this.log('debug', category, message, data);
  }
  info(category: LogCategory, message: string, data?: unknown): void {
    this.log('info', category, message, data);
  }
  warn(category: LogCategory, message: string, data?: unknown): void {
    this.log('warn', category, message, data);
  }
  error(category: LogCategory, message: string, data?: unknown): void {
    this.log('error', category, message, data);
  }

  clear(): void {
    this.buffer.length = 0;
    this._entries.set([]);
  }

  snapshot(): LogEntry[] {
    return [...this.buffer];
  }
}
