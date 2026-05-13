'use strict';

// In-memory rate limiter — replace buckets with Redis in production.
class RateLimiter {
  constructor() {
    this.userMinute = new Map();
    this.userHour   = new Map();
    this.userDay    = new Map();
    this.ipHour     = new Map();
    this.concurrent = new Set();
    this.global     = { count: 0, resetAt: this._nextMidnightUTC() };
  }

  _nextMidnightUTC() {
    const d = new Date();
    return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() + 1);
  }

  _bucket(map, key, windowMs) {
    const now = Date.now();
    let b = map.get(key);
    if (!b || now >= b.resetAt) {
      b = { count: 0, resetAt: now + windowMs };
      map.set(key, b);
    }
    return b;
  }

  check(userId, ip) {
    const now = Date.now();

    if (now >= this.global.resetAt) {
      this.global = { count: 0, resetAt: this._nextMidnightUTC() };
    }
    if (this.global.count >= 400)
      return { ok: false, status: 503, message: 'Service busy. Try again tomorrow.' };

    if (this.concurrent.has(userId))
      return { ok: false, status: 409, message: 'Finish your previous message first.' };

    if (this._bucket(this.userMinute, userId, 60_000).count >= 3)
      return { ok: false, status: 429, message: 'Too many requests — please wait a minute.' };

    if (this._bucket(this.userHour, userId, 3_600_000).count >= 10)
      return { ok: false, status: 429, message: 'Hourly limit reached. Try again in an hour.' };

    if (this._bucket(this.userDay, userId, 86_400_000).count >= 25)
      return { ok: false, status: 429, message: 'Daily limit reached. Try again tomorrow.' };

    if (this._bucket(this.ipHour, ip, 3_600_000).count >= 40)
      return { ok: false, status: 429, message: 'Too many requests from your network.' };

    return { ok: true };
  }

  consume(userId, ip) {
    this.global.count++;
    this._bucket(this.userMinute, userId, 60_000).count++;
    this._bucket(this.userHour,   userId, 3_600_000).count++;
    this._bucket(this.userDay,    userId, 86_400_000).count++;
    this._bucket(this.ipHour,     ip,     3_600_000).count++;
    this.concurrent.add(userId);
  }

  release(userId) {
    this.concurrent.delete(userId);
  }

  stats(userId, ip) {
    return {
      userMinute: this._bucket(this.userMinute, userId, 60_000).count,
      userHour:   this._bucket(this.userHour,   userId, 3_600_000).count,
      userDay:    this._bucket(this.userDay,     userId, 86_400_000).count,
      ipHour:     this._bucket(this.ipHour,      ip,    3_600_000).count,
      globalDay:  this.global.count,
    };
  }
}

module.exports = new RateLimiter();
