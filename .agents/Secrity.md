# 🔐 SECURITY.md — Master Security Intelligence File

> This is the single source of truth for all security requirements, threat models,
> controls, and implementation standards. Claude must read and enforce this file
> across every task that touches authentication, data, APIs, infrastructure, or user input.

---

## 📋 Table of Contents

1. [Security Philosophy](#1-security-philosophy)
2. [STRIDE Threat Model](#2-stride-threat-model)
3. [OWASP Top 10 — Full Coverage](#3-owasp-top-10--full-coverage)
4. [Authentication & JWT Hardening](#4-authentication--jwt-hardening)
5. [Session Management](#5-session-management)
6. [Authorization & Access Control](#6-authorization--access-control)
7. [Input Validation & Injection Prevention](#7-input-validation--injection-prevention)
8. [Data Protection & Cryptography](#8-data-protection--cryptography)
9. [API Security](#9-api-security)
10. [DDoS & Rate Limiting](#10-ddos--rate-limiting)
11. [HTTP Security Headers](#11-http-security-headers)
12. [Audit Logging & Monitoring](#12-audit-logging--monitoring)
13. [Secret & Config Management](#13-secret--config-management)
14. [Dependency & Supply Chain Security](#14-dependency--supply-chain-security)
15. [Security Requirements Model](#15-security-requirements-model)
16. [Compliance Mapping](#16-compliance-mapping)
17. [Incident Response Playbook](#17-incident-response-playbook)
18. [Security Checklist — Definition of Done](#18-security-checklist--definition-of-done)

---

## 1. Security Philosophy

```
NEVER trust. ALWAYS verify. ALWAYS encrypt. ALWAYS log. ALWAYS least-privilege.
```

### Core Principles
| Principle | Meaning |
|---|---|
| **Defense in Depth** | Multiple layers — one layer failing must not compromise the system |
| **Least Privilege** | Every component gets the minimum access it needs, nothing more |
| **Fail Secure** | On error, deny access — never grant it |
| **Zero Trust** | No implicit trust based on network location or prior authentication |
| **Security by Design** | Security is architected in, not bolted on |
| **Assume Breach** | Design as if the attacker is already inside |

### Claude Behavior Rules for Security Tasks
- ❌ Never write hardcoded credentials, secrets, or keys in any file
- ❌ Never suggest disabling security controls "for now" or "in dev"
- ❌ Never produce code that suppresses security errors silently
- ✅ Always flag when a proposed design introduces a vulnerability
- ✅ Always prefer well-audited libraries over custom crypto/auth implementations
- ✅ Always write tests for every security control

---

## 2. STRIDE Threat Model

STRIDE is the required framework for analyzing threats before writing security requirements.

```
S — Spoofing       → Who are you really?
T — Tampering      → Did this data change?
R — Repudiation    → Who did this action?
I — Info Disclosure → Who can see this data?
D — Denial of Service → Can the system be overwhelmed?
E — Elevation of Privilege → Can a user act beyond their role?
```

### STRIDE → Control Mapping

| STRIDE Threat | Security Domain | Primary Controls |
|---|---|---|
| **Spoofing** | Authentication | MFA, strong passwords, token verification, anti-phishing |
| **Tampering** | Integrity | Input validation, HMAC, digital signatures, parameterized queries |
| **Repudiation** | Audit Logging | Immutable logs, signed audit trails, non-repudiation tokens |
| **Information Disclosure** | Data Protection | Encryption at rest/transit, access control, data masking |
| **Denial of Service** | Availability | Rate limiting, WAF, CDN, circuit breakers, resource quotas |
| **Elevation of Privilege** | Authorization | RBAC/ABAC, server-side checks, principle of least privilege |

### Risk Scoring Matrix
```
Priority = Impact × Likelihood

CRITICAL  = score ≥ 12  → Must fix before deployment
HIGH      = score 6–11  → Must fix within current sprint
MEDIUM    = score 3–5   → Fix within 2 sprints
LOW       = score 1–2   → Track in backlog
```

---

## 3. OWASP Top 10 — Full Coverage

### A01 — Broken Access Control ⚠️ CRITICAL

**What it is:** Users act outside their intended permissions.

```typescript
// ❌ BAD — client-supplied role trusted
app.get('/admin', (req, res) => {
  if (req.body.role === 'admin') grantAccess(); // NEVER trust client
});

// ✅ GOOD — server-side RBAC
app.get('/admin', authenticate, authorize(['admin']), (req, res) => {
  grantAccess();
});

// ✅ GOOD — IDOR prevention: always scope to authenticated user
app.get('/orders/:id', authenticate, async (req, res) => {
  const order = await db.orders.findFirst({
    where: { id: req.params.id, userId: req.user.id } // scope to owner
  });
  if (!order) return res.status(404).json({ error: 'Not found' });
  res.json(order);
});
```

**Controls:**
- Enforce access control on every server-side route — no exceptions
- Default deny: if no explicit permission, deny
- Log all access control failures with user identity
- Disable directory listing on web servers
- Invalidate JWTs on logout (server-side deny list)

---

### A02 — Cryptographic Failures

**What it is:** Sensitive data exposed due to weak/missing encryption.

```typescript
import bcrypt from 'bcrypt';
import { createCipheriv, createDecipheriv, randomBytes } from 'crypto';

// ✅ Password hashing — bcrypt, never MD5/SHA1/plain
const SALT_ROUNDS = 12;
const hash = await bcrypt.hash(plainPassword, SALT_ROUNDS);
const valid = await bcrypt.compare(plainPassword, hash);

// ✅ AES-256-GCM for field-level encryption (PII, PAN, PHI)
const ALGORITHM = 'aes-256-gcm';
function encryptField(plaintext: string, key: Buffer): string {
  const iv = randomBytes(16);
  const cipher = createCipheriv(ALGORITHM, key, iv);
  const encrypted = Buffer.concat([cipher.update(plaintext, 'utf8'), cipher.final()]);
  const tag = cipher.getAuthTag();
  return Buffer.concat([iv, tag, encrypted]).toString('base64');
}

function decryptField(ciphertext: string, key: Buffer): string {
  const data = Buffer.from(ciphertext, 'base64');
  const iv = data.subarray(0, 16);
  const tag = data.subarray(16, 32);
  const encrypted = data.subarray(32);
  const decipher = createDecipheriv(ALGORITHM, key, iv);
  decipher.setAuthTag(tag);
  return decipher.update(encrypted) + decipher.final('utf8');
}
```

**Controls:**
- TLS 1.2+ required; TLS 1.3 preferred
- No MD5, SHA1, DES, RC4, 3DES — ever
- AES-256-GCM for symmetric encryption
- RSA-2048+ or ECDSA P-256+ for asymmetric
- Rotate encryption keys annually or on compromise
- Never log or expose encrypted fields in plaintext

---

### A03 — Injection

**What it is:** Untrusted data sent to an interpreter as part of a command or query.

```typescript
import Joi from 'joi';
import { PrismaClient } from '@prisma/client';

const db = new PrismaClient();

// ❌ BAD — SQL Injection
// db.$queryRaw(`SELECT * FROM users WHERE email = '${email}'`);

// ✅ GOOD — Parameterized via ORM
const user = await db.user.findFirst({ where: { email } });

// ✅ GOOD — Raw SQL with tagged template (Prisma auto-escapes)
const results = await db.$queryRaw`SELECT * FROM users WHERE email = ${email}`;

// ✅ Input validation schema
const loginSchema = Joi.object({
  email:    Joi.string().email().max(254).required(),
  password: Joi.string().min(8).max(128).required(),
});

// ✅ NoSQL Injection prevention (MongoDB)
// ❌ BAD: db.users.find({ username: req.body.username }) — object injection
// ✅ GOOD: always cast to string
const safeUsername = String(req.body.username);
db.users.find({ username: safeUsername });

// ✅ LDAP Injection — escape special chars
function escapeLdap(input: string): string {
  return input.replace(/[\\*\(\)\x00]/g, (c) => `\\${c.charCodeAt(0).toString(16)}`);
}

// ✅ Command Injection — never use exec/shell with user input
import { execFile } from 'child_process'; // NOT exec()
execFile('/usr/bin/convert', [safeFilename, outputPath], callback);
```

---

### A04 — Insecure Design

**Controls:**
- Perform STRIDE threat modeling before implementation
- Write security requirements before writing code
- Use security design patterns (e.g., secure defaults, fail-safe)
- Conduct design-level security review for any feature touching auth, data, or payments
- Implement business logic abuse controls (e.g., one coupon per account, not just per request)

---

### A05 — Security Misconfiguration

```typescript
// ✅ Production hardening checklist (enforced via code)

// Remove default/sample endpoints
if (process.env.NODE_ENV === 'production') {
  app.disable('x-powered-by');           // Don't reveal Express
  // Ensure no /swagger-ui in prod unless explicitly allowed
  // Ensure no debug endpoints (/health/detailed, /metrics) are public
}

// ✅ CORS — explicit allowlist, never wildcard in production
import cors from 'cors';
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || '').split(',');

app.use(cors({
  origin: (origin, cb) => {
    if (!origin || ALLOWED_ORIGINS.includes(origin)) return cb(null, true);
    cb(new Error('CORS policy violation'));
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-CSRF-Token'],
}));
```

---

### A06 — Vulnerable and Outdated Components

```bash
# Run in CI — fail build if high/critical vulnerabilities found
npm audit --audit-level=high

# Automated dependency updates (add to CI weekly)
npx npm-check-updates -u

# License compliance check
npx license-checker --onlyAllow 'MIT;Apache-2.0;BSD-3-Clause;ISC'

# Container scanning
docker scan myapp:latest
trivy image myapp:latest
```

**Policy:**
- No dependencies with known CRITICAL CVEs in production
- HIGH CVEs must be patched within 14 days
- Pin dependency versions in `package-lock.json` / `requirements.txt`
- Review new dependencies before adding (supply chain attack prevention)

---

### A07 — Identification and Authentication Failures → See Section 4

### A08 — Software and Data Integrity Failures

```typescript
// ✅ Verify integrity of downloaded artifacts
import { createHash } from 'crypto';
import { readFileSync } from 'fs';

function verifyChecksum(filePath: string, expectedSha256: string): boolean {
  const hash = createHash('sha256').update(readFileSync(filePath)).digest('hex');
  return hash === expectedSha256;
}

// ✅ Webhook signature verification (e.g., Stripe, GitHub)
import { timingSafeEqual } from 'crypto';

function verifyWebhookSignature(
  payload: Buffer,
  signature: string,
  secret: string
): boolean {
  const expected = createHash('sha256')
    .update(secret)
    .update(payload)
    .digest('hex');
  // Use timing-safe compare to prevent timing attacks
  return timingSafeEqual(Buffer.from(signature), Buffer.from(expected));
}
```

---

### A09 — Security Logging and Monitoring Failures → See Section 12

### A10 — Server-Side Request Forgery (SSRF)

```typescript
import { URL } from 'url';
import ipRangeCheck from 'ip-range-check';

const BLOCKED_IP_RANGES = [
  '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16',
  '127.0.0.0/8', '169.254.0.0/16', '::1/128', 'fc00::/7'
];

async function safeRequest(userSuppliedUrl: string): Promise<Response> {
  const parsed = new URL(userSuppliedUrl);

  // Only allow http/https schemes
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error('Invalid scheme');
  }

  // Resolve hostname and block internal IPs
  const { address } = await dns.lookup(parsed.hostname);
  if (ipRangeCheck(address, BLOCKED_IP_RANGES)) {
    throw new Error('SSRF: Internal address blocked');
  }

  return fetch(userSuppliedUrl, { redirect: 'error' }); // block redirects
}
```

---

## 4. Authentication & JWT Hardening

### Password Policy

```typescript
import zxcvbn from 'zxcvbn'; // NIST-aligned strength estimation

const PASSWORD_POLICY = {
  minLength: 12,
  requireUppercase: true,
  requireLowercase: true,
  requireDigit: true,
  requireSpecial: true,
  minStrengthScore: 3,       // zxcvbn score 0-4
  bcryptRounds: 12,
  maxLength: 128,            // Prevent bcrypt DoS
};

function validatePassword(password: string): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  if (password.length < PASSWORD_POLICY.minLength)
    errors.push(`Minimum ${PASSWORD_POLICY.minLength} characters required`);
  if (password.length > PASSWORD_POLICY.maxLength)
    errors.push('Password too long');
  const result = zxcvbn(password);
  if (result.score < PASSWORD_POLICY.minStrengthScore)
    errors.push(`Weak password: ${result.feedback.suggestions.join(', ')}`);
  return { valid: errors.length === 0, errors };
}
```

### JWT — Secure Implementation

```typescript
import jwt from 'jsonwebtoken';
import { randomUUID } from 'crypto';

const ACCESS_TOKEN_TTL  = '15m';
const REFRESH_TOKEN_TTL = '7d';

// ✅ Token payload — minimal, no sensitive data
interface TokenPayload {
  sub: string;          // user ID
  jti: string;          // unique token ID (for revocation)
  iat: number;
  exp: number;
  roles: string[];      // roles, not permissions (keep payload small)
  sessionId: string;
}

// ✅ Sign with RS256 (asymmetric) — preferred over HS256 in multi-service
// If single service: HS256 with 256-bit secret is acceptable
function issueTokens(userId: string, roles: string[], sessionId: string) {
  const jti = randomUUID();
  const accessToken = jwt.sign(
    { sub: userId, jti, roles, sessionId },
    process.env.JWT_PRIVATE_KEY!,
    { algorithm: 'RS256', expiresIn: ACCESS_TOKEN_TTL, issuer: 'api.yourapp.com' }
  );

  const refreshJti = randomUUID();
  const refreshToken = jwt.sign(
    { sub: userId, jti: refreshJti, sessionId },
    process.env.JWT_REFRESH_PRIVATE_KEY!,
    { algorithm: 'RS256', expiresIn: REFRESH_TOKEN_TTL, issuer: 'api.yourapp.com' }
  );

  return { accessToken, refreshToken, jti, refreshJti };
}

// ✅ Verify with full options — never skip
function verifyToken(token: string): TokenPayload {
  return jwt.verify(token, process.env.JWT_PUBLIC_KEY!, {
    algorithms: ['RS256'],         // explicit allowlist — NEVER allow 'none'
    issuer: 'api.yourapp.com',
    complete: false,
  }) as TokenPayload;
}
```

### Refresh Token Rotation with Reuse Detection

```typescript
// Redis-backed token store for revocation and rotation
import { Redis } from 'ioredis';
const redis = new Redis(process.env.REDIS_URL!);

const TOKEN_PREFIX = 'rt:';
const REVOKED_PREFIX = 'revoked_jti:';

async function rotateRefreshToken(oldRefreshToken: string) {
  let payload: TokenPayload;
  try {
    payload = verifyToken(oldRefreshToken);
  } catch {
    throw new Error('Invalid refresh token');
  }

  // ✅ Reuse detection — if token already revoked, invalidate entire session
  const isRevoked = await redis.exists(`${REVOKED_PREFIX}${payload.jti}`);
  if (isRevoked) {
    // Token reuse detected — possible theft — kill entire session
    await redis.del(`session:${payload.sessionId}`);
    throw new Error('Refresh token reuse detected — session terminated');
  }

  // Revoke the old token
  await redis.setex(`${REVOKED_PREFIX}${payload.jti}`, 7 * 24 * 3600, '1');

  // Issue new tokens
  const { accessToken, refreshToken, jti, refreshJti } = issueTokens(
    payload.sub, payload.roles ?? [], payload.sessionId
  );

  // Store new refresh token reference
  await redis.setex(`${TOKEN_PREFIX}${refreshJti}`, 7 * 24 * 3600, payload.sub);

  return { accessToken, refreshToken };
}

// ✅ Logout — revoke both tokens
async function logout(accessJti: string, refreshJti: string) {
  await Promise.all([
    redis.setex(`${REVOKED_PREFIX}${accessJti}`, 15 * 60, '1'),
    redis.setex(`${REVOKED_PREFIX}${refreshJti}`, 7 * 24 * 3600, '1'),
  ]);
}
```

### Multi-Factor Authentication (MFA)

```typescript
import speakeasy from 'speakeasy';
import QRCode from 'qrcode';

// ✅ TOTP setup (Google Authenticator / Authy compatible)
async function setupMFA(userId: string, userEmail: string) {
  const secret = speakeasy.generateSecret({
    name: `YourApp (${userEmail})`,
    length: 20,
  });

  // Store encrypted secret — never plaintext
  await db.mfaSecret.upsert({
    where: { userId },
    create: { userId, secret: encryptField(secret.base32, getMasterKey()) },
    update: { secret: encryptField(secret.base32, getMasterKey()) },
  });

  const otpauthUrl = secret.otpauth_url!;
  const qrDataUrl = await QRCode.toDataURL(otpauthUrl);
  return { qrDataUrl, backupCodes: generateBackupCodes(userId) };
}

function verifyMFA(userId: string, token: string, encryptedSecret: string): boolean {
  const secret = decryptField(encryptedSecret, getMasterKey());
  return speakeasy.totp.verify({
    secret,
    encoding: 'base32',
    token,
    window: 1,  // allow 30s clock drift only
  });
}
```

---

## 5. Session Management

```typescript
import session from 'express-session';
import RedisStore from 'connect-redis';

// ✅ Secure session configuration
app.use(session({
  store: new RedisStore({ client: redis }),
  secret: process.env.SESSION_SECRET!,   // 256-bit random secret
  name: '__Host-sid',                    // __Host- prefix = most secure
  resave: false,
  saveUninitialized: false,
  rolling: true,                         // reset TTL on activity
  cookie: {
    secure: true,        // HTTPS only
    httpOnly: true,      // No JS access
    sameSite: 'strict',  // CSRF protection
    maxAge: 15 * 60 * 1000, // 15 minutes idle timeout
    domain: undefined,   // Don't set domain — limits to exact hostname
    path: '/',
  },
}));

// ✅ Session fixation prevention — regenerate after login
app.post('/login', async (req, res) => {
  const user = await authenticateUser(req.body.email, req.body.password);
  if (!user) return res.status(401).json({ error: 'Invalid credentials' });

  // Destroy old session, create new one
  req.session.regenerate((err) => {
    if (err) return next(err);
    req.session.userId = user.id;
    req.session.roles  = user.roles;
    req.session.loginAt = Date.now();
    req.session.ip = req.ip;
    req.session.ua = req.headers['user-agent'];
    res.json({ success: true });
  });
});

// ✅ Session binding — detect hijacking
function validateSession(req: Request, res: Response, next: NextFunction) {
  if (!req.session.userId) return res.status(401).json({ error: 'Unauthenticated' });

  // Detect session hijacking via IP/UA mismatch (optional — careful with mobile)
  if (req.session.ip && req.session.ip !== req.ip) {
    req.session.destroy(() => {});
    return res.status(401).json({ error: 'Session invalidated — IP mismatch' });
  }
  next();
}

// ✅ Absolute session timeout — force re-login after max duration
const MAX_SESSION_AGE = 8 * 60 * 60 * 1000; // 8 hours
function enforceAbsoluteTimeout(req, res, next) {
  if (req.session.loginAt && Date.now() - req.session.loginAt > MAX_SESSION_AGE) {
    req.session.destroy(() => {});
    return res.status(401).json({ error: 'Session expired — please log in again' });
  }
  next();
}
```

---

## 6. Authorization & Access Control

### RBAC Implementation

```typescript
// Role → Permission mapping (stored in DB, cached in Redis)
const ROLE_PERMISSIONS: Record<string, string[]> = {
  'super_admin': ['*'],
  'admin':       ['users:read', 'users:write', 'orders:read', 'orders:write', 'reports:read'],
  'manager':     ['orders:read', 'orders:write', 'reports:read'],
  'user':        ['orders:read', 'profile:read', 'profile:write'],
  'readonly':    ['orders:read', 'profile:read'],
};

function hasPermission(userRoles: string[], required: string): boolean {
  return userRoles.some(role => {
    const perms = ROLE_PERMISSIONS[role] ?? [];
    return perms.includes('*') || perms.includes(required);
  });
}

// ✅ Middleware factory
function authorize(permission: string) {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!req.user) return res.status(401).json({ error: 'Unauthenticated' });
    if (!hasPermission(req.user.roles, permission)) {
      securityLogger.warn('Authorization failure', {
        userId: req.user.id,
        permission,
        path: req.path,
        ip: req.ip,
      });
      return res.status(403).json({ error: 'Forbidden' });
    }
    next();
  };
}

// Usage
app.delete('/api/users/:id', authenticate, authorize('users:write'), deleteUser);
```

### Row-Level Security (ABAC)

```typescript
// Policy engine — attribute-based access
type ResourcePolicy = (user: AuthUser, resource: any) => boolean;

const RESOURCE_POLICIES: Record<string, ResourcePolicy> = {
  'order:read':   (user, order) => user.id === order.userId || hasPermission(user.roles, 'orders:read'),
  'order:write':  (user, order) => user.id === order.userId && order.status === 'draft',
  'invoice:read': (user, invoice) => user.organizationId === invoice.orgId,
};

function canAccess(user: AuthUser, action: string, resource: any): boolean {
  const policy = RESOURCE_POLICIES[action];
  if (!policy) return false;
  return policy(user, resource);
}
```

---

## 7. Input Validation & Injection Prevention

### Comprehensive Validation Strategy

```typescript
import Joi from 'joi';
import DOMPurify from 'isomorphic-dompurify';
import validator from 'validator';

// ✅ Schema-first validation — validate BEFORE any processing
const schemas = {
  createUser: Joi.object({
    email:    Joi.string().email({ tlds: { allow: true } }).max(254).required(),
    password: Joi.string().min(12).max(128).required(),
    name:     Joi.string().min(1).max(100).pattern(/^[\p{L}\s'-]+$/u).required(),
    phone:    Joi.string().pattern(/^\+?[1-9]\d{7,14}$/).optional(),
  }),

  pagination: Joi.object({
    page:  Joi.number().integer().min(1).max(10000).default(1),
    limit: Joi.number().integer().min(1).max(100).default(20),
    sort:  Joi.string().valid('asc', 'desc').default('asc'),
    // ✅ Only allow specific sort fields — never pass raw to ORDER BY
    sortBy: Joi.string().valid('createdAt', 'name', 'email').default('createdAt'),
  }),
};

function validate<T>(schema: Joi.Schema, data: unknown): T {
  const { error, value } = schema.validate(data, {
    abortEarly: false,
    stripUnknown: true,   // Remove unknown fields — mass assignment prevention
    convert: false,       // No type coercion surprises
  });
  if (error) throw new ValidationError(error.details.map(d => d.message));
  return value as T;
}

// ✅ HTML sanitization — for user-generated rich content
function sanitizeHtml(dirty: string): string {
  return DOMPurify.sanitize(dirty, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br'],
    ALLOWED_ATTR: ['href'],
    ALLOWED_URI_REGEXP: /^https?:/i,    // no javascript:, data:
    FORCE_BODY: true,
  });
}

// ✅ File upload validation
const ALLOWED_MIME_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'application/pdf']);
const MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024; // 5MB

async function validateFileUpload(file: Express.Multer.File): Promise<void> {
  // Check MIME type (from magic bytes, not extension)
  const { fileTypeFromBuffer } = await import('file-type');
  const detected = await fileTypeFromBuffer(file.buffer);

  if (!detected || !ALLOWED_MIME_TYPES.has(detected.mime)) {
    throw new ValidationError('Invalid file type');
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    throw new ValidationError('File too large');
  }
  // Sanitize filename — prevent path traversal
  const safeName = path.basename(file.originalname).replace(/[^a-z0-9._-]/gi, '_');
  file.originalname = safeName;
}
```

### Path Traversal Prevention

```typescript
import path from 'path';

const BASE_UPLOAD_DIR = path.resolve('/app/uploads');

function safeFilePath(userInput: string): string {
  // Resolve and verify path stays within base directory
  const resolved = path.resolve(BASE_UPLOAD_DIR, path.basename(userInput));
  if (!resolved.startsWith(BASE_UPLOAD_DIR + path.sep)) {
    throw new Error('Path traversal attempt detected');
  }
  return resolved;
}
```

---

## 8. Data Protection & Cryptography

### Approved Algorithms Reference

| Use Case | Approved | Prohibited |
|---|---|---|
| **Password hashing** | bcrypt (cost≥12), Argon2id | MD5, SHA1, SHA256, plain |
| **Symmetric encryption** | AES-256-GCM, ChaCha20-Poly1305 | DES, 3DES, RC4, AES-ECB |
| **Asymmetric encryption** | RSA-2048+, ECDSA P-256+ | RSA-1024, DSA |
| **Key derivation** | PBKDF2-SHA256 (600k+ iter), Argon2id | MD5-crypt, SHA1-crypt |
| **Hashing (non-password)** | SHA-256, SHA-3, BLAKE2 | MD5, SHA1 |
| **TLS** | TLS 1.2, TLS 1.3 | SSL 2/3, TLS 1.0/1.1 |
| **JWT signing** | RS256, ES256 | none, HS256 (if multi-service) |

### PII / Sensitive Data Handling

```typescript
// ✅ Data classification
enum DataClassification {
  PUBLIC       = 'public',
  INTERNAL     = 'internal',
  CONFIDENTIAL = 'confidential', // PII — name, email, phone
  RESTRICTED   = 'restricted',   // PAN, SSN, PHI, passwords
}

// ✅ Tokenization for PAN (card numbers)
// Store token in DB, real value in HSM/vault
async function tokenizePan(pan: string): Promise<string> {
  const { data } = await vault.write('transit/encrypt/pan', {
    plaintext: Buffer.from(pan).toString('base64'),
  });
  return data.ciphertext; // store this token
}

// ✅ Data masking for logs and responses
function maskEmail(email: string): string {
  const [user, domain] = email.split('@');
  return `${user.slice(0, 2)}****@${domain}`;
}

function maskCard(pan: string): string {
  return `****-****-****-${pan.slice(-4)}`;
}

// ✅ Ensure sensitive fields never reach logs
const USER_RESPONSE_FIELDS = ['id', 'email', 'name', 'roles', 'createdAt'];
function sanitizeUserForResponse(user: any) {
  return Object.fromEntries(
    USER_RESPONSE_FIELDS.map(f => [f, user[f]])
  );
}
```

---

## 9. API Security

### Complete Express Security Stack

```typescript
import express from 'express';
import helmet from 'helmet';
import compression from 'compression';
import hpp from 'hpp';                   // HTTP Parameter Pollution
import mongoSanitize from 'express-mongo-sanitize';

const app = express();

// ✅ Body size limits — prevent payload DoS
app.use(express.json({ limit: '100kb' }));
app.use(express.urlencoded({ extended: true, limit: '100kb' }));

// ✅ HPP — prevent duplicate query params attack
app.use(hpp());

// ✅ NoSQL injection sanitization
app.use(mongoSanitize());

// ✅ Remove technology fingerprints
app.disable('x-powered-by');
app.disable('etag');

// ✅ Request ID for tracing (security logging)
import { v4 as uuid } from 'uuid';
app.use((req, _res, next) => {
  req.id = (req.headers['x-request-id'] as string) || uuid();
  next();
});
```

### API Versioning & Deprecation

```typescript
// ✅ Version in URL path, not header (easier to audit/log/WAF-rule)
app.use('/api/v1', v1Router);
app.use('/api/v2', v2Router);

// Sunset old versions with warnings
app.use('/api/v1', (req, res, next) => {
  res.set('Sunset', 'Sat, 31 Dec 2025 23:59:59 GMT');
  res.set('Deprecation', 'true');
  next();
});
```

### Error Handling — No Information Leakage

```typescript
// ✅ Generic error handler — NEVER expose stack traces in production
app.use((err: Error, req: Request, res: Response, _next: NextFunction) => {
  const requestId = req.id;

  // Log full error internally
  logger.error({ err, requestId, path: req.path, userId: req.user?.id });

  // Send sanitized error to client
  if (err.name === 'ValidationError') {
    return res.status(400).json({ error: 'Validation failed', details: err.message, requestId });
  }
  if (err.name === 'UnauthorizedError') {
    return res.status(401).json({ error: 'Authentication required', requestId });
  }

  // All other errors — generic 500
  res.status(500).json({
    error: 'An internal error occurred',
    requestId,
    // ❌ NEVER: stack: err.stack, message: err.message (in production)
  });
});
```

---

## 10. DDoS & Rate Limiting

```typescript
import rateLimit from 'express-rate-limit';
import RedisStore from 'rate-limit-redis';
import slowDown from 'express-slow-down';

// ✅ Distributed rate limiting (Redis-backed — works across replicas)
const createLimiter = (windowMs: number, max: number, message: string) =>
  rateLimit({
    windowMs,
    max,
    message: { error: message },
    standardHeaders: true,
    legacyHeaders: false,
    store: new RedisStore({ client: redis, prefix: 'rl:' }),
    keyGenerator: (req) => req.ip + ':' + req.path, // per IP per endpoint
    skip: (req) => req.ip === process.env.HEALTH_CHECK_IP, // whitelist health checker
  });

// Endpoint-specific limits
const limiters = {
  // Very strict — brute force target
  login:          createLimiter(15 * 60_000, 5,   'Too many login attempts. Try again in 15 minutes.'),
  register:       createLimiter(60 * 60_000, 3,   'Too many registrations from this IP.'),
  passwordReset:  createLimiter(60 * 60_000, 3,   'Too many password reset requests.'),
  mfaVerify:      createLimiter(15 * 60_000, 10,  'Too many MFA attempts.'),

  // Standard API
  api:            createLimiter(60_000,        100, 'Rate limit exceeded.'),
  publicApi:      createLimiter(60_000,        20,  'Rate limit exceeded.'),

  // Expensive operations
  search:         createLimiter(60_000,        10,  'Too many search requests.'),
  export:         createLimiter(60 * 60_000,   5,   'Too many export requests.'),
  upload:         createLimiter(60_000,        5,   'Too many uploads.'),
};

// ✅ Progressive slow-down (before hard block)
const speedLimiter = slowDown({
  windowMs: 15 * 60_000,
  delayAfter: 50,           // start slowing after 50 requests
  delayMs: (used) => (used - 50) * 100, // add 100ms per request over limit
  maxDelayMs: 5000,
});

app.use('/api/', speedLimiter);
app.use('/api/', limiters.api);
app.post('/api/auth/login', limiters.login);
app.post('/api/auth/register', limiters.register);
app.post('/api/auth/reset-password', limiters.passwordReset);

// ✅ Account lockout after repeated failures
const LOCKOUT_THRESHOLD = 5;
const LOCKOUT_DURATION  = 30 * 60; // 30 minutes

async function recordLoginAttempt(userId: string, success: boolean): Promise<void> {
  const key = `login_attempts:${userId}`;
  if (success) {
    await redis.del(key);
    return;
  }
  const attempts = await redis.incr(key);
  if (attempts === 1) await redis.expire(key, LOCKOUT_DURATION);
  if (attempts >= LOCKOUT_THRESHOLD) {
    await redis.setex(`locked:${userId}`, LOCKOUT_DURATION, '1');
    securityLogger.warn('Account locked', { userId });
    // Notify user via email
  }
}
```

---

## 11. HTTP Security Headers

```typescript
import helmet from 'helmet';

app.use(helmet({
  // ✅ Content Security Policy
  contentSecurityPolicy: {
    useDefaults: false,
    directives: {
      defaultSrc:     ["'self'"],
      scriptSrc:      ["'self'"],                            // no unsafe-inline in prod
      scriptSrcAttr:  ["'none'"],
      styleSrc:       ["'self'", "'unsafe-inline'"],         // inline styles common in SSR
      imgSrc:         ["'self'", 'data:', 'https:'],
      connectSrc:     ["'self'", process.env.API_URL!],
      fontSrc:        ["'self'", 'https:', 'data:'],
      objectSrc:      ["'none'"],
      mediaSrc:       ["'self'"],
      frameSrc:       ["'none'"],
      frameAncestors: ["'none'"],                            // clickjacking
      formAction:     ["'self'"],
      baseUri:        ["'self'"],
      upgradeInsecureRequests: [],
      reportUri:      ['/api/csp-report'],
    },
  },

  // ✅ HSTS — forces HTTPS for 1 year, including subdomains
  strictTransportSecurity: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true,
  },

  // ✅ Prevents MIME sniffing
  noSniff: true,

  // ✅ Referrer Policy
  referrerPolicy: { policy: 'strict-origin-when-cross-origin' },

  // ✅ Permissions Policy — disable unneeded browser features
  permissionsPolicy: {
    features: {
      camera:             [],
      microphone:         [],
      geolocation:        [],
      payment:            [],
      usb:                [],
      interest_cohort:    [],
    },
  },

  // ✅ Cross-Origin policies
  crossOriginEmbedderPolicy: true,
  crossOriginOpenerPolicy:   { policy: 'same-origin' },
  crossOriginResourcePolicy: { policy: 'same-origin' },
}));

// ✅ Additional headers not covered by helmet
app.use((_, res, next) => {
  res.set('X-Content-Type-Options', 'nosniff');
  res.set('X-Frame-Options', 'DENY');
  res.set('X-DNS-Prefetch-Control', 'off');
  res.set('Expect-CT', 'enforce, max-age=86400');
  next();
});
```

---

## 12. Audit Logging & Monitoring

### What to Log (and What NOT to Log)

```typescript
import pino from 'pino';

// ✅ Structured, tamper-evident security logger
const securityLogger = pino({
  name: 'security',
  level: 'info',
  redact: {
    // ✅ NEVER log these fields — even accidentally
    paths: [
      'password', '*.password', 'token', '*.token',
      'secret', 'cvv', 'pan', 'ssn', 'authorization',
      'cookie', '*.cookie', 'req.headers.authorization',
      'req.headers.cookie',
    ],
    remove: true,
  },
  serializers: {
    req: (req) => ({
      id: req.id,
      method: req.method,
      url: req.url,
      ip: req.ip,
      userAgent: req.headers['user-agent'],
    }),
  },
});

// ✅ Security event taxonomy
enum SecurityEvent {
  LOGIN_SUCCESS         = 'auth.login.success',
  LOGIN_FAILURE         = 'auth.login.failure',
  LOGIN_LOCKED          = 'auth.login.locked',
  LOGOUT               = 'auth.logout',
  TOKEN_REFRESH         = 'auth.token.refresh',
  TOKEN_REUSE_DETECTED  = 'auth.token.reuse',
  MFA_CHALLENGE         = 'auth.mfa.challenge',
  MFA_SUCCESS           = 'auth.mfa.success',
  MFA_FAILURE           = 'auth.mfa.failure',
  AUTHZ_FAILURE         = 'authz.denied',
  PASSWORD_CHANGE       = 'user.password.changed',
  PASSWORD_RESET        = 'user.password.reset',
  ACCOUNT_CREATED       = 'user.account.created',
  ACCOUNT_DELETED       = 'user.account.deleted',
  PRIVILEGE_CHANGE      = 'user.privilege.changed',
  DATA_EXPORT           = 'data.export',
  SENSITIVE_ACCESS      = 'data.sensitive.accessed',
  RATE_LIMIT_HIT        = 'security.ratelimit.hit',
  INJECTION_ATTEMPT     = 'security.injection.attempt',
  CSRF_VIOLATION        = 'security.csrf.violation',
  SUSPICIOUS_ACTIVITY   = 'security.suspicious',
}

function logSecurityEvent(
  event: SecurityEvent,
  context: { userId?: string; ip: string; metadata?: Record<string, unknown> }
) {
  securityLogger.info({
    event,
    timestamp: new Date().toISOString(),
    ...context,
  });
}
```

### Alerting Thresholds

| Event | Threshold | Action |
|---|---|---|
| Failed logins (same IP) | 10 in 5 min | Alert + auto-block IP |
| Failed logins (same user) | 5 in 15 min | Lock account + notify user |
| Token reuse detected | 1 | Kill session + alert SOC |
| SSRF attempts | 3 in 1 hour | Alert SOC |
| Injection attempt | 1 | Alert + log WAF rule |
| New admin privilege grant | Always | Alert SOC + require approval |
| Data export > 10k records | Always | Alert + require MFA step-up |

---

## 13. Secret & Config Management

```bash
# ✅ .env.example — commit this (no real values)
NODE_ENV=development
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
REDIS_URL=redis://localhost:6379
JWT_PRIVATE_KEY_PATH=/run/secrets/jwt_private_key
JWT_PUBLIC_KEY_PATH=/run/secrets/jwt_public_key
SESSION_SECRET=<256-bit-random>
ENCRYPTION_KEY_PATH=/run/secrets/encryption_key
ALLOWED_ORIGINS=https://app.yourapp.com,https://admin.yourapp.com

# ✅ .gitignore — MUST have these
.env
.env.local
.env.production
*.pem
*.key
*.p12
*.pfx
secrets/
```

```typescript
// ✅ Fail-fast on missing secrets at startup
const REQUIRED_ENV_VARS = [
  'DATABASE_URL', 'REDIS_URL',
  'JWT_PRIVATE_KEY_PATH', 'JWT_PUBLIC_KEY_PATH',
  'SESSION_SECRET', 'ENCRYPTION_KEY_PATH',
  'ALLOWED_ORIGINS',
];

function validateEnvironment(): void {
  const missing = REQUIRED_ENV_VARS.filter(v => !process.env[v]);
  if (missing.length) {
    throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
  }
  if (process.env.SESSION_SECRET!.length < 32) {
    throw new Error('SESSION_SECRET must be at least 32 characters');
  }
}

validateEnvironment(); // Call before app.listen()
```

**Secret Rotation Policy:**

| Secret | Rotation Frequency |
|---|---|
| JWT signing keys | Every 90 days |
| Encryption master key | Every 365 days (with re-encryption) |
| API keys (third-party) | Every 90 days or on suspected compromise |
| Session secrets | Every 30 days |
| Database credentials | Every 90 days |
| On any compromise | Immediately |

---

## 14. Dependency & Supply Chain Security

```yaml
# .github/workflows/security.yml
name: Security Scan

on: [push, pull_request, schedule]
  schedule:
    - cron: '0 0 * * 1' # Weekly Monday scan

jobs:
  dependency-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: NPM Audit
        run: npm audit --audit-level=high

      - name: Snyk Scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high

      - name: License Check
        run: npx license-checker --onlyAllow 'MIT;Apache-2.0;BSD-3-Clause;ISC;BSD-2-Clause'

      - name: Secrets Scan
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          only-verified: true

  container-scan:
    runs-on: ubuntu-latest
    steps:
      - name: Trivy Container Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'myapp:${{ github.sha }}'
          format: 'sarif'
          severity: 'HIGH,CRITICAL'
          exit-code: '1'
```

---

## 15. Security Requirements Model

### Requirement Template

```
ID:          SR-XXX
Title:       [Short imperative — "Enforce MFA for admin roles"]
Type:        FUNCTIONAL | NON_FUNCTIONAL | CONSTRAINT
Domain:      AUTHENTICATION | AUTHORIZATION | DATA_PROTECTION |
             AUDIT_LOGGING | INPUT_VALIDATION | ERROR_HANDLING |
             SESSION_MANAGEMENT | CRYPTOGRAPHY | NETWORK_SECURITY | AVAILABILITY
Priority:    CRITICAL | HIGH | MEDIUM | LOW
Risk:        [Impact if not implemented]
Threat Refs: [STRIDE threats this mitigates]
Compliance:  [PCI-DSS/HIPAA/GDPR/SOC2/OWASP controls]

Description:
[Clear, testable statement of what the system must do]

Acceptance Criteria:
- [ ] [Verifiable condition 1]
- [ ] [Verifiable condition 2]
- [ ] [Verifiable condition 3]

Test Cases:
1. [Happy path test]
2. [Negative/attack test]
3. [Edge case test]

Definition of Done:
- [ ] Implementation complete
- [ ] Unit + security tests pass
- [ ] Code review by security-aware reviewer
- [ ] Penetration test or DAST scan clean
- [ ] Documentation updated
```

### Sample Requirements by Domain

```
SR-001 | Authentication | CRITICAL
Enforce MFA for all users with admin or elevated roles.
Threat: SPOOFING | Compliance: OWASP V2.2, PCI-DSS 8.3

SR-002 | Session Management | HIGH
Regenerate session ID after successful login to prevent session fixation.
Threat: SPOOFING | Compliance: OWASP V3.4

SR-003 | Data Protection | CRITICAL
Encrypt all PII fields (name, email, phone, DOB) at rest using AES-256-GCM.
Threat: INFORMATION DISCLOSURE | Compliance: GDPR Art.32, PCI-DSS 3.5

SR-004 | Audit Logging | HIGH
Log all authentication events with user ID, IP, timestamp, and outcome.
Threat: REPUDIATION | Compliance: SOC2 CC7.2, PCI-DSS 10.2

SR-005 | Authorization | CRITICAL
Enforce server-side authorization on every API endpoint — no client trust.
Threat: ELEVATION OF PRIVILEGE | Compliance: OWASP V4.1, PCI-DSS 7.1

SR-006 | Availability | HIGH
Rate-limit authentication endpoints to 5 requests per IP per 15 minutes.
Threat: DENIAL OF SERVICE | Compliance: OWASP V11.1

SR-007 | Input Validation | CRITICAL
Validate and sanitize all user input before processing or persistence.
Threat: TAMPERING, INJECTION | Compliance: OWASP V5.1, V5.3
```

---

## 16. Compliance Mapping

| Control Domain | OWASP ASVS | PCI-DSS | HIPAA | GDPR |
|---|---|---|---|---|
| Authentication | V2.1–V2.10 | 8.1–8.6 | 164.312(d) | Art. 32 |
| Session Mgmt | V3.1–V3.7 | 8.1.5–8.2 | 164.312(a) | Art. 32 |
| Authorization | V4.1–V4.3 | 7.1–7.2 | 164.312(a)(1) | Art. 25 |
| Input Validation | V5.1–V5.5 | 6.3.2 | 164.312(c) | Art. 32 |
| Cryptography | V6.1–V6.4 | 3.4–3.6, 4.1 | 164.312(e)(2)(ii) | Art. 32 |
| Error Handling | V7.1–V7.4 | 6.2.4 | — | — |
| Data Protection | V8.1–V8.3 | 3.4, 4.1 | 164.312(a)(2)(iv) | Art. 32, 25 |
| Audit Logging | V7.1–V7.4 | 10.1–10.7 | 164.312(b) | Art. 30 |
| SSRF | V10.1 | 6.3 | — | — |
| HTTPS/TLS | V9.1–V9.3 | 4.1 | 164.312(e)(1) | Art. 32 |
| Secrets Mgmt | V6.4 | 6.3 | 164.312(a)(2)(i) | Art. 32 |

---

## 17. Incident Response Playbook

### Severity Classification

| Level | Definition | Response Time |
|---|---|---|
| **SEV-1** | Active breach, data exfiltration, ransomware | 15 minutes |
| **SEV-2** | Credential compromise, unauthorized admin access | 1 hour |
| **SEV-3** | Suspicious activity, anomaly detected | 4 hours |
| **SEV-4** | Policy violation, failed attack attempt | 24 hours |

### Response Steps

```
1. DETECT     → Alert fires (SIEM / monitoring / user report)
2. TRIAGE     → Classify severity, assign incident commander
3. CONTAIN    → Isolate affected systems, revoke compromised credentials
4. ERADICATE  → Remove threat, patch vulnerability, rotate secrets
5. RECOVER    → Restore from clean backup, verify integrity
6. REVIEW     → Post-mortem, update controls, update this document
```

### Immediate Actions on Token/Session Compromise

```typescript
// ✅ Emergency: revoke ALL sessions for a user
async function revokeAllUserSessions(userId: string): Promise<void> {
  // 1. Get all active refresh tokens
  const tokenKeys = await redis.keys(`rt:user:${userId}:*`);
  // 2. Revoke all of them
  if (tokenKeys.length) await redis.del(...tokenKeys);
  // 3. Add user to global revocation list (checked in auth middleware)
  await redis.setex(`revoked_user:${userId}`, 24 * 3600, Date.now().toString());
  // 4. Log the event
  logSecurityEvent(SecurityEvent.SUSPICIOUS_ACTIVITY, { userId, ip: 'system', metadata: { reason: 'emergency_revocation' } });
}

// ✅ Check revocation in auth middleware
async function isUserRevoked(userId: string): Promise<boolean> {
  return (await redis.exists(`revoked_user:${userId}`)) === 1;
}
```

---

## 18. Security Checklist — Definition of Done

Before any feature touching auth, data, or APIs ships:

### Authentication & Session
- [ ] Passwords hashed with bcrypt (cost ≥ 12) or Argon2id
- [ ] JWT signed with RS256; `none` algorithm explicitly rejected
- [ ] Access tokens ≤ 15 min TTL
- [ ] Refresh token rotation with reuse detection implemented
- [ ] Session regenerated after login (fixation prevention)
- [ ] Session has idle timeout (15 min) and absolute timeout (8 hr)
- [ ] MFA available for sensitive operations
- [ ] Account lockout after 5 failed attempts

### Authorization
- [ ] Every endpoint has explicit `authenticate` + `authorize` middleware
- [ ] Resource ownership checked (no IDOR vulnerabilities)
- [ ] Server-side permission check — no client-supplied role trust
- [ ] Privilege changes require re-authentication

### Input & Output
- [ ] All inputs validated with schema (Joi or equivalent)
- [ ] `stripUnknown: true` on all validation schemas
- [ ] Parameterized queries only — no string concatenation with SQL
- [ ] User-generated HTML sanitized with DOMPurify
- [ ] File uploads validated by MIME type (magic bytes), size, and filename
- [ ] Error responses do not expose stack traces, SQL, or internal paths

### Data Protection
- [ ] PII/PAN encrypted at rest (AES-256-GCM)
- [ ] All traffic over TLS 1.2+ (1.3 preferred)
- [ ] Sensitive fields redacted from logs
- [ ] Data masking applied in non-production environments

### Headers & CSRF
- [ ] Helmet configured with full CSP
- [ ] HSTS enabled with preload
- [ ] CORS allowlist is explicit (no wildcards)
- [ ] CSRF protection on all state-changing endpoints

### Rate Limiting
- [ ] Auth endpoints rate-limited (≤ 5 req/15 min per IP)
- [ ] General API rate-limited (≤ 100 req/min per IP)
- [ ] Expensive endpoints (search, export, upload) separately limited

### Secrets & Config
- [ ] No secrets in source code
- [ ] All secrets in environment variables or vault
- [ ] `.env` in `.gitignore`
- [ ] Startup fails fast if required env vars missing

### Logging & Monitoring
- [ ] Auth events logged (success, failure, lockout)
- [ ] Authorization failures logged with user ID + resource
- [ ] No passwords, tokens, or PII in logs
- [ ] Alerts configured for anomalous patterns

### Dependencies
- [ ] `npm audit` passes at high/critical level
- [ ] No GPL-incompatible licenses (if proprietary)
- [ ] Secrets scan (`trufflehog`) passes
- [ ] Container scan passes (no CRITICAL CVEs)

### Tests
- [ ] Unit tests for all auth/authz logic
- [ ] Negative tests: unauthenticated request → 401
- [ ] Negative tests: unauthorized request → 403
- [ ] Injection tests: SQL, XSS, path traversal
- [ ] Rate limit tests verify limits are enforced

---

*Version: 2.0.0 | Last updated: 2026-03-27*
*Owner: Security Engineering*
*Review cycle: Quarterly or after any security incident*
*Compliance: OWASP ASVS L2 | PCI-DSS v4 | GDPR | SOC 2 Type II*