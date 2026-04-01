# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.2.x   | ✅ Current |
| 2.1.x   | ❌ End of life |
| < 2.1   | ❌ End of life |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub Issues.**

If you discover a security vulnerability in this project, please report it by:

1. **Email**: Send details to bchiheb598@gmail.com.
2. **Subject line**: `[SECURITY] Predictive Maintenance API — <brief description>`

### What to include

- A clear description of the vulnerability and its potential impact.
- Steps to reproduce the issue (proof of concept if applicable).
- The version(s) affected.
- Any suggested mitigations if you have them.

### What to expect

- **Acknowledgement**: Within 48 hours of your report.
- **Status update**: Within 7 days, with an assessment of severity and likely timeline.
- **Resolution**: Critical vulnerabilities patched within 14 days; others within 30 days.
- **Credit**: Reporters are credited in the release notes unless they prefer to remain anonymous.

## Known Security Considerations

- **API Key Authentication**: The `/predict` endpoint supports optional API key auth via `X-Api-Key`. Key comparison is performed using `secrets.compare_digest` to prevent timing attacks. Set the `API_KEY` environment variable in production.
- **CORS**: Restrict `ALLOWED_ORIGINS` to your actual frontend domain in production. Never use `*` in production.
- **Model File**: `model.pkl` is a serialised Python object. Only load models from trusted sources — never accept model uploads from untrusted users.
- **Input Validation**: All sensor values are validated against known physical ranges before reaching the model. Out-of-range inputs return HTTP 422.
