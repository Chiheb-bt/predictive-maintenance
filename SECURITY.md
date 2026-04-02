# Security Policy

## Supported Versions

We currently provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.2.x   | :white_check_mark: |
| < 2.2.0 | :x:                |

## Reporting a Vulnerability

We take the security of Sentinel Industrial API seriously. If you believe you have found a security vulnerability, please report it to us as soon as possible.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please email [security@sentinel-ai.com](mailto:security@sentinel-ai.com) with a description of the issue. We will acknowledge your report within 24–48 hours and provide a timeline for a fix.

## Security Features in this Project

Sentinel implements several industry-standard security practices:
- **Constant-time API Key Comparison**: Prevents timing attacks on the `X-Api-Key` header.
- **Strict Input Validation**: Pydantic and custom range checks prevent payload injection or out-of-bounds model inputs.
- **Environment Isolation**: Sensitive configuration (keys, ports) is strictly managed via environment variables.
- **Dependency Auditing**: CI pipelines include linting and type-checking to ensure code integrity.
