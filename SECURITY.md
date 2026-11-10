# Security Policy

Luxtari AlphaForge is an offline research and simulation framework. Only
`paper`, `dry_run`, and `sim` execution modes are accepted. Live broker
adapters, MT5/Windows bridges, credential fields, and network market-data
ingestion are out of scope for this edition.

## Detailed controls

See [`docs/SECURITY.md`](docs/SECURITY.md) for fail-closed execution rules,
kill-switch behavior, and trusted model-artifact loading requirements.

## Reporting a vulnerability

If you discover a security issue in this tree:

1. Do **not** open a public issue that includes exploit details, credentials,
   or private host paths.
2. Prefer a private report to the project operators through whatever channel
   they designate when a public forge or contact path is published.
3. Until a public contact channel exists, keep findings local to operators of
   this repository and share only what is needed for remediation.

Do not invent or publish personal mailbox addresses in this file. Do not commit
secrets, broker credentials, or private host paths.
