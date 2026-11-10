# Dependency Lock — Luxtari AlphaForge

- Python: CPython 3.12 (verified on 3.12.3)
- Platform class: Linux x86_64 / glibc (lock built on a matching host)
- Direct pins: `requirements-stage3.min.txt`
- Hash lock: `requirements-stage3.lock.txt`
- Optional heavy packages: `requirements-optional-ml.txt`

The lock was generated from wheels downloaded with `pip download
--only-binary=:all:` and SHA-256 hashed without installing packages. It includes
resolved transitive dependencies for that platform class. A Stage 4D
`pip-audit==2.10.1` run against `requirements-stage3.lock.txt` with
`--require-hashes` exited 0 with no known vulnerabilities reported. See
`docs/STAGE4D_PIP_AUDIT.md`. This is not a full supply-chain attestation.

Supported public install path:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: --require-hashes -r requirements-stage3.lock.txt
```
