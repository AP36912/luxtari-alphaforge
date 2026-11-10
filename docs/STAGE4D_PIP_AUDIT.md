# Stage 4D dependency vulnerability audit (historical)

## Change

- Approved upgrade: `pytest==8.3.5` → `pytest==9.0.3`
- New required transitive (pytest 9): `Pygments==2.21.0`
- Lock regenerated: `requirements-stage3.lock.txt`
- Install path: hash-pinned `--only-binary=:all:` into a local project venv

## Audit

Tool: `pip-audit==2.10.1` against `requirements-stage3.lock.txt` with
`--require-hashes`.

- Exit code: **0**
- Result: **No known vulnerabilities found**
- Remaining advisories: **none reported**

This records a completed advisory scan for the locked dependency set on the
documented CPython 3.12 / Linux x86_64 platform. It is not a full supply-chain
attestation.
