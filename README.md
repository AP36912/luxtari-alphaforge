# Luxtari AlphaForge

**Luxtari AlphaForge** (`luxtari-alphaforge`) is an Apache-2.0 offline quantitative
research and simulation framework: causal features, triple-barrier labeling,
model utilities, chronology-safe backtesting, paper/sim execution, and
reproducible reports.

Proprietary alpha and real-broker connectivity are excluded. The `live/` package
contains offline paper/simulation helpers only; IBKR and other real brokers are
future work.

Version **v0.1.0** is a local research baseline. Public remotes, forge
publication, and release tags remain operator-gated.

## Quick start

Create a local virtual environment from the hash-pinned lock, then validate:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: --require-hashes -r requirements-stage3.lock.txt
.venv/bin/pytest -q
.venv/bin/python run_experiment.py --synthetic --seed 42
.venv/bin/python run_experiment.py --config config/templates/minimal_experiment.yaml --validate-only
```

Core direct pins are in `requirements-stage3.min.txt`; the platform lock is
`requirements-stage3.lock.txt`. Heavy libraries are documented separately in
`requirements-optional-ml.txt`.

## Execution boundary

Only `paper`, `dry_run`, and `sim` are accepted. `paper` and `sim` use local
`PaperBroker`; `dry_run` records a deterministic no-inventory submission. Live
flags, MT5/bridge settings, broker credentials, and demo automation keys are
rejected during config parsing. IBKR and every real broker are future work.
Verified backtest/paper parity is limited to discrete `-1/0/+1` signals,
`fixed_shares`, `contract_size=1`, and shared execution assumptions (including
stops under tested policies); see `docs/SIMULATION.md`.

## Unsupported operations

No network market data, real order submission, partial fills, broker account
reconciliation, or proprietary strategy gating. See `docs/KNOWN_LIMITATIONS.md`.

## Identity and packaging

| Item | Value |
|---|---|
| Display name | Luxtari AlphaForge |
| Future GitHub repository | `luxtari-alphaforge` |
| Future distribution name | `luxtari-alphaforge` |
| Future import namespace | `luxtari_alphaforge` (not applied until a tested migration) |
| Version | `v0.1.0` |

Current Python imports remain the existing package layout for compatibility.
See `docs/PUBLIC_ATTRIBUTION.md` and `docs/PROJECT_IDENTITY.md`.
Security policy: `SECURITY.md` (details in `docs/SECURITY.md`).
