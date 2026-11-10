# Known Limitations

- No live brokers, IBKR, MT5, Windows bridge, credentials, or network market data.
- No proprietary alpha or strategy-setup gating.
- Partial fills and multi-position inventory are unsupported.
- `dry_run` is a deterministic no-inventory adapter, not a broker sandbox.
- Optional PyArrow/LightGBM/XGBoost/Torch dependencies are not core.
- The hash lock targets CPython 3.12 on Linux x86_64.
- Serialized model loading is denied by default; joblib/pickle remain unsafe formats
  even under explicit local trust (see `docs/SECURITY.md`).
- Verified paper/backtest parity is limited to discrete `-1/0/+1` signals,
  `fixed_shares`, `contract_size=1`, and shared execution assumptions under tested
  policies — not universal simulator parity.
- Synthetic offline reports are reproducibility checks only; they do not validate
  profitability or exchange/broker equivalence.
- `config_digest_sha256` from the synthetic demo includes host platform identity
  and is same-host only; use `input_checksum_sha256` for portable input checks
  (see `docs/OFFLINE_RESEARCH.md`).
- The hash lock was advisory-scanned with pip-audit after upgrading pytest to
  9.0.3; see `docs/STAGE4D_PIP_AUDIT.md`. An audit exit of zero is not a full
  supply-chain attestation.
