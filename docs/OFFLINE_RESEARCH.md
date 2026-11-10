# Offline research

## Synthetic demo

```bash
.venv/bin/python run_experiment.py --synthetic --seed 42
```

When run from the project root, this writes the CWD-relative path
`outputs/public_offline_seed_42.json` (created under the current working
directory’s `outputs/` folder).

The report includes:
- seed, rows, strategy id
- feature settings used
- execution assumptions (fill basis, proportional costs)
- sizing and initial capital
- `config_digest_sha256` over the hashed config `spec`
- `input_checksum_sha256` of the synthetic OHLCV CSV bytes
- metrics with chronology notes

`config_digest_sha256` hashes the full offline `spec`, which currently includes:
seed, rows, strategy, instrument, timeframe, feature_config, assumptions,
sizing, initial_capital, bars_per_year, cost_notes, external_validity,
metric_definitions, and host `platform_identity` (Python version,
`platform.platform()`, machine, and implementation). Because
`platform_identity` is included, the digest is **same-host** reproducible.
Use `input_checksum_sha256` for portable synthetic-input reproducibility across
machines that share the same seed and generator. Changing narrative fields such
as `metric_definitions` or `external_validity` also changes `config_digest_sha256`.

## External validity

Synthetic bars exist to prove chronology-safe plumbing and reproducibility.
They are **not** evidence of real-market profitability, CME/futures equivalence,
IBKR performance, or a tradable edge.

## Costs and commissions

Backtest equity paths apply configured half-spread, slippage, and commission rate
at fills. Trade-leg summary fields `win_rate` / `profit_factor` are computed from
`TradeLeg.pnl` and **exclude commissions** (spread/slippage still enter via fill
prices). Treat equity-path results as the cost-aware reference for this demo.

## Parity scope

Paper/backtest parity remains narrow. See `docs/SIMULATION.md`.

## Metric definitions

- `n_trades`: number of `TradeLeg` closed-lot records with realized PnL emitted by
  the backtest engine (see `backtest.execution.TradeLeg`).
- `win_rate` / `profit_factor`: from `TradeLeg.pnl` (spread/slippage via fill prices; **exclude commissions**).
- Equity-path returns include all configured proportional spread, slippage, and commission.
