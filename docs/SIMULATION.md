# Simulation

Verified paper/backtest parity is intentionally narrow:

- Discrete target signals `-1/0/+1`
- `fixed_shares` sizing (including values other than 1)
- `contract_size=1`
- Shared `ExecutionAssumptions` (fill basis + proportional costs)
- Mid-path cash, equity, and signed position under the above
- Stop-loss under the explicitly tested `stop_level` long/`bar_close` path in the suite

`fraction_of_equity` is not supported by the `PaperBroker` parity path and fails
explicitly when converting such a `BacktestConfig`. This is **not** a claim of
universal simulator parity across all sizing modes, stop/take-profit matrices,
or contract sizes.

Trade-leg metrics and account realized PnL can attribute commissions differently.
The verified bar-level cash/equity path includes commissions; consumers should not
assume every trade-leg PnL field uses identical commission attribution.

`paper`/`sim` mutate local inventory; `dry_run` never does. Real brokers (including
IBKR) are future work and are not present in this public edition.

## Research claims

Synthetic and paper/sim results are for methodology verification. They do not
establish profitable strategies, CME equivalence, IBKR validation, or universal
parity.
