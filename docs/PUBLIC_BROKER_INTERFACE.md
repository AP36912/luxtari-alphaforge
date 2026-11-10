# Public Broker Interface

## Supported modes

- `paper`: local fills and inventory
- `sim`: same local fill engine for deterministic simulation
- `dry_run`: safety evaluation plus no-inventory submission record

`ExecutionGateway.process_risk_approved_bar(OrderRequest)` returns an
`OrderResult`. Rejections set `success=false` and `rejection_reason`. Construction
and parsing reject every other mode.

## Safety order

The gateway evaluates emergency/file kill switches, signal age, daily loss,
symbol policy, position/notional/concurrency limits, and duplicate cooldown
before calling an adapter. `connect`, `disconnect`, and `health_check` are local
adapter lifecycle methods only. Real broker and IBKR support are future work.
