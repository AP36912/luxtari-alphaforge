# Public Architecture

Layers:

1. Data: CSV OHLCV loaders + schema + bars/preprocessing
2. Research: features, labeling (triple-barrier), models, regime, validation
3. Backtest: engine + shared execution assumptions
4. Control: safety validators, order tracker, reconciliation concepts
5. Brokers: PaperBrokerAdapter + DemoBrokerAdapter (sim stub)
6. Reporting/audit/tracking

Future IBKR adapters are out of scope for this export and must plug into
BrokerAdapter without changing research defaults.
