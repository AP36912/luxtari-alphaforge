# Security

The public execution surface fails closed. Only `paper`, `dry_run`, and `sim`
parse. Removed live, MT5, bridge, credential, lifecycle, and automation keys are
errors rather than ignored input.

`kill_switch_enabled` means monitoring is enabled; it does not halt by itself.
`emergency_stop=true` or a present kill-switch file blocks execution. Stale
signals, duplicate cooldown, daily loss, allow/deny lists, size, notional, and
concurrency limits are enforced by the gateway.

Serialized model loading is denied by default because pickle/joblib and model
archives can execute or instantiate unsafe content. Enabling it requires a
non-empty `allowed_roots` list and a SHA-256 digest verified through a trusted,
out-of-band source. Artifact paths must resolve beneath an allowed root, and any
symlink in the artifact directory tree is rejected.

Before deserialization, every regular artifact file is copied into a private,
read-only temporary snapshot. The snapshot is hashed and approved, and loaders
read only from that snapshot. The snapshot is deleted after the model is fully
loaded, preventing modification or substitution of the original artifact between
integrity verification and deserialization. An allowed path alone is not
sufficient.

```python
from pathlib import Path

from models.artifact_trust import (
    ModelArtifactTrustPolicy,
    clear_model_artifact_trust_policy,
    set_model_artifact_trust_policy,
)
from models.registry import load_model

model_root = Path("artifacts/models").resolve()
model_path = model_root / "production-signal"
expected_sha256 = "<sha256-from-trusted-release-metadata>"

set_model_artifact_trust_policy(
    ModelArtifactTrustPolicy(
        allow_serialized_load=True,
        allowed_sha256=frozenset({expected_sha256}),
        allowed_roots=(model_root,),
    )
)
try:
    model = load_model(model_path)
finally:
    clear_model_artifact_trust_policy()
```

Do not approve a digest obtained from an untrusted artifact itself. Distribute
approved digests separately from model files, and clear the policy when loading
is complete. Do not commit secrets. There is no real-broker implementation;
IBKR remains future work.
