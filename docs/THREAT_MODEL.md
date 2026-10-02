# Threat model (stub, completed in M7)

Purpose is strictly defensive: detect presentation attacks. No attack-generation tooling is
included; the synthetic fixture is generic image degradation for tests.

| Attack | In scope? |
|---|---|
| Printed photo | Yes (target) |
| Screen replay (phone / tablet) | Yes (target) |
| Cut-out / partial face | TBD |
| 3D / silicone mask | **No** |
| Deepfake / virtual-camera injection | **No** |
