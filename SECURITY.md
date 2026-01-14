# Security Policy

EmoLoop processes real-time physiological data (ECG, EDA, respiration) and machine learning predictions within a multiprocessing architecture for research purposes. Security focuses on research integrity, data privacy, and system reliability.

## Supported Versions

EmoLoop releases research versions only. No formal security updates are provided.

| Version | Supported    |
| ------- | ------------ |
| main    | ✅           |
| dev     | ❌           |
| tags    | Case-by-case |

## Reporting Vulnerabilities

Report security issues through GitHub Issues with the `security` label. Provide:

- Affected component (Stream, Sync, Manager, ModelTrainer, Processing)
- Minimal reproducible steps
- Impact on data processing or multiprocessing locks/queues
- LSL stream or physiological signal interactions (if relevant)

Issues receive research-appropriate review during active development. Critical research integrity issues receive priority assessment.

## Scope

This policy covers the core multiprocessing framework only:

```yml
✅  Included: LSL stream handling, buffer synchronisation, feature extraction,
  online model retraining, inter-process communication (queues/locks)

❌  Excluded:
  External LSL streams, VR environments, third-party signal processing,
  user self-report interfaces, deployment infrastructure
```

## Expected Response

Vulnerability reports trigger:

1. **Research Review**: Technical assessment of impact on core framework
2. **Code Analysis**: Verification against physiological data pipeline and ML model integrity
3. **Resolution Planning**: Research-appropriate mitigation or workaround
4. **Documentation**: Updated guidance in README.md architecture diagrams

No immediate triage or fixed response times are guaranteed. Research development cycles determine resolution priority.
