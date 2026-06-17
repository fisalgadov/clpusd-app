# Security Policy (Phase 1)

This repository is a confidential CLP/USD multi-horizon forecast app. Phase 1 controls must remain minimal, reversible, and non-functional.

1. No real FX/model/forecast data in Git unless intentionally tracked and reviewed.
2. No credentials, API keys, or tokens in Git.
3. App code, Git LFS configuration, and production model artifacts are protected.
4. Agents must not touch `app.py`, `requirements.txt`, `.gitattributes`, or `model_CLPUSD_*.pkl`.
5. Phase 1 must remain functionally identical.
6. Commits should pass targeted pre-commit checks.
