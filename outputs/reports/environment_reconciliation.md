# Environment reconciliation and portability repair

The pre-Phase-E `requirements.txt` did not match the runtime recorded in the locked Phase-D run. Phase E reconciled the core versions to the runtime that reproduced the locked outputs. The final publication repair separates two purposes:

- `requirements-lock.txt` is the **portable install lock**. It contains only the project’s direct/transitive Python dependencies, pinned to versions used by the reproducing runtime. It contains no local `file:///` paths and no OpenAI/runtime-specific packages.
- `environment_snapshot.txt` retains the original complete `pip freeze --all` snapshot solely for provenance. It may contain host/runtime-specific entries and is **not** an installable environment specification.
- `requirements-data-acquisition.txt` remains separate because final Phase-D/E reproduction uses saved Phase-B data and does not require live acquisition.

Clean install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

This packaging repair changes no locked model, prediction, metric, robustness result, or conclusion.
