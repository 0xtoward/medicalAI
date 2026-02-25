# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
Medical ML research project for predicting thyroid treatment outcomes after radioactive iodine (I-131) therapy. Each `.py` file is an independent ML experiment script (no web server, no database, no Docker). The Jupyter notebook `solve.ipynb` combines several analyses.

### Data file
The dataset `700.xlsx` is **gitignored** (sensitive medical records). Scripts cannot run without it. For testing, generate synthetic data with `python3 generate_test_data.py`.

### Running scripts
All scripts are standalone — run them directly with `python3 <script>.py` from the workspace root. For example:
- `python3 rf.py` — Random Forest
- `python3 mlp.py --no-grid` — MLP (fast mode; omit `--no-grid` for full grid search)
- `python3 hybrid.py` — Hybrid LSTM
- `python3 causal_infer.py` — Causal Forest (econml)
- `python3 verify.py` — 3-month Sniper MLP
- `python3 all.py` — Grand comparison of 5 model architectures

### Known compatibility issues
- **`causalnex`** requires Python <3.11. The system has Python 3.12, so `causal.py` (NOTEARS DAG discovery) cannot run. All other scripts work fine on 3.12.
- **`xgboost.py`** has a naming conflict — the filename shadows the `xgboost` library. Running `python3 xgboost.py` will fail with `AttributeError: module 'xgboost' has no attribute 'XGBClassifier'`. To work around this, rename or copy the file (e.g., `cp xgboost.py xgb_experiment.py && python3 xgb_experiment.py`).
- **`tabpfn`** (used in `pfn.py` and `solve.ipynb`) requires HuggingFace authentication to download the gated model. Run `huggingface-cli login` first, or set `HF_TOKEN` env var after accepting model terms at https://huggingface.co/Prior-Labs/tabpfn_2_5.

### Linting
No linter configuration exists in the repo. You can run `ruff check .` if needed (ruff is installed as a tabpfn dependency).

### PATH
User-installed pip binaries are in `~/.local/bin`. Ensure `export PATH="$HOME/.local/bin:$PATH"` is set.
