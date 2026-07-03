# Contributing

Thanks for your interest! This repository is a completed internship project,
so the bar for changes is "keep it reproducible and honest."

## Ground rules

- **Reproducibility first.** Every number in the README and documents comes
  from `python main.py` at seed 42. If your change affects any output,
  rerun the pipeline and update the affected evidence files in `outputs/`
  and figures in `graphs/` together with the code — never edit a measured
  value by hand.
- **No data in git.** Do not commit the dataset CSVs, the generated
  `clean_emails.csv`, virtual environments, or the two large model dumps
  (all covered by `.gitignore`).
- **Keep it modular.** New functionality belongs in a module under `src/`
  with a short docstring (purpose / input / output), wired into `main.py`.

## Workflow

1. Open an issue describing the change first (bug, improvement, question).
2. Fork, create a branch, make the change.
3. Verify locally: `python -m py_compile main.py src/*.py`, then a full
   `python main.py` run if outputs are affected.
4. Open a pull request that says what changed and how you verified it.

Small fixes (typos, broken links, clearer docs) are welcome as direct PRs.
