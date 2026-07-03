# Security Policy

## Scope

This repository is an educational machine learning project. The trained
model is a coursework artifact for studying text classification — it is
**not** a production email filter and should not be deployed to protect
real mailboxes without independent evaluation on the target mail stream
(see the "corpus membership" caution in `outputs/best_model_analysis.txt`).

## Supported versions

Only the latest release (`1.0.0`) is maintained. There are no backported
security fixes.

## Reporting a vulnerability

If you find a security issue in the code (for example, unsafe file handling
or a dependency vulnerability), please report it privately via GitHub's
**Security > Report a vulnerability** (private security advisory) on this
repository rather than opening a public issue. Reports are reviewed on a
best-effort basis; there is no bug bounty. Dependency alerts from GitHub
(Dependabot) are welcome as pull requests.
