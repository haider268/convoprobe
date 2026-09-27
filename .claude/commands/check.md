Run every quality gate for convoprobe and report the result of each one.

Run from the convoprobe folder. If `uv` is not on PATH, use
`C:\Users\maria\.local\bin\uv.exe`.

1. `uv run ruff check`
2. `uv run ruff format --check`
3. `uv run mypy src`
4. `uv run pytest`

Report pass or fail per gate, the test count, and total coverage. Do not fix
anything unless asked.
